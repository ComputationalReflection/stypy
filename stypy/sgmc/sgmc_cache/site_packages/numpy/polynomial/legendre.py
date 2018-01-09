
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Legendre Series (:mod: `numpy.polynomial.legendre`)
3: ===================================================
4: 
5: .. currentmodule:: numpy.polynomial.polynomial
6: 
7: This module provides a number of objects (mostly functions) useful for
8: dealing with Legendre series, including a `Legendre` class that
9: encapsulates the usual arithmetic operations.  (General information
10: on how this module represents and works with such polynomials is in the
11: docstring for its "parent" sub-package, `numpy.polynomial`).
12: 
13: Constants
14: ---------
15: 
16: .. autosummary::
17:    :toctree: generated/
18: 
19:    legdomain            Legendre series default domain, [-1,1].
20:    legzero              Legendre series that evaluates identically to 0.
21:    legone               Legendre series that evaluates identically to 1.
22:    legx                 Legendre series for the identity map, ``f(x) = x``.
23: 
24: Arithmetic
25: ----------
26: 
27: .. autosummary::
28:    :toctree: generated/
29: 
30:    legmulx              multiply a Legendre series in P_i(x) by x.
31:    legadd               add two Legendre series.
32:    legsub               subtract one Legendre series from another.
33:    legmul               multiply two Legendre series.
34:    legdiv               divide one Legendre series by another.
35:    legpow               raise a Legendre series to an positive integer power
36:    legval               evaluate a Legendre series at given points.
37:    legval2d             evaluate a 2D Legendre series at given points.
38:    legval3d             evaluate a 3D Legendre series at given points.
39:    leggrid2d            evaluate a 2D Legendre series on a Cartesian product.
40:    leggrid3d            evaluate a 3D Legendre series on a Cartesian product.
41: 
42: Calculus
43: --------
44: 
45: .. autosummary::
46:    :toctree: generated/
47: 
48:    legder               differentiate a Legendre series.
49:    legint               integrate a Legendre series.
50: 
51: Misc Functions
52: --------------
53: 
54: .. autosummary::
55:    :toctree: generated/
56: 
57:    legfromroots          create a Legendre series with specified roots.
58:    legroots              find the roots of a Legendre series.
59:    legvander             Vandermonde-like matrix for Legendre polynomials.
60:    legvander2d           Vandermonde-like matrix for 2D power series.
61:    legvander3d           Vandermonde-like matrix for 3D power series.
62:    leggauss              Gauss-Legendre quadrature, points and weights.
63:    legweight             Legendre weight function.
64:    legcompanion          symmetrized companion matrix in Legendre form.
65:    legfit                least-squares fit returning a Legendre series.
66:    legtrim               trim leading coefficients from a Legendre series.
67:    legline               Legendre series representing given straight line.
68:    leg2poly              convert a Legendre series to a polynomial.
69:    poly2leg              convert a polynomial to a Legendre series.
70: 
71: Classes
72: -------
73:     Legendre            A Legendre series class.
74: 
75: See also
76: --------
77: numpy.polynomial.polynomial
78: numpy.polynomial.chebyshev
79: numpy.polynomial.laguerre
80: numpy.polynomial.hermite
81: numpy.polynomial.hermite_e
82: 
83: '''
84: from __future__ import division, absolute_import, print_function
85: 
86: import warnings
87: import numpy as np
88: import numpy.linalg as la
89: 
90: from . import polyutils as pu
91: from ._polybase import ABCPolyBase
92: 
93: __all__ = [
94:     'legzero', 'legone', 'legx', 'legdomain', 'legline', 'legadd',
95:     'legsub', 'legmulx', 'legmul', 'legdiv', 'legpow', 'legval', 'legder',
96:     'legint', 'leg2poly', 'poly2leg', 'legfromroots', 'legvander',
97:     'legfit', 'legtrim', 'legroots', 'Legendre', 'legval2d', 'legval3d',
98:     'leggrid2d', 'leggrid3d', 'legvander2d', 'legvander3d', 'legcompanion',
99:     'leggauss', 'legweight']
100: 
101: legtrim = pu.trimcoef
102: 
103: 
104: def poly2leg(pol):
105:     '''
106:     Convert a polynomial to a Legendre series.
107: 
108:     Convert an array representing the coefficients of a polynomial (relative
109:     to the "standard" basis) ordered from lowest degree to highest, to an
110:     array of the coefficients of the equivalent Legendre series, ordered
111:     from lowest to highest degree.
112: 
113:     Parameters
114:     ----------
115:     pol : array_like
116:         1-D array containing the polynomial coefficients
117: 
118:     Returns
119:     -------
120:     c : ndarray
121:         1-D array containing the coefficients of the equivalent Legendre
122:         series.
123: 
124:     See Also
125:     --------
126:     leg2poly
127: 
128:     Notes
129:     -----
130:     The easy way to do conversions between polynomial basis sets
131:     is to use the convert method of a class instance.
132: 
133:     Examples
134:     --------
135:     >>> from numpy import polynomial as P
136:     >>> p = P.Polynomial(np.arange(4))
137:     >>> p
138:     Polynomial([ 0.,  1.,  2.,  3.], [-1.,  1.])
139:     >>> c = P.Legendre(P.poly2leg(p.coef))
140:     >>> c
141:     Legendre([ 1.  ,  3.25,  1.  ,  0.75], [-1.,  1.])
142: 
143:     '''
144:     [pol] = pu.as_series([pol])
145:     deg = len(pol) - 1
146:     res = 0
147:     for i in range(deg, -1, -1):
148:         res = legadd(legmulx(res), pol[i])
149:     return res
150: 
151: 
152: def leg2poly(c):
153:     '''
154:     Convert a Legendre series to a polynomial.
155: 
156:     Convert an array representing the coefficients of a Legendre series,
157:     ordered from lowest degree to highest, to an array of the coefficients
158:     of the equivalent polynomial (relative to the "standard" basis) ordered
159:     from lowest to highest degree.
160: 
161:     Parameters
162:     ----------
163:     c : array_like
164:         1-D array containing the Legendre series coefficients, ordered
165:         from lowest order term to highest.
166: 
167:     Returns
168:     -------
169:     pol : ndarray
170:         1-D array containing the coefficients of the equivalent polynomial
171:         (relative to the "standard" basis) ordered from lowest order term
172:         to highest.
173: 
174:     See Also
175:     --------
176:     poly2leg
177: 
178:     Notes
179:     -----
180:     The easy way to do conversions between polynomial basis sets
181:     is to use the convert method of a class instance.
182: 
183:     Examples
184:     --------
185:     >>> c = P.Legendre(range(4))
186:     >>> c
187:     Legendre([ 0.,  1.,  2.,  3.], [-1.,  1.])
188:     >>> p = c.convert(kind=P.Polynomial)
189:     >>> p
190:     Polynomial([-1. , -3.5,  3. ,  7.5], [-1.,  1.])
191:     >>> P.leg2poly(range(4))
192:     array([-1. , -3.5,  3. ,  7.5])
193: 
194: 
195:     '''
196:     from .polynomial import polyadd, polysub, polymulx
197: 
198:     [c] = pu.as_series([c])
199:     n = len(c)
200:     if n < 3:
201:         return c
202:     else:
203:         c0 = c[-2]
204:         c1 = c[-1]
205:         # i is the current degree of c1
206:         for i in range(n - 1, 1, -1):
207:             tmp = c0
208:             c0 = polysub(c[i - 2], (c1*(i - 1))/i)
209:             c1 = polyadd(tmp, (polymulx(c1)*(2*i - 1))/i)
210:         return polyadd(c0, polymulx(c1))
211: 
212: #
213: # These are constant arrays are of integer type so as to be compatible
214: # with the widest range of other types, such as Decimal.
215: #
216: 
217: # Legendre
218: legdomain = np.array([-1, 1])
219: 
220: # Legendre coefficients representing zero.
221: legzero = np.array([0])
222: 
223: # Legendre coefficients representing one.
224: legone = np.array([1])
225: 
226: # Legendre coefficients representing the identity x.
227: legx = np.array([0, 1])
228: 
229: 
230: def legline(off, scl):
231:     '''
232:     Legendre series whose graph is a straight line.
233: 
234: 
235: 
236:     Parameters
237:     ----------
238:     off, scl : scalars
239:         The specified line is given by ``off + scl*x``.
240: 
241:     Returns
242:     -------
243:     y : ndarray
244:         This module's representation of the Legendre series for
245:         ``off + scl*x``.
246: 
247:     See Also
248:     --------
249:     polyline, chebline
250: 
251:     Examples
252:     --------
253:     >>> import numpy.polynomial.legendre as L
254:     >>> L.legline(3,2)
255:     array([3, 2])
256:     >>> L.legval(-3, L.legline(3,2)) # should be -3
257:     -3.0
258: 
259:     '''
260:     if scl != 0:
261:         return np.array([off, scl])
262:     else:
263:         return np.array([off])
264: 
265: 
266: def legfromroots(roots):
267:     '''
268:     Generate a Legendre series with given roots.
269: 
270:     The function returns the coefficients of the polynomial
271: 
272:     .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
273: 
274:     in Legendre form, where the `r_n` are the roots specified in `roots`.
275:     If a zero has multiplicity n, then it must appear in `roots` n times.
276:     For instance, if 2 is a root of multiplicity three and 3 is a root of
277:     multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
278:     roots can appear in any order.
279: 
280:     If the returned coefficients are `c`, then
281: 
282:     .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)
283: 
284:     The coefficient of the last term is not generally 1 for monic
285:     polynomials in Legendre form.
286: 
287:     Parameters
288:     ----------
289:     roots : array_like
290:         Sequence containing the roots.
291: 
292:     Returns
293:     -------
294:     out : ndarray
295:         1-D array of coefficients.  If all roots are real then `out` is a
296:         real array, if some of the roots are complex, then `out` is complex
297:         even if all the coefficients in the result are real (see Examples
298:         below).
299: 
300:     See Also
301:     --------
302:     polyfromroots, chebfromroots, lagfromroots, hermfromroots,
303:     hermefromroots.
304: 
305:     Examples
306:     --------
307:     >>> import numpy.polynomial.legendre as L
308:     >>> L.legfromroots((-1,0,1)) # x^3 - x relative to the standard basis
309:     array([ 0. , -0.4,  0. ,  0.4])
310:     >>> j = complex(0,1)
311:     >>> L.legfromroots((-j,j)) # x^2 + 1 relative to the standard basis
312:     array([ 1.33333333+0.j,  0.00000000+0.j,  0.66666667+0.j])
313: 
314:     '''
315:     if len(roots) == 0:
316:         return np.ones(1)
317:     else:
318:         [roots] = pu.as_series([roots], trim=False)
319:         roots.sort()
320:         p = [legline(-r, 1) for r in roots]
321:         n = len(p)
322:         while n > 1:
323:             m, r = divmod(n, 2)
324:             tmp = [legmul(p[i], p[i+m]) for i in range(m)]
325:             if r:
326:                 tmp[0] = legmul(tmp[0], p[-1])
327:             p = tmp
328:             n = m
329:         return p[0]
330: 
331: 
332: def legadd(c1, c2):
333:     '''
334:     Add one Legendre series to another.
335: 
336:     Returns the sum of two Legendre series `c1` + `c2`.  The arguments
337:     are sequences of coefficients ordered from lowest order term to
338:     highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
339: 
340:     Parameters
341:     ----------
342:     c1, c2 : array_like
343:         1-D arrays of Legendre series coefficients ordered from low to
344:         high.
345: 
346:     Returns
347:     -------
348:     out : ndarray
349:         Array representing the Legendre series of their sum.
350: 
351:     See Also
352:     --------
353:     legsub, legmul, legdiv, legpow
354: 
355:     Notes
356:     -----
357:     Unlike multiplication, division, etc., the sum of two Legendre series
358:     is a Legendre series (without having to "reproject" the result onto
359:     the basis set) so addition, just like that of "standard" polynomials,
360:     is simply "component-wise."
361: 
362:     Examples
363:     --------
364:     >>> from numpy.polynomial import legendre as L
365:     >>> c1 = (1,2,3)
366:     >>> c2 = (3,2,1)
367:     >>> L.legadd(c1,c2)
368:     array([ 4.,  4.,  4.])
369: 
370:     '''
371:     # c1, c2 are trimmed copies
372:     [c1, c2] = pu.as_series([c1, c2])
373:     if len(c1) > len(c2):
374:         c1[:c2.size] += c2
375:         ret = c1
376:     else:
377:         c2[:c1.size] += c1
378:         ret = c2
379:     return pu.trimseq(ret)
380: 
381: 
382: def legsub(c1, c2):
383:     '''
384:     Subtract one Legendre series from another.
385: 
386:     Returns the difference of two Legendre series `c1` - `c2`.  The
387:     sequences of coefficients are from lowest order term to highest, i.e.,
388:     [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
389: 
390:     Parameters
391:     ----------
392:     c1, c2 : array_like
393:         1-D arrays of Legendre series coefficients ordered from low to
394:         high.
395: 
396:     Returns
397:     -------
398:     out : ndarray
399:         Of Legendre series coefficients representing their difference.
400: 
401:     See Also
402:     --------
403:     legadd, legmul, legdiv, legpow
404: 
405:     Notes
406:     -----
407:     Unlike multiplication, division, etc., the difference of two Legendre
408:     series is a Legendre series (without having to "reproject" the result
409:     onto the basis set) so subtraction, just like that of "standard"
410:     polynomials, is simply "component-wise."
411: 
412:     Examples
413:     --------
414:     >>> from numpy.polynomial import legendre as L
415:     >>> c1 = (1,2,3)
416:     >>> c2 = (3,2,1)
417:     >>> L.legsub(c1,c2)
418:     array([-2.,  0.,  2.])
419:     >>> L.legsub(c2,c1) # -C.legsub(c1,c2)
420:     array([ 2.,  0., -2.])
421: 
422:     '''
423:     # c1, c2 are trimmed copies
424:     [c1, c2] = pu.as_series([c1, c2])
425:     if len(c1) > len(c2):
426:         c1[:c2.size] -= c2
427:         ret = c1
428:     else:
429:         c2 = -c2
430:         c2[:c1.size] += c1
431:         ret = c2
432:     return pu.trimseq(ret)
433: 
434: 
435: def legmulx(c):
436:     '''Multiply a Legendre series by x.
437: 
438:     Multiply the Legendre series `c` by x, where x is the independent
439:     variable.
440: 
441: 
442:     Parameters
443:     ----------
444:     c : array_like
445:         1-D array of Legendre series coefficients ordered from low to
446:         high.
447: 
448:     Returns
449:     -------
450:     out : ndarray
451:         Array representing the result of the multiplication.
452: 
453:     Notes
454:     -----
455:     The multiplication uses the recursion relationship for Legendre
456:     polynomials in the form
457: 
458:     .. math::
459: 
460:       xP_i(x) = ((i + 1)*P_{i + 1}(x) + i*P_{i - 1}(x))/(2i + 1)
461: 
462:     '''
463:     # c is a trimmed copy
464:     [c] = pu.as_series([c])
465:     # The zero series needs special treatment
466:     if len(c) == 1 and c[0] == 0:
467:         return c
468: 
469:     prd = np.empty(len(c) + 1, dtype=c.dtype)
470:     prd[0] = c[0]*0
471:     prd[1] = c[0]
472:     for i in range(1, len(c)):
473:         j = i + 1
474:         k = i - 1
475:         s = i + j
476:         prd[j] = (c[i]*j)/s
477:         prd[k] += (c[i]*i)/s
478:     return prd
479: 
480: 
481: def legmul(c1, c2):
482:     '''
483:     Multiply one Legendre series by another.
484: 
485:     Returns the product of two Legendre series `c1` * `c2`.  The arguments
486:     are sequences of coefficients, from lowest order "term" to highest,
487:     e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
488: 
489:     Parameters
490:     ----------
491:     c1, c2 : array_like
492:         1-D arrays of Legendre series coefficients ordered from low to
493:         high.
494: 
495:     Returns
496:     -------
497:     out : ndarray
498:         Of Legendre series coefficients representing their product.
499: 
500:     See Also
501:     --------
502:     legadd, legsub, legdiv, legpow
503: 
504:     Notes
505:     -----
506:     In general, the (polynomial) product of two C-series results in terms
507:     that are not in the Legendre polynomial basis set.  Thus, to express
508:     the product as a Legendre series, it is necessary to "reproject" the
509:     product onto said basis set, which may produce "unintuitive" (but
510:     correct) results; see Examples section below.
511: 
512:     Examples
513:     --------
514:     >>> from numpy.polynomial import legendre as L
515:     >>> c1 = (1,2,3)
516:     >>> c2 = (3,2)
517:     >>> P.legmul(c1,c2) # multiplication requires "reprojection"
518:     array([  4.33333333,  10.4       ,  11.66666667,   3.6       ])
519: 
520:     '''
521:     # s1, s2 are trimmed copies
522:     [c1, c2] = pu.as_series([c1, c2])
523: 
524:     if len(c1) > len(c2):
525:         c = c2
526:         xs = c1
527:     else:
528:         c = c1
529:         xs = c2
530: 
531:     if len(c) == 1:
532:         c0 = c[0]*xs
533:         c1 = 0
534:     elif len(c) == 2:
535:         c0 = c[0]*xs
536:         c1 = c[1]*xs
537:     else:
538:         nd = len(c)
539:         c0 = c[-2]*xs
540:         c1 = c[-1]*xs
541:         for i in range(3, len(c) + 1):
542:             tmp = c0
543:             nd = nd - 1
544:             c0 = legsub(c[-i]*xs, (c1*(nd - 1))/nd)
545:             c1 = legadd(tmp, (legmulx(c1)*(2*nd - 1))/nd)
546:     return legadd(c0, legmulx(c1))
547: 
548: 
549: def legdiv(c1, c2):
550:     '''
551:     Divide one Legendre series by another.
552: 
553:     Returns the quotient-with-remainder of two Legendre series
554:     `c1` / `c2`.  The arguments are sequences of coefficients from lowest
555:     order "term" to highest, e.g., [1,2,3] represents the series
556:     ``P_0 + 2*P_1 + 3*P_2``.
557: 
558:     Parameters
559:     ----------
560:     c1, c2 : array_like
561:         1-D arrays of Legendre series coefficients ordered from low to
562:         high.
563: 
564:     Returns
565:     -------
566:     quo, rem : ndarrays
567:         Of Legendre series coefficients representing the quotient and
568:         remainder.
569: 
570:     See Also
571:     --------
572:     legadd, legsub, legmul, legpow
573: 
574:     Notes
575:     -----
576:     In general, the (polynomial) division of one Legendre series by another
577:     results in quotient and remainder terms that are not in the Legendre
578:     polynomial basis set.  Thus, to express these results as a Legendre
579:     series, it is necessary to "reproject" the results onto the Legendre
580:     basis set, which may produce "unintuitive" (but correct) results; see
581:     Examples section below.
582: 
583:     Examples
584:     --------
585:     >>> from numpy.polynomial import legendre as L
586:     >>> c1 = (1,2,3)
587:     >>> c2 = (3,2,1)
588:     >>> L.legdiv(c1,c2) # quotient "intuitive," remainder not
589:     (array([ 3.]), array([-8., -4.]))
590:     >>> c2 = (0,1,2,3)
591:     >>> L.legdiv(c2,c1) # neither "intuitive"
592:     (array([-0.07407407,  1.66666667]), array([-1.03703704, -2.51851852]))
593: 
594:     '''
595:     # c1, c2 are trimmed copies
596:     [c1, c2] = pu.as_series([c1, c2])
597:     if c2[-1] == 0:
598:         raise ZeroDivisionError()
599: 
600:     lc1 = len(c1)
601:     lc2 = len(c2)
602:     if lc1 < lc2:
603:         return c1[:1]*0, c1
604:     elif lc2 == 1:
605:         return c1/c2[-1], c1[:1]*0
606:     else:
607:         quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
608:         rem = c1
609:         for i in range(lc1 - lc2, - 1, -1):
610:             p = legmul([0]*i + [1], c2)
611:             q = rem[-1]/p[-1]
612:             rem = rem[:-1] - q*p[:-1]
613:             quo[i] = q
614:         return quo, pu.trimseq(rem)
615: 
616: 
617: def legpow(c, pow, maxpower=16):
618:     '''Raise a Legendre series to a power.
619: 
620:     Returns the Legendre series `c` raised to the power `pow`. The
621:     arguement `c` is a sequence of coefficients ordered from low to high.
622:     i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``
623: 
624:     Parameters
625:     ----------
626:     c : array_like
627:         1-D array of Legendre series coefficients ordered from low to
628:         high.
629:     pow : integer
630:         Power to which the series will be raised
631:     maxpower : integer, optional
632:         Maximum power allowed. This is mainly to limit growth of the series
633:         to unmanageable size. Default is 16
634: 
635:     Returns
636:     -------
637:     coef : ndarray
638:         Legendre series of power.
639: 
640:     See Also
641:     --------
642:     legadd, legsub, legmul, legdiv
643: 
644:     Examples
645:     --------
646: 
647:     '''
648:     # c is a trimmed copy
649:     [c] = pu.as_series([c])
650:     power = int(pow)
651:     if power != pow or power < 0:
652:         raise ValueError("Power must be a non-negative integer.")
653:     elif maxpower is not None and power > maxpower:
654:         raise ValueError("Power is too large")
655:     elif power == 0:
656:         return np.array([1], dtype=c.dtype)
657:     elif power == 1:
658:         return c
659:     else:
660:         # This can be made more efficient by using powers of two
661:         # in the usual way.
662:         prd = c
663:         for i in range(2, power + 1):
664:             prd = legmul(prd, c)
665:         return prd
666: 
667: 
668: def legder(c, m=1, scl=1, axis=0):
669:     '''
670:     Differentiate a Legendre series.
671: 
672:     Returns the Legendre series coefficients `c` differentiated `m` times
673:     along `axis`.  At each iteration the result is multiplied by `scl` (the
674:     scaling factor is for use in a linear change of variable). The argument
675:     `c` is an array of coefficients from low to high degree along each
676:     axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
677:     while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
678:     2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
679:     ``y``.
680: 
681:     Parameters
682:     ----------
683:     c : array_like
684:         Array of Legendre series coefficients. If c is multidimensional the
685:         different axis correspond to different variables with the degree in
686:         each axis given by the corresponding index.
687:     m : int, optional
688:         Number of derivatives taken, must be non-negative. (Default: 1)
689:     scl : scalar, optional
690:         Each differentiation is multiplied by `scl`.  The end result is
691:         multiplication by ``scl**m``.  This is for use in a linear change of
692:         variable. (Default: 1)
693:     axis : int, optional
694:         Axis over which the derivative is taken. (Default: 0).
695: 
696:         .. versionadded:: 1.7.0
697: 
698:     Returns
699:     -------
700:     der : ndarray
701:         Legendre series of the derivative.
702: 
703:     See Also
704:     --------
705:     legint
706: 
707:     Notes
708:     -----
709:     In general, the result of differentiating a Legendre series does not
710:     resemble the same operation on a power series. Thus the result of this
711:     function may be "unintuitive," albeit correct; see Examples section
712:     below.
713: 
714:     Examples
715:     --------
716:     >>> from numpy.polynomial import legendre as L
717:     >>> c = (1,2,3,4)
718:     >>> L.legder(c)
719:     array([  6.,   9.,  20.])
720:     >>> L.legder(c, 3)
721:     array([ 60.])
722:     >>> L.legder(c, scl=-1)
723:     array([ -6.,  -9., -20.])
724:     >>> L.legder(c, 2,-1)
725:     array([  9.,  60.])
726: 
727:     '''
728:     c = np.array(c, ndmin=1, copy=1)
729:     if c.dtype.char in '?bBhHiIlLqQpP':
730:         c = c.astype(np.double)
731:     cnt, iaxis = [int(t) for t in [m, axis]]
732: 
733:     if cnt != m:
734:         raise ValueError("The order of derivation must be integer")
735:     if cnt < 0:
736:         raise ValueError("The order of derivation must be non-negative")
737:     if iaxis != axis:
738:         raise ValueError("The axis must be integer")
739:     if not -c.ndim <= iaxis < c.ndim:
740:         raise ValueError("The axis is out of range")
741:     if iaxis < 0:
742:         iaxis += c.ndim
743: 
744:     if cnt == 0:
745:         return c
746: 
747:     c = np.rollaxis(c, iaxis)
748:     n = len(c)
749:     if cnt >= n:
750:         c = c[:1]*0
751:     else:
752:         for i in range(cnt):
753:             n = n - 1
754:             c *= scl
755:             der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
756:             for j in range(n, 2, -1):
757:                 der[j - 1] = (2*j - 1)*c[j]
758:                 c[j - 2] += c[j]
759:             if n > 1:
760:                 der[1] = 3*c[2]
761:             der[0] = c[1]
762:             c = der
763:     c = np.rollaxis(c, 0, iaxis + 1)
764:     return c
765: 
766: 
767: def legint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
768:     '''
769:     Integrate a Legendre series.
770: 
771:     Returns the Legendre series coefficients `c` integrated `m` times from
772:     `lbnd` along `axis`. At each iteration the resulting series is
773:     **multiplied** by `scl` and an integration constant, `k`, is added.
774:     The scaling factor is for use in a linear change of variable.  ("Buyer
775:     beware": note that, depending on what one is doing, one may want `scl`
776:     to be the reciprocal of what one might expect; for more information,
777:     see the Notes section below.)  The argument `c` is an array of
778:     coefficients from low to high degree along each axis, e.g., [1,2,3]
779:     represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
780:     represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
781:     2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
782: 
783:     Parameters
784:     ----------
785:     c : array_like
786:         Array of Legendre series coefficients. If c is multidimensional the
787:         different axis correspond to different variables with the degree in
788:         each axis given by the corresponding index.
789:     m : int, optional
790:         Order of integration, must be positive. (Default: 1)
791:     k : {[], list, scalar}, optional
792:         Integration constant(s).  The value of the first integral at
793:         ``lbnd`` is the first value in the list, the value of the second
794:         integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
795:         default), all constants are set to zero.  If ``m == 1``, a single
796:         scalar can be given instead of a list.
797:     lbnd : scalar, optional
798:         The lower bound of the integral. (Default: 0)
799:     scl : scalar, optional
800:         Following each integration the result is *multiplied* by `scl`
801:         before the integration constant is added. (Default: 1)
802:     axis : int, optional
803:         Axis over which the integral is taken. (Default: 0).
804: 
805:         .. versionadded:: 1.7.0
806: 
807:     Returns
808:     -------
809:     S : ndarray
810:         Legendre series coefficient array of the integral.
811: 
812:     Raises
813:     ------
814:     ValueError
815:         If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
816:         ``np.isscalar(scl) == False``.
817: 
818:     See Also
819:     --------
820:     legder
821: 
822:     Notes
823:     -----
824:     Note that the result of each integration is *multiplied* by `scl`.
825:     Why is this important to note?  Say one is making a linear change of
826:     variable :math:`u = ax + b` in an integral relative to `x`.  Then
827:     .. math::`dx = du/a`, so one will need to set `scl` equal to
828:     :math:`1/a` - perhaps not what one would have first thought.
829: 
830:     Also note that, in general, the result of integrating a C-series needs
831:     to be "reprojected" onto the C-series basis set.  Thus, typically,
832:     the result of this function is "unintuitive," albeit correct; see
833:     Examples section below.
834: 
835:     Examples
836:     --------
837:     >>> from numpy.polynomial import legendre as L
838:     >>> c = (1,2,3)
839:     >>> L.legint(c)
840:     array([ 0.33333333,  0.4       ,  0.66666667,  0.6       ])
841:     >>> L.legint(c, 3)
842:     array([  1.66666667e-02,  -1.78571429e-02,   4.76190476e-02,
843:             -1.73472348e-18,   1.90476190e-02,   9.52380952e-03])
844:     >>> L.legint(c, k=3)
845:     array([ 3.33333333,  0.4       ,  0.66666667,  0.6       ])
846:     >>> L.legint(c, lbnd=-2)
847:     array([ 7.33333333,  0.4       ,  0.66666667,  0.6       ])
848:     >>> L.legint(c, scl=2)
849:     array([ 0.66666667,  0.8       ,  1.33333333,  1.2       ])
850: 
851:     '''
852:     c = np.array(c, ndmin=1, copy=1)
853:     if c.dtype.char in '?bBhHiIlLqQpP':
854:         c = c.astype(np.double)
855:     if not np.iterable(k):
856:         k = [k]
857:     cnt, iaxis = [int(t) for t in [m, axis]]
858: 
859:     if cnt != m:
860:         raise ValueError("The order of integration must be integer")
861:     if cnt < 0:
862:         raise ValueError("The order of integration must be non-negative")
863:     if len(k) > cnt:
864:         raise ValueError("Too many integration constants")
865:     if iaxis != axis:
866:         raise ValueError("The axis must be integer")
867:     if not -c.ndim <= iaxis < c.ndim:
868:         raise ValueError("The axis is out of range")
869:     if iaxis < 0:
870:         iaxis += c.ndim
871: 
872:     if cnt == 0:
873:         return c
874: 
875:     c = np.rollaxis(c, iaxis)
876:     k = list(k) + [0]*(cnt - len(k))
877:     for i in range(cnt):
878:         n = len(c)
879:         c *= scl
880:         if n == 1 and np.all(c[0] == 0):
881:             c[0] += k[i]
882:         else:
883:             tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
884:             tmp[0] = c[0]*0
885:             tmp[1] = c[0]
886:             if n > 1:
887:                 tmp[2] = c[1]/3
888:             for j in range(2, n):
889:                 t = c[j]/(2*j + 1)
890:                 tmp[j + 1] = t
891:                 tmp[j - 1] -= t
892:             tmp[0] += k[i] - legval(lbnd, tmp)
893:             c = tmp
894:     c = np.rollaxis(c, 0, iaxis + 1)
895:     return c
896: 
897: 
898: def legval(x, c, tensor=True):
899:     '''
900:     Evaluate a Legendre series at points x.
901: 
902:     If `c` is of length `n + 1`, this function returns the value:
903: 
904:     .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)
905: 
906:     The parameter `x` is converted to an array only if it is a tuple or a
907:     list, otherwise it is treated as a scalar. In either case, either `x`
908:     or its elements must support multiplication and addition both with
909:     themselves and with the elements of `c`.
910: 
911:     If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
912:     `c` is multidimensional, then the shape of the result depends on the
913:     value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
914:     x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
915:     scalars have shape (,).
916: 
917:     Trailing zeros in the coefficients will be used in the evaluation, so
918:     they should be avoided if efficiency is a concern.
919: 
920:     Parameters
921:     ----------
922:     x : array_like, compatible object
923:         If `x` is a list or tuple, it is converted to an ndarray, otherwise
924:         it is left unchanged and treated as a scalar. In either case, `x`
925:         or its elements must support addition and multiplication with
926:         with themselves and with the elements of `c`.
927:     c : array_like
928:         Array of coefficients ordered so that the coefficients for terms of
929:         degree n are contained in c[n]. If `c` is multidimensional the
930:         remaining indices enumerate multiple polynomials. In the two
931:         dimensional case the coefficients may be thought of as stored in
932:         the columns of `c`.
933:     tensor : boolean, optional
934:         If True, the shape of the coefficient array is extended with ones
935:         on the right, one for each dimension of `x`. Scalars have dimension 0
936:         for this action. The result is that every column of coefficients in
937:         `c` is evaluated for every element of `x`. If False, `x` is broadcast
938:         over the columns of `c` for the evaluation.  This keyword is useful
939:         when `c` is multidimensional. The default value is True.
940: 
941:         .. versionadded:: 1.7.0
942: 
943:     Returns
944:     -------
945:     values : ndarray, algebra_like
946:         The shape of the return value is described above.
947: 
948:     See Also
949:     --------
950:     legval2d, leggrid2d, legval3d, leggrid3d
951: 
952:     Notes
953:     -----
954:     The evaluation uses Clenshaw recursion, aka synthetic division.
955: 
956:     Examples
957:     --------
958: 
959:     '''
960:     c = np.array(c, ndmin=1, copy=0)
961:     if c.dtype.char in '?bBhHiIlLqQpP':
962:         c = c.astype(np.double)
963:     if isinstance(x, (tuple, list)):
964:         x = np.asarray(x)
965:     if isinstance(x, np.ndarray) and tensor:
966:         c = c.reshape(c.shape + (1,)*x.ndim)
967: 
968:     if len(c) == 1:
969:         c0 = c[0]
970:         c1 = 0
971:     elif len(c) == 2:
972:         c0 = c[0]
973:         c1 = c[1]
974:     else:
975:         nd = len(c)
976:         c0 = c[-2]
977:         c1 = c[-1]
978:         for i in range(3, len(c) + 1):
979:             tmp = c0
980:             nd = nd - 1
981:             c0 = c[-i] - (c1*(nd - 1))/nd
982:             c1 = tmp + (c1*x*(2*nd - 1))/nd
983:     return c0 + c1*x
984: 
985: 
986: def legval2d(x, y, c):
987:     '''
988:     Evaluate a 2-D Legendre series at points (x, y).
989: 
990:     This function returns the values:
991: 
992:     .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)
993: 
994:     The parameters `x` and `y` are converted to arrays only if they are
995:     tuples or a lists, otherwise they are treated as a scalars and they
996:     must have the same shape after conversion. In either case, either `x`
997:     and `y` or their elements must support multiplication and addition both
998:     with themselves and with the elements of `c`.
999: 
1000:     If `c` is a 1-D array a one is implicitly appended to its shape to make
1001:     it 2-D. The shape of the result will be c.shape[2:] + x.shape.
1002: 
1003:     Parameters
1004:     ----------
1005:     x, y : array_like, compatible objects
1006:         The two dimensional series is evaluated at the points `(x, y)`,
1007:         where `x` and `y` must have the same shape. If `x` or `y` is a list
1008:         or tuple, it is first converted to an ndarray, otherwise it is left
1009:         unchanged and if it isn't an ndarray it is treated as a scalar.
1010:     c : array_like
1011:         Array of coefficients ordered so that the coefficient of the term
1012:         of multi-degree i,j is contained in ``c[i,j]``. If `c` has
1013:         dimension greater than two the remaining indices enumerate multiple
1014:         sets of coefficients.
1015: 
1016:     Returns
1017:     -------
1018:     values : ndarray, compatible object
1019:         The values of the two dimensional Legendre series at points formed
1020:         from pairs of corresponding values from `x` and `y`.
1021: 
1022:     See Also
1023:     --------
1024:     legval, leggrid2d, legval3d, leggrid3d
1025: 
1026:     Notes
1027:     -----
1028: 
1029:     .. versionadded::1.7.0
1030: 
1031:     '''
1032:     try:
1033:         x, y = np.array((x, y), copy=0)
1034:     except:
1035:         raise ValueError('x, y are incompatible')
1036: 
1037:     c = legval(x, c)
1038:     c = legval(y, c, tensor=False)
1039:     return c
1040: 
1041: 
1042: def leggrid2d(x, y, c):
1043:     '''
1044:     Evaluate a 2-D Legendre series on the Cartesian product of x and y.
1045: 
1046:     This function returns the values:
1047: 
1048:     .. math:: p(a,b) = \sum_{i,j} c_{i,j} * L_i(a) * L_j(b)
1049: 
1050:     where the points `(a, b)` consist of all pairs formed by taking
1051:     `a` from `x` and `b` from `y`. The resulting points form a grid with
1052:     `x` in the first dimension and `y` in the second.
1053: 
1054:     The parameters `x` and `y` are converted to arrays only if they are
1055:     tuples or a lists, otherwise they are treated as a scalars. In either
1056:     case, either `x` and `y` or their elements must support multiplication
1057:     and addition both with themselves and with the elements of `c`.
1058: 
1059:     If `c` has fewer than two dimensions, ones are implicitly appended to
1060:     its shape to make it 2-D. The shape of the result will be c.shape[2:] +
1061:     x.shape + y.shape.
1062: 
1063:     Parameters
1064:     ----------
1065:     x, y : array_like, compatible objects
1066:         The two dimensional series is evaluated at the points in the
1067:         Cartesian product of `x` and `y`.  If `x` or `y` is a list or
1068:         tuple, it is first converted to an ndarray, otherwise it is left
1069:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
1070:     c : array_like
1071:         Array of coefficients ordered so that the coefficient of the term of
1072:         multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
1073:         greater than two the remaining indices enumerate multiple sets of
1074:         coefficients.
1075: 
1076:     Returns
1077:     -------
1078:     values : ndarray, compatible object
1079:         The values of the two dimensional Chebyshev series at points in the
1080:         Cartesian product of `x` and `y`.
1081: 
1082:     See Also
1083:     --------
1084:     legval, legval2d, legval3d, leggrid3d
1085: 
1086:     Notes
1087:     -----
1088: 
1089:     .. versionadded::1.7.0
1090: 
1091:     '''
1092:     c = legval(x, c)
1093:     c = legval(y, c)
1094:     return c
1095: 
1096: 
1097: def legval3d(x, y, z, c):
1098:     '''
1099:     Evaluate a 3-D Legendre series at points (x, y, z).
1100: 
1101:     This function returns the values:
1102: 
1103:     .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)
1104: 
1105:     The parameters `x`, `y`, and `z` are converted to arrays only if
1106:     they are tuples or a lists, otherwise they are treated as a scalars and
1107:     they must have the same shape after conversion. In either case, either
1108:     `x`, `y`, and `z` or their elements must support multiplication and
1109:     addition both with themselves and with the elements of `c`.
1110: 
1111:     If `c` has fewer than 3 dimensions, ones are implicitly appended to its
1112:     shape to make it 3-D. The shape of the result will be c.shape[3:] +
1113:     x.shape.
1114: 
1115:     Parameters
1116:     ----------
1117:     x, y, z : array_like, compatible object
1118:         The three dimensional series is evaluated at the points
1119:         `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
1120:         any of `x`, `y`, or `z` is a list or tuple, it is first converted
1121:         to an ndarray, otherwise it is left unchanged and if it isn't an
1122:         ndarray it is  treated as a scalar.
1123:     c : array_like
1124:         Array of coefficients ordered so that the coefficient of the term of
1125:         multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
1126:         greater than 3 the remaining indices enumerate multiple sets of
1127:         coefficients.
1128: 
1129:     Returns
1130:     -------
1131:     values : ndarray, compatible object
1132:         The values of the multidimensional polynomial on points formed with
1133:         triples of corresponding values from `x`, `y`, and `z`.
1134: 
1135:     See Also
1136:     --------
1137:     legval, legval2d, leggrid2d, leggrid3d
1138: 
1139:     Notes
1140:     -----
1141: 
1142:     .. versionadded::1.7.0
1143: 
1144:     '''
1145:     try:
1146:         x, y, z = np.array((x, y, z), copy=0)
1147:     except:
1148:         raise ValueError('x, y, z are incompatible')
1149: 
1150:     c = legval(x, c)
1151:     c = legval(y, c, tensor=False)
1152:     c = legval(z, c, tensor=False)
1153:     return c
1154: 
1155: 
1156: def leggrid3d(x, y, z, c):
1157:     '''
1158:     Evaluate a 3-D Legendre series on the Cartesian product of x, y, and z.
1159: 
1160:     This function returns the values:
1161: 
1162:     .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)
1163: 
1164:     where the points `(a, b, c)` consist of all triples formed by taking
1165:     `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
1166:     a grid with `x` in the first dimension, `y` in the second, and `z` in
1167:     the third.
1168: 
1169:     The parameters `x`, `y`, and `z` are converted to arrays only if they
1170:     are tuples or a lists, otherwise they are treated as a scalars. In
1171:     either case, either `x`, `y`, and `z` or their elements must support
1172:     multiplication and addition both with themselves and with the elements
1173:     of `c`.
1174: 
1175:     If `c` has fewer than three dimensions, ones are implicitly appended to
1176:     its shape to make it 3-D. The shape of the result will be c.shape[3:] +
1177:     x.shape + y.shape + z.shape.
1178: 
1179:     Parameters
1180:     ----------
1181:     x, y, z : array_like, compatible objects
1182:         The three dimensional series is evaluated at the points in the
1183:         Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
1184:         list or tuple, it is first converted to an ndarray, otherwise it is
1185:         left unchanged and, if it isn't an ndarray, it is treated as a
1186:         scalar.
1187:     c : array_like
1188:         Array of coefficients ordered so that the coefficients for terms of
1189:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
1190:         greater than two the remaining indices enumerate multiple sets of
1191:         coefficients.
1192: 
1193:     Returns
1194:     -------
1195:     values : ndarray, compatible object
1196:         The values of the two dimensional polynomial at points in the Cartesian
1197:         product of `x` and `y`.
1198: 
1199:     See Also
1200:     --------
1201:     legval, legval2d, leggrid2d, legval3d
1202: 
1203:     Notes
1204:     -----
1205: 
1206:     .. versionadded::1.7.0
1207: 
1208:     '''
1209:     c = legval(x, c)
1210:     c = legval(y, c)
1211:     c = legval(z, c)
1212:     return c
1213: 
1214: 
1215: def legvander(x, deg):
1216:     '''Pseudo-Vandermonde matrix of given degree.
1217: 
1218:     Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
1219:     `x`. The pseudo-Vandermonde matrix is defined by
1220: 
1221:     .. math:: V[..., i] = L_i(x)
1222: 
1223:     where `0 <= i <= deg`. The leading indices of `V` index the elements of
1224:     `x` and the last index is the degree of the Legendre polynomial.
1225: 
1226:     If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
1227:     array ``V = legvander(x, n)``, then ``np.dot(V, c)`` and
1228:     ``legval(x, c)`` are the same up to roundoff. This equivalence is
1229:     useful both for least squares fitting and for the evaluation of a large
1230:     number of Legendre series of the same degree and sample points.
1231: 
1232:     Parameters
1233:     ----------
1234:     x : array_like
1235:         Array of points. The dtype is converted to float64 or complex128
1236:         depending on whether any of the elements are complex. If `x` is
1237:         scalar it is converted to a 1-D array.
1238:     deg : int
1239:         Degree of the resulting matrix.
1240: 
1241:     Returns
1242:     -------
1243:     vander : ndarray
1244:         The pseudo-Vandermonde matrix. The shape of the returned matrix is
1245:         ``x.shape + (deg + 1,)``, where The last index is the degree of the
1246:         corresponding Legendre polynomial.  The dtype will be the same as
1247:         the converted `x`.
1248: 
1249:     '''
1250:     ideg = int(deg)
1251:     if ideg != deg:
1252:         raise ValueError("deg must be integer")
1253:     if ideg < 0:
1254:         raise ValueError("deg must be non-negative")
1255: 
1256:     x = np.array(x, copy=0, ndmin=1) + 0.0
1257:     dims = (ideg + 1,) + x.shape
1258:     dtyp = x.dtype
1259:     v = np.empty(dims, dtype=dtyp)
1260:     # Use forward recursion to generate the entries. This is not as accurate
1261:     # as reverse recursion in this application but it is more efficient.
1262:     v[0] = x*0 + 1
1263:     if ideg > 0:
1264:         v[1] = x
1265:         for i in range(2, ideg + 1):
1266:             v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
1267:     return np.rollaxis(v, 0, v.ndim)
1268: 
1269: 
1270: def legvander2d(x, y, deg):
1271:     '''Pseudo-Vandermonde matrix of given degrees.
1272: 
1273:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1274:     points `(x, y)`. The pseudo-Vandermonde matrix is defined by
1275: 
1276:     .. math:: V[..., deg[1]*i + j] = L_i(x) * L_j(y),
1277: 
1278:     where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
1279:     `V` index the points `(x, y)` and the last index encodes the degrees of
1280:     the Legendre polynomials.
1281: 
1282:     If ``V = legvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
1283:     correspond to the elements of a 2-D coefficient array `c` of shape
1284:     (xdeg + 1, ydeg + 1) in the order
1285: 
1286:     .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
1287: 
1288:     and ``np.dot(V, c.flat)`` and ``legval2d(x, y, c)`` will be the same
1289:     up to roundoff. This equivalence is useful both for least squares
1290:     fitting and for the evaluation of a large number of 2-D Legendre
1291:     series of the same degrees and sample points.
1292: 
1293:     Parameters
1294:     ----------
1295:     x, y : array_like
1296:         Arrays of point coordinates, all of the same shape. The dtypes
1297:         will be converted to either float64 or complex128 depending on
1298:         whether any of the elements are complex. Scalars are converted to
1299:         1-D arrays.
1300:     deg : list of ints
1301:         List of maximum degrees of the form [x_deg, y_deg].
1302: 
1303:     Returns
1304:     -------
1305:     vander2d : ndarray
1306:         The shape of the returned matrix is ``x.shape + (order,)``, where
1307:         :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
1308:         as the converted `x` and `y`.
1309: 
1310:     See Also
1311:     --------
1312:     legvander, legvander3d. legval2d, legval3d
1313: 
1314:     Notes
1315:     -----
1316: 
1317:     .. versionadded::1.7.0
1318: 
1319:     '''
1320:     ideg = [int(d) for d in deg]
1321:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1322:     if is_valid != [1, 1]:
1323:         raise ValueError("degrees must be non-negative integers")
1324:     degx, degy = ideg
1325:     x, y = np.array((x, y), copy=0) + 0.0
1326: 
1327:     vx = legvander(x, degx)
1328:     vy = legvander(y, degy)
1329:     v = vx[..., None]*vy[..., None,:]
1330:     return v.reshape(v.shape[:-2] + (-1,))
1331: 
1332: 
1333: def legvander3d(x, y, z, deg):
1334:     '''Pseudo-Vandermonde matrix of given degrees.
1335: 
1336:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1337:     points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
1338:     then The pseudo-Vandermonde matrix is defined by
1339: 
1340:     .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),
1341: 
1342:     where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
1343:     indices of `V` index the points `(x, y, z)` and the last index encodes
1344:     the degrees of the Legendre polynomials.
1345: 
1346:     If ``V = legvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
1347:     of `V` correspond to the elements of a 3-D coefficient array `c` of
1348:     shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
1349: 
1350:     .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
1351: 
1352:     and ``np.dot(V, c.flat)`` and ``legval3d(x, y, z, c)`` will be the
1353:     same up to roundoff. This equivalence is useful both for least squares
1354:     fitting and for the evaluation of a large number of 3-D Legendre
1355:     series of the same degrees and sample points.
1356: 
1357:     Parameters
1358:     ----------
1359:     x, y, z : array_like
1360:         Arrays of point coordinates, all of the same shape. The dtypes will
1361:         be converted to either float64 or complex128 depending on whether
1362:         any of the elements are complex. Scalars are converted to 1-D
1363:         arrays.
1364:     deg : list of ints
1365:         List of maximum degrees of the form [x_deg, y_deg, z_deg].
1366: 
1367:     Returns
1368:     -------
1369:     vander3d : ndarray
1370:         The shape of the returned matrix is ``x.shape + (order,)``, where
1371:         :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
1372:         be the same as the converted `x`, `y`, and `z`.
1373: 
1374:     See Also
1375:     --------
1376:     legvander, legvander3d. legval2d, legval3d
1377: 
1378:     Notes
1379:     -----
1380: 
1381:     .. versionadded::1.7.0
1382: 
1383:     '''
1384:     ideg = [int(d) for d in deg]
1385:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1386:     if is_valid != [1, 1, 1]:
1387:         raise ValueError("degrees must be non-negative integers")
1388:     degx, degy, degz = ideg
1389:     x, y, z = np.array((x, y, z), copy=0) + 0.0
1390: 
1391:     vx = legvander(x, degx)
1392:     vy = legvander(y, degy)
1393:     vz = legvander(z, degz)
1394:     v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
1395:     return v.reshape(v.shape[:-3] + (-1,))
1396: 
1397: 
1398: def legfit(x, y, deg, rcond=None, full=False, w=None):
1399:     '''
1400:     Least squares fit of Legendre series to data.
1401: 
1402:     Return the coefficients of a Legendre series of degree `deg` that is the
1403:     least squares fit to the data values `y` given at points `x`. If `y` is
1404:     1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
1405:     fits are done, one for each column of `y`, and the resulting
1406:     coefficients are stored in the corresponding columns of a 2-D return.
1407:     The fitted polynomial(s) are in the form
1408: 
1409:     .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),
1410: 
1411:     where `n` is `deg`.
1412: 
1413:     Parameters
1414:     ----------
1415:     x : array_like, shape (M,)
1416:         x-coordinates of the M sample points ``(x[i], y[i])``.
1417:     y : array_like, shape (M,) or (M, K)
1418:         y-coordinates of the sample points. Several data sets of sample
1419:         points sharing the same x-coordinates can be fitted at once by
1420:         passing in a 2D-array that contains one dataset per column.
1421:     deg : int or 1-D array_like
1422:         Degree(s) of the fitting polynomials. If `deg` is a single integer
1423:         all terms up to and including the `deg`'th term are included in the
1424:         fit. For Numpy versions >= 1.11 a list of integers specifying the
1425:         degrees of the terms to include may be used instead.
1426:     rcond : float, optional
1427:         Relative condition number of the fit. Singular values smaller than
1428:         this relative to the largest singular value will be ignored. The
1429:         default value is len(x)*eps, where eps is the relative precision of
1430:         the float type, about 2e-16 in most cases.
1431:     full : bool, optional
1432:         Switch determining nature of return value. When it is False (the
1433:         default) just the coefficients are returned, when True diagnostic
1434:         information from the singular value decomposition is also returned.
1435:     w : array_like, shape (`M`,), optional
1436:         Weights. If not None, the contribution of each point
1437:         ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
1438:         weights are chosen so that the errors of the products ``w[i]*y[i]``
1439:         all have the same variance.  The default value is None.
1440: 
1441:         .. versionadded:: 1.5.0
1442: 
1443:     Returns
1444:     -------
1445:     coef : ndarray, shape (M,) or (M, K)
1446:         Legendre coefficients ordered from low to high. If `y` was
1447:         2-D, the coefficients for the data in column k of `y` are in
1448:         column `k`. If `deg` is specified as a list, coefficients for
1449:         terms not included in the fit are set equal to zero in the
1450:         returned `coef`.
1451: 
1452:     [residuals, rank, singular_values, rcond] : list
1453:         These values are only returned if `full` = True
1454: 
1455:         resid -- sum of squared residuals of the least squares fit
1456:         rank -- the numerical rank of the scaled Vandermonde matrix
1457:         sv -- singular values of the scaled Vandermonde matrix
1458:         rcond -- value of `rcond`.
1459: 
1460:         For more details, see `linalg.lstsq`.
1461: 
1462:     Warns
1463:     -----
1464:     RankWarning
1465:         The rank of the coefficient matrix in the least-squares fit is
1466:         deficient. The warning is only raised if `full` = False.  The
1467:         warnings can be turned off by
1468: 
1469:         >>> import warnings
1470:         >>> warnings.simplefilter('ignore', RankWarning)
1471: 
1472:     See Also
1473:     --------
1474:     chebfit, polyfit, lagfit, hermfit, hermefit
1475:     legval : Evaluates a Legendre series.
1476:     legvander : Vandermonde matrix of Legendre series.
1477:     legweight : Legendre weight function (= 1).
1478:     linalg.lstsq : Computes a least-squares fit from the matrix.
1479:     scipy.interpolate.UnivariateSpline : Computes spline fits.
1480: 
1481:     Notes
1482:     -----
1483:     The solution is the coefficients of the Legendre series `p` that
1484:     minimizes the sum of the weighted squared errors
1485: 
1486:     .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
1487: 
1488:     where :math:`w_j` are the weights. This problem is solved by setting up
1489:     as the (typically) overdetermined matrix equation
1490: 
1491:     .. math:: V(x) * c = w * y,
1492: 
1493:     where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
1494:     coefficients to be solved for, `w` are the weights, and `y` are the
1495:     observed values.  This equation is then solved using the singular value
1496:     decomposition of `V`.
1497: 
1498:     If some of the singular values of `V` are so small that they are
1499:     neglected, then a `RankWarning` will be issued. This means that the
1500:     coefficient values may be poorly determined. Using a lower order fit
1501:     will usually get rid of the warning.  The `rcond` parameter can also be
1502:     set to a value smaller than its default, but the resulting fit may be
1503:     spurious and have large contributions from roundoff error.
1504: 
1505:     Fits using Legendre series are usually better conditioned than fits
1506:     using power series, but much can depend on the distribution of the
1507:     sample points and the smoothness of the data. If the quality of the fit
1508:     is inadequate splines may be a good alternative.
1509: 
1510:     References
1511:     ----------
1512:     .. [1] Wikipedia, "Curve fitting",
1513:            http://en.wikipedia.org/wiki/Curve_fitting
1514: 
1515:     Examples
1516:     --------
1517: 
1518:     '''
1519:     x = np.asarray(x) + 0.0
1520:     y = np.asarray(y) + 0.0
1521:     deg = np.asarray(deg)
1522: 
1523:     # check arguments.
1524:     if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
1525:         raise TypeError("deg must be an int or non-empty 1-D array of int")
1526:     if deg.min() < 0:
1527:         raise ValueError("expected deg >= 0")
1528:     if x.ndim != 1:
1529:         raise TypeError("expected 1D vector for x")
1530:     if x.size == 0:
1531:         raise TypeError("expected non-empty vector for x")
1532:     if y.ndim < 1 or y.ndim > 2:
1533:         raise TypeError("expected 1D or 2D array for y")
1534:     if len(x) != len(y):
1535:         raise TypeError("expected x and y to have same length")
1536: 
1537:     if deg.ndim == 0:
1538:         lmax = deg
1539:         order = lmax + 1
1540:         van = legvander(x, lmax)
1541:     else:
1542:         deg = np.sort(deg)
1543:         lmax = deg[-1]
1544:         order = len(deg)
1545:         van = legvander(x, lmax)[:, deg]
1546: 
1547:     # set up the least squares matrices in transposed form
1548:     lhs = van.T
1549:     rhs = y.T
1550:     if w is not None:
1551:         w = np.asarray(w) + 0.0
1552:         if w.ndim != 1:
1553:             raise TypeError("expected 1D vector for w")
1554:         if len(x) != len(w):
1555:             raise TypeError("expected x and w to have same length")
1556:         # apply weights. Don't use inplace operations as they
1557:         # can cause problems with NA.
1558:         lhs = lhs * w
1559:         rhs = rhs * w
1560: 
1561:     # set rcond
1562:     if rcond is None:
1563:         rcond = len(x)*np.finfo(x.dtype).eps
1564: 
1565:     # Determine the norms of the design matrix columns.
1566:     if issubclass(lhs.dtype.type, np.complexfloating):
1567:         scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
1568:     else:
1569:         scl = np.sqrt(np.square(lhs).sum(1))
1570:     scl[scl == 0] = 1
1571: 
1572:     # Solve the least squares problem.
1573:     c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
1574:     c = (c.T/scl).T
1575: 
1576:     # Expand c to include non-fitted coefficients which are set to zero
1577:     if deg.ndim > 0:
1578:         if c.ndim == 2:
1579:             cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
1580:         else:
1581:             cc = np.zeros(lmax+1, dtype=c.dtype)
1582:         cc[deg] = c
1583:         c = cc
1584: 
1585:     # warn on rank reduction
1586:     if rank != order and not full:
1587:         msg = "The fit may be poorly conditioned"
1588:         warnings.warn(msg, pu.RankWarning)
1589: 
1590:     if full:
1591:         return c, [resids, rank, s, rcond]
1592:     else:
1593:         return c
1594: 
1595: 
1596: def legcompanion(c):
1597:     '''Return the scaled companion matrix of c.
1598: 
1599:     The basis polynomials are scaled so that the companion matrix is
1600:     symmetric when `c` is an Legendre basis polynomial. This provides
1601:     better eigenvalue estimates than the unscaled case and for basis
1602:     polynomials the eigenvalues are guaranteed to be real if
1603:     `numpy.linalg.eigvalsh` is used to obtain them.
1604: 
1605:     Parameters
1606:     ----------
1607:     c : array_like
1608:         1-D array of Legendre series coefficients ordered from low to high
1609:         degree.
1610: 
1611:     Returns
1612:     -------
1613:     mat : ndarray
1614:         Scaled companion matrix of dimensions (deg, deg).
1615: 
1616:     Notes
1617:     -----
1618: 
1619:     .. versionadded::1.7.0
1620: 
1621:     '''
1622:     # c is a trimmed copy
1623:     [c] = pu.as_series([c])
1624:     if len(c) < 2:
1625:         raise ValueError('Series must have maximum degree of at least 1.')
1626:     if len(c) == 2:
1627:         return np.array([[-c[0]/c[1]]])
1628: 
1629:     n = len(c) - 1
1630:     mat = np.zeros((n, n), dtype=c.dtype)
1631:     scl = 1./np.sqrt(2*np.arange(n) + 1)
1632:     top = mat.reshape(-1)[1::n+1]
1633:     bot = mat.reshape(-1)[n::n+1]
1634:     top[...] = np.arange(1, n)*scl[:n-1]*scl[1:n]
1635:     bot[...] = top
1636:     mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*(n/(2*n - 1))
1637:     return mat
1638: 
1639: 
1640: def legroots(c):
1641:     '''
1642:     Compute the roots of a Legendre series.
1643: 
1644:     Return the roots (a.k.a. "zeros") of the polynomial
1645: 
1646:     .. math:: p(x) = \\sum_i c[i] * L_i(x).
1647: 
1648:     Parameters
1649:     ----------
1650:     c : 1-D array_like
1651:         1-D array of coefficients.
1652: 
1653:     Returns
1654:     -------
1655:     out : ndarray
1656:         Array of the roots of the series. If all the roots are real,
1657:         then `out` is also real, otherwise it is complex.
1658: 
1659:     See Also
1660:     --------
1661:     polyroots, chebroots, lagroots, hermroots, hermeroots
1662: 
1663:     Notes
1664:     -----
1665:     The root estimates are obtained as the eigenvalues of the companion
1666:     matrix, Roots far from the origin of the complex plane may have large
1667:     errors due to the numerical instability of the series for such values.
1668:     Roots with multiplicity greater than 1 will also show larger errors as
1669:     the value of the series near such points is relatively insensitive to
1670:     errors in the roots. Isolated roots near the origin can be improved by
1671:     a few iterations of Newton's method.
1672: 
1673:     The Legendre series basis polynomials aren't powers of ``x`` so the
1674:     results of this function may seem unintuitive.
1675: 
1676:     Examples
1677:     --------
1678:     >>> import numpy.polynomial.legendre as leg
1679:     >>> leg.legroots((1, 2, 3, 4)) # 4L_3 + 3L_2 + 2L_1 + 1L_0, all real roots
1680:     array([-0.85099543, -0.11407192,  0.51506735])
1681: 
1682:     '''
1683:     # c is a trimmed copy
1684:     [c] = pu.as_series([c])
1685:     if len(c) < 2:
1686:         return np.array([], dtype=c.dtype)
1687:     if len(c) == 2:
1688:         return np.array([-c[0]/c[1]])
1689: 
1690:     m = legcompanion(c)
1691:     r = la.eigvals(m)
1692:     r.sort()
1693:     return r
1694: 
1695: 
1696: def leggauss(deg):
1697:     '''
1698:     Gauss-Legendre quadrature.
1699: 
1700:     Computes the sample points and weights for Gauss-Legendre quadrature.
1701:     These sample points and weights will correctly integrate polynomials of
1702:     degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
1703:     the weight function :math:`f(x) = 1`.
1704: 
1705:     Parameters
1706:     ----------
1707:     deg : int
1708:         Number of sample points and weights. It must be >= 1.
1709: 
1710:     Returns
1711:     -------
1712:     x : ndarray
1713:         1-D ndarray containing the sample points.
1714:     y : ndarray
1715:         1-D ndarray containing the weights.
1716: 
1717:     Notes
1718:     -----
1719: 
1720:     .. versionadded::1.7.0
1721: 
1722:     The results have only been tested up to degree 100, higher degrees may
1723:     be problematic. The weights are determined by using the fact that
1724: 
1725:     .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))
1726: 
1727:     where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
1728:     is the k'th root of :math:`L_n`, and then scaling the results to get
1729:     the right value when integrating 1.
1730: 
1731:     '''
1732:     ideg = int(deg)
1733:     if ideg != deg or ideg < 1:
1734:         raise ValueError("deg must be a non-negative integer")
1735: 
1736:     # first approximation of roots. We use the fact that the companion
1737:     # matrix is symmetric in this case in order to obtain better zeros.
1738:     c = np.array([0]*deg + [1])
1739:     m = legcompanion(c)
1740:     x = la.eigvalsh(m)
1741: 
1742:     # improve roots by one application of Newton
1743:     dy = legval(x, c)
1744:     df = legval(x, legder(c))
1745:     x -= dy/df
1746: 
1747:     # compute the weights. We scale the factor to avoid possible numerical
1748:     # overflow.
1749:     fm = legval(x, c[1:])
1750:     fm /= np.abs(fm).max()
1751:     df /= np.abs(df).max()
1752:     w = 1/(fm * df)
1753: 
1754:     # for Legendre we can also symmetrize
1755:     w = (w + w[::-1])/2
1756:     x = (x - x[::-1])/2
1757: 
1758:     # scale w to get the right value
1759:     w *= 2. / w.sum()
1760: 
1761:     return x, w
1762: 
1763: 
1764: def legweight(x):
1765:     '''
1766:     Weight function of the Legendre polynomials.
1767: 
1768:     The weight function is :math:`1` and the interval of integration is
1769:     :math:`[-1, 1]`. The Legendre polynomials are orthogonal, but not
1770:     normalized, with respect to this weight function.
1771: 
1772:     Parameters
1773:     ----------
1774:     x : array_like
1775:        Values at which the weight function will be computed.
1776: 
1777:     Returns
1778:     -------
1779:     w : ndarray
1780:        The weight function at `x`.
1781: 
1782:     Notes
1783:     -----
1784: 
1785:     .. versionadded::1.7.0
1786: 
1787:     '''
1788:     w = x*0.0 + 1.0
1789:     return w
1790: 
1791: #
1792: # Legendre series class
1793: #
1794: 
1795: class Legendre(ABCPolyBase):
1796:     '''A Legendre series class.
1797: 
1798:     The Legendre class provides the standard Python numerical methods
1799:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
1800:     attributes and methods listed in the `ABCPolyBase` documentation.
1801: 
1802:     Parameters
1803:     ----------
1804:     coef : array_like
1805:         Legendre coefficients in order of increasing degree, i.e.,
1806:         ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``.
1807:     domain : (2,) array_like, optional
1808:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
1809:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
1810:         The default value is [-1, 1].
1811:     window : (2,) array_like, optional
1812:         Window, see `domain` for its use. The default value is [-1, 1].
1813: 
1814:         .. versionadded:: 1.6.0
1815: 
1816:     '''
1817:     # Virtual Functions
1818:     _add = staticmethod(legadd)
1819:     _sub = staticmethod(legsub)
1820:     _mul = staticmethod(legmul)
1821:     _div = staticmethod(legdiv)
1822:     _pow = staticmethod(legpow)
1823:     _val = staticmethod(legval)
1824:     _int = staticmethod(legint)
1825:     _der = staticmethod(legder)
1826:     _fit = staticmethod(legfit)
1827:     _line = staticmethod(legline)
1828:     _roots = staticmethod(legroots)
1829:     _fromroots = staticmethod(legfromroots)
1830: 
1831:     # Virtual properties
1832:     nickname = 'leg'
1833:     domain = np.array(legdomain)
1834:     window = np.array(legdomain)
1835: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_173658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', '\nLegendre Series (:mod: `numpy.polynomial.legendre`)\n===================================================\n\n.. currentmodule:: numpy.polynomial.polynomial\n\nThis module provides a number of objects (mostly functions) useful for\ndealing with Legendre series, including a `Legendre` class that\nencapsulates the usual arithmetic operations.  (General information\non how this module represents and works with such polynomials is in the\ndocstring for its "parent" sub-package, `numpy.polynomial`).\n\nConstants\n---------\n\n.. autosummary::\n   :toctree: generated/\n\n   legdomain            Legendre series default domain, [-1,1].\n   legzero              Legendre series that evaluates identically to 0.\n   legone               Legendre series that evaluates identically to 1.\n   legx                 Legendre series for the identity map, ``f(x) = x``.\n\nArithmetic\n----------\n\n.. autosummary::\n   :toctree: generated/\n\n   legmulx              multiply a Legendre series in P_i(x) by x.\n   legadd               add two Legendre series.\n   legsub               subtract one Legendre series from another.\n   legmul               multiply two Legendre series.\n   legdiv               divide one Legendre series by another.\n   legpow               raise a Legendre series to an positive integer power\n   legval               evaluate a Legendre series at given points.\n   legval2d             evaluate a 2D Legendre series at given points.\n   legval3d             evaluate a 3D Legendre series at given points.\n   leggrid2d            evaluate a 2D Legendre series on a Cartesian product.\n   leggrid3d            evaluate a 3D Legendre series on a Cartesian product.\n\nCalculus\n--------\n\n.. autosummary::\n   :toctree: generated/\n\n   legder               differentiate a Legendre series.\n   legint               integrate a Legendre series.\n\nMisc Functions\n--------------\n\n.. autosummary::\n   :toctree: generated/\n\n   legfromroots          create a Legendre series with specified roots.\n   legroots              find the roots of a Legendre series.\n   legvander             Vandermonde-like matrix for Legendre polynomials.\n   legvander2d           Vandermonde-like matrix for 2D power series.\n   legvander3d           Vandermonde-like matrix for 3D power series.\n   leggauss              Gauss-Legendre quadrature, points and weights.\n   legweight             Legendre weight function.\n   legcompanion          symmetrized companion matrix in Legendre form.\n   legfit                least-squares fit returning a Legendre series.\n   legtrim               trim leading coefficients from a Legendre series.\n   legline               Legendre series representing given straight line.\n   leg2poly              convert a Legendre series to a polynomial.\n   poly2leg              convert a polynomial to a Legendre series.\n\nClasses\n-------\n    Legendre            A Legendre series class.\n\nSee also\n--------\nnumpy.polynomial.polynomial\nnumpy.polynomial.chebyshev\nnumpy.polynomial.laguerre\nnumpy.polynomial.hermite\nnumpy.polynomial.hermite_e\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 86, 0))

# 'import warnings' statement (line 86)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 86, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 87, 0))

# 'import numpy' statement (line 87)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_173659 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 87, 0), 'numpy')

if (type(import_173659) is not StypyTypeError):

    if (import_173659 != 'pyd_module'):
        __import__(import_173659)
        sys_modules_173660 = sys.modules[import_173659]
        import_module(stypy.reporting.localization.Localization(__file__, 87, 0), 'np', sys_modules_173660.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 87, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'numpy', import_173659)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 88, 0))

# 'import numpy.linalg' statement (line 88)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_173661 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 88, 0), 'numpy.linalg')

if (type(import_173661) is not StypyTypeError):

    if (import_173661 != 'pyd_module'):
        __import__(import_173661)
        sys_modules_173662 = sys.modules[import_173661]
        import_module(stypy.reporting.localization.Localization(__file__, 88, 0), 'la', sys_modules_173662.module_type_store, module_type_store)
    else:
        import numpy.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 88, 0), 'la', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'numpy.linalg', import_173661)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 90, 0))

# 'from numpy.polynomial import pu' statement (line 90)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_173663 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy.polynomial')

if (type(import_173663) is not StypyTypeError):

    if (import_173663 != 'pyd_module'):
        __import__(import_173663)
        sys_modules_173664 = sys.modules[import_173663]
        import_from_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy.polynomial', sys_modules_173664.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 90, 0), __file__, sys_modules_173664, sys_modules_173664.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'numpy.polynomial', import_173663)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 0))

# 'from numpy.polynomial._polybase import ABCPolyBase' statement (line 91)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_173665 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'numpy.polynomial._polybase')

if (type(import_173665) is not StypyTypeError):

    if (import_173665 != 'pyd_module'):
        __import__(import_173665)
        sys_modules_173666 = sys.modules[import_173665]
        import_from_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'numpy.polynomial._polybase', sys_modules_173666.module_type_store, module_type_store, ['ABCPolyBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 91, 0), __file__, sys_modules_173666, sys_modules_173666.module_type_store, module_type_store)
    else:
        from numpy.polynomial._polybase import ABCPolyBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'numpy.polynomial._polybase', None, module_type_store, ['ABCPolyBase'], [ABCPolyBase])

else:
    # Assigning a type to the variable 'numpy.polynomial._polybase' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'numpy.polynomial._polybase', import_173665)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 93):

# Assigning a List to a Name (line 93):
__all__ = ['legzero', 'legone', 'legx', 'legdomain', 'legline', 'legadd', 'legsub', 'legmulx', 'legmul', 'legdiv', 'legpow', 'legval', 'legder', 'legint', 'leg2poly', 'poly2leg', 'legfromroots', 'legvander', 'legfit', 'legtrim', 'legroots', 'Legendre', 'legval2d', 'legval3d', 'leggrid2d', 'leggrid3d', 'legvander2d', 'legvander3d', 'legcompanion', 'leggauss', 'legweight']
module_type_store.set_exportable_members(['legzero', 'legone', 'legx', 'legdomain', 'legline', 'legadd', 'legsub', 'legmulx', 'legmul', 'legdiv', 'legpow', 'legval', 'legder', 'legint', 'leg2poly', 'poly2leg', 'legfromroots', 'legvander', 'legfit', 'legtrim', 'legroots', 'Legendre', 'legval2d', 'legval3d', 'leggrid2d', 'leggrid3d', 'legvander2d', 'legvander3d', 'legcompanion', 'leggauss', 'legweight'])

# Obtaining an instance of the builtin type 'list' (line 93)
list_173667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 93)
# Adding element type (line 93)
str_173668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'str', 'legzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173668)
# Adding element type (line 93)
str_173669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 15), 'str', 'legone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173669)
# Adding element type (line 93)
str_173670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'legx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173670)
# Adding element type (line 93)
str_173671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'str', 'legdomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173671)
# Adding element type (line 93)
str_173672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 46), 'str', 'legline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173672)
# Adding element type (line 93)
str_173673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 57), 'str', 'legadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173673)
# Adding element type (line 93)
str_173674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'str', 'legsub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173674)
# Adding element type (line 93)
str_173675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 14), 'str', 'legmulx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173675)
# Adding element type (line 93)
str_173676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'str', 'legmul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173676)
# Adding element type (line 93)
str_173677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 35), 'str', 'legdiv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173677)
# Adding element type (line 93)
str_173678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 45), 'str', 'legpow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173678)
# Adding element type (line 93)
str_173679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 55), 'str', 'legval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173679)
# Adding element type (line 93)
str_173680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 65), 'str', 'legder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173680)
# Adding element type (line 93)
str_173681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'str', 'legint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173681)
# Adding element type (line 93)
str_173682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 14), 'str', 'leg2poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173682)
# Adding element type (line 93)
str_173683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'str', 'poly2leg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173683)
# Adding element type (line 93)
str_173684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 38), 'str', 'legfromroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173684)
# Adding element type (line 93)
str_173685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 54), 'str', 'legvander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173685)
# Adding element type (line 93)
str_173686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'str', 'legfit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173686)
# Adding element type (line 93)
str_173687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 14), 'str', 'legtrim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173687)
# Adding element type (line 93)
str_173688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'str', 'legroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173688)
# Adding element type (line 93)
str_173689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 37), 'str', 'Legendre')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173689)
# Adding element type (line 93)
str_173690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 49), 'str', 'legval2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173690)
# Adding element type (line 93)
str_173691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 61), 'str', 'legval3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173691)
# Adding element type (line 93)
str_173692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'str', 'leggrid2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173692)
# Adding element type (line 93)
str_173693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 17), 'str', 'leggrid3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173693)
# Adding element type (line 93)
str_173694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'str', 'legvander2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173694)
# Adding element type (line 93)
str_173695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 45), 'str', 'legvander3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173695)
# Adding element type (line 93)
str_173696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 60), 'str', 'legcompanion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173696)
# Adding element type (line 93)
str_173697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'str', 'leggauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173697)
# Adding element type (line 93)
str_173698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'str', 'legweight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 10), list_173667, str_173698)

# Assigning a type to the variable '__all__' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), '__all__', list_173667)

# Assigning a Attribute to a Name (line 101):

# Assigning a Attribute to a Name (line 101):
# Getting the type of 'pu' (line 101)
pu_173699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'pu')
# Obtaining the member 'trimcoef' of a type (line 101)
trimcoef_173700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 10), pu_173699, 'trimcoef')
# Assigning a type to the variable 'legtrim' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'legtrim', trimcoef_173700)

@norecursion
def poly2leg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poly2leg'
    module_type_store = module_type_store.open_function_context('poly2leg', 104, 0, False)
    
    # Passed parameters checking function
    poly2leg.stypy_localization = localization
    poly2leg.stypy_type_of_self = None
    poly2leg.stypy_type_store = module_type_store
    poly2leg.stypy_function_name = 'poly2leg'
    poly2leg.stypy_param_names_list = ['pol']
    poly2leg.stypy_varargs_param_name = None
    poly2leg.stypy_kwargs_param_name = None
    poly2leg.stypy_call_defaults = defaults
    poly2leg.stypy_call_varargs = varargs
    poly2leg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poly2leg', ['pol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poly2leg', localization, ['pol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poly2leg(...)' code ##################

    str_173701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', '\n    Convert a polynomial to a Legendre series.\n\n    Convert an array representing the coefficients of a polynomial (relative\n    to the "standard" basis) ordered from lowest degree to highest, to an\n    array of the coefficients of the equivalent Legendre series, ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    pol : array_like\n        1-D array containing the polynomial coefficients\n\n    Returns\n    -------\n    c : ndarray\n        1-D array containing the coefficients of the equivalent Legendre\n        series.\n\n    See Also\n    --------\n    leg2poly\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> p = P.Polynomial(np.arange(4))\n    >>> p\n    Polynomial([ 0.,  1.,  2.,  3.], [-1.,  1.])\n    >>> c = P.Legendre(P.poly2leg(p.coef))\n    >>> c\n    Legendre([ 1.  ,  3.25,  1.  ,  0.75], [-1.,  1.])\n\n    ')
    
    # Assigning a Call to a List (line 144):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 144)
    # Processing the call arguments (line 144)
    
    # Obtaining an instance of the builtin type 'list' (line 144)
    list_173704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 144)
    # Adding element type (line 144)
    # Getting the type of 'pol' (line 144)
    pol_173705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'pol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 25), list_173704, pol_173705)
    
    # Processing the call keyword arguments (line 144)
    kwargs_173706 = {}
    # Getting the type of 'pu' (line 144)
    pu_173702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 144)
    as_series_173703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), pu_173702, 'as_series')
    # Calling as_series(args, kwargs) (line 144)
    as_series_call_result_173707 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), as_series_173703, *[list_173704], **kwargs_173706)
    
    # Assigning a type to the variable 'call_assignment_173603' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'call_assignment_173603', as_series_call_result_173707)
    
    # Assigning a Call to a Name (line 144):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173711 = {}
    # Getting the type of 'call_assignment_173603' (line 144)
    call_assignment_173603_173708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'call_assignment_173603', False)
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___173709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 4), call_assignment_173603_173708, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173712 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173709, *[int_173710], **kwargs_173711)
    
    # Assigning a type to the variable 'call_assignment_173604' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'call_assignment_173604', getitem___call_result_173712)
    
    # Assigning a Name to a Name (line 144):
    # Getting the type of 'call_assignment_173604' (line 144)
    call_assignment_173604_173713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'call_assignment_173604')
    # Assigning a type to the variable 'pol' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 5), 'pol', call_assignment_173604_173713)
    
    # Assigning a BinOp to a Name (line 145):
    
    # Assigning a BinOp to a Name (line 145):
    
    # Call to len(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'pol' (line 145)
    pol_173715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 14), 'pol', False)
    # Processing the call keyword arguments (line 145)
    kwargs_173716 = {}
    # Getting the type of 'len' (line 145)
    len_173714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 10), 'len', False)
    # Calling len(args, kwargs) (line 145)
    len_call_result_173717 = invoke(stypy.reporting.localization.Localization(__file__, 145, 10), len_173714, *[pol_173715], **kwargs_173716)
    
    int_173718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'int')
    # Applying the binary operator '-' (line 145)
    result_sub_173719 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 10), '-', len_call_result_173717, int_173718)
    
    # Assigning a type to the variable 'deg' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'deg', result_sub_173719)
    
    # Assigning a Num to a Name (line 146):
    
    # Assigning a Num to a Name (line 146):
    int_173720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 10), 'int')
    # Assigning a type to the variable 'res' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'res', int_173720)
    
    
    # Call to range(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'deg' (line 147)
    deg_173722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'deg', False)
    int_173723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'int')
    int_173724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 28), 'int')
    # Processing the call keyword arguments (line 147)
    kwargs_173725 = {}
    # Getting the type of 'range' (line 147)
    range_173721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'range', False)
    # Calling range(args, kwargs) (line 147)
    range_call_result_173726 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), range_173721, *[deg_173722, int_173723, int_173724], **kwargs_173725)
    
    # Testing the type of a for loop iterable (line 147)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 4), range_call_result_173726)
    # Getting the type of the for loop variable (line 147)
    for_loop_var_173727 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 4), range_call_result_173726)
    # Assigning a type to the variable 'i' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'i', for_loop_var_173727)
    # SSA begins for a for statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to legadd(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to legmulx(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'res' (line 148)
    res_173730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'res', False)
    # Processing the call keyword arguments (line 148)
    kwargs_173731 = {}
    # Getting the type of 'legmulx' (line 148)
    legmulx_173729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'legmulx', False)
    # Calling legmulx(args, kwargs) (line 148)
    legmulx_call_result_173732 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), legmulx_173729, *[res_173730], **kwargs_173731)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 148)
    i_173733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'i', False)
    # Getting the type of 'pol' (line 148)
    pol_173734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'pol', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___173735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), pol_173734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_173736 = invoke(stypy.reporting.localization.Localization(__file__, 148, 35), getitem___173735, i_173733)
    
    # Processing the call keyword arguments (line 148)
    kwargs_173737 = {}
    # Getting the type of 'legadd' (line 148)
    legadd_173728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'legadd', False)
    # Calling legadd(args, kwargs) (line 148)
    legadd_call_result_173738 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), legadd_173728, *[legmulx_call_result_173732, subscript_call_result_173736], **kwargs_173737)
    
    # Assigning a type to the variable 'res' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'res', legadd_call_result_173738)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 149)
    res_173739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type', res_173739)
    
    # ################# End of 'poly2leg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poly2leg' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_173740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173740)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poly2leg'
    return stypy_return_type_173740

# Assigning a type to the variable 'poly2leg' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'poly2leg', poly2leg)

@norecursion
def leg2poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'leg2poly'
    module_type_store = module_type_store.open_function_context('leg2poly', 152, 0, False)
    
    # Passed parameters checking function
    leg2poly.stypy_localization = localization
    leg2poly.stypy_type_of_self = None
    leg2poly.stypy_type_store = module_type_store
    leg2poly.stypy_function_name = 'leg2poly'
    leg2poly.stypy_param_names_list = ['c']
    leg2poly.stypy_varargs_param_name = None
    leg2poly.stypy_kwargs_param_name = None
    leg2poly.stypy_call_defaults = defaults
    leg2poly.stypy_call_varargs = varargs
    leg2poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'leg2poly', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'leg2poly', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'leg2poly(...)' code ##################

    str_173741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'str', '\n    Convert a Legendre series to a polynomial.\n\n    Convert an array representing the coefficients of a Legendre series,\n    ordered from lowest degree to highest, to an array of the coefficients\n    of the equivalent polynomial (relative to the "standard" basis) ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array containing the Legendre series coefficients, ordered\n        from lowest order term to highest.\n\n    Returns\n    -------\n    pol : ndarray\n        1-D array containing the coefficients of the equivalent polynomial\n        (relative to the "standard" basis) ordered from lowest order term\n        to highest.\n\n    See Also\n    --------\n    poly2leg\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> c = P.Legendre(range(4))\n    >>> c\n    Legendre([ 0.,  1.,  2.,  3.], [-1.,  1.])\n    >>> p = c.convert(kind=P.Polynomial)\n    >>> p\n    Polynomial([-1. , -3.5,  3. ,  7.5], [-1.,  1.])\n    >>> P.leg2poly(range(4))\n    array([-1. , -3.5,  3. ,  7.5])\n\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 196, 4))
    
    # 'from numpy.polynomial.polynomial import polyadd, polysub, polymulx' statement (line 196)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_173742 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy.polynomial.polynomial')

    if (type(import_173742) is not StypyTypeError):

        if (import_173742 != 'pyd_module'):
            __import__(import_173742)
            sys_modules_173743 = sys.modules[import_173742]
            import_from_module(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy.polynomial.polynomial', sys_modules_173743.module_type_store, module_type_store, ['polyadd', 'polysub', 'polymulx'])
            nest_module(stypy.reporting.localization.Localization(__file__, 196, 4), __file__, sys_modules_173743, sys_modules_173743.module_type_store, module_type_store)
        else:
            from numpy.polynomial.polynomial import polyadd, polysub, polymulx

            import_from_module(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy.polynomial.polynomial', None, module_type_store, ['polyadd', 'polysub', 'polymulx'], [polyadd, polysub, polymulx])

    else:
        # Assigning a type to the variable 'numpy.polynomial.polynomial' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy.polynomial.polynomial', import_173742)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Assigning a Call to a List (line 198):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'list' (line 198)
    list_173746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 198)
    # Adding element type (line 198)
    # Getting the type of 'c' (line 198)
    c_173747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), list_173746, c_173747)
    
    # Processing the call keyword arguments (line 198)
    kwargs_173748 = {}
    # Getting the type of 'pu' (line 198)
    pu_173744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 198)
    as_series_173745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 10), pu_173744, 'as_series')
    # Calling as_series(args, kwargs) (line 198)
    as_series_call_result_173749 = invoke(stypy.reporting.localization.Localization(__file__, 198, 10), as_series_173745, *[list_173746], **kwargs_173748)
    
    # Assigning a type to the variable 'call_assignment_173605' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'call_assignment_173605', as_series_call_result_173749)
    
    # Assigning a Call to a Name (line 198):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173753 = {}
    # Getting the type of 'call_assignment_173605' (line 198)
    call_assignment_173605_173750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'call_assignment_173605', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___173751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 4), call_assignment_173605_173750, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173754 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173751, *[int_173752], **kwargs_173753)
    
    # Assigning a type to the variable 'call_assignment_173606' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'call_assignment_173606', getitem___call_result_173754)
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'call_assignment_173606' (line 198)
    call_assignment_173606_173755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'call_assignment_173606')
    # Assigning a type to the variable 'c' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 5), 'c', call_assignment_173606_173755)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to len(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'c' (line 199)
    c_173757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'c', False)
    # Processing the call keyword arguments (line 199)
    kwargs_173758 = {}
    # Getting the type of 'len' (line 199)
    len_173756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'len', False)
    # Calling len(args, kwargs) (line 199)
    len_call_result_173759 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), len_173756, *[c_173757], **kwargs_173758)
    
    # Assigning a type to the variable 'n' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'n', len_call_result_173759)
    
    
    # Getting the type of 'n' (line 200)
    n_173760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 7), 'n')
    int_173761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 11), 'int')
    # Applying the binary operator '<' (line 200)
    result_lt_173762 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 7), '<', n_173760, int_173761)
    
    # Testing the type of an if condition (line 200)
    if_condition_173763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 4), result_lt_173762)
    # Assigning a type to the variable 'if_condition_173763' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'if_condition_173763', if_condition_173763)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 201)
    c_173764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', c_173764)
    # SSA branch for the else part of an if statement (line 200)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 203):
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_173765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 15), 'int')
    # Getting the type of 'c' (line 203)
    c_173766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___173767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 13), c_173766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_173768 = invoke(stypy.reporting.localization.Localization(__file__, 203, 13), getitem___173767, int_173765)
    
    # Assigning a type to the variable 'c0' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'c0', subscript_call_result_173768)
    
    # Assigning a Subscript to a Name (line 204):
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_173769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 15), 'int')
    # Getting the type of 'c' (line 204)
    c_173770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___173771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 13), c_173770, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_173772 = invoke(stypy.reporting.localization.Localization(__file__, 204, 13), getitem___173771, int_173769)
    
    # Assigning a type to the variable 'c1' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'c1', subscript_call_result_173772)
    
    
    # Call to range(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'n' (line 206)
    n_173774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'n', False)
    int_173775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 27), 'int')
    # Applying the binary operator '-' (line 206)
    result_sub_173776 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 23), '-', n_173774, int_173775)
    
    int_173777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'int')
    int_173778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 33), 'int')
    # Processing the call keyword arguments (line 206)
    kwargs_173779 = {}
    # Getting the type of 'range' (line 206)
    range_173773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'range', False)
    # Calling range(args, kwargs) (line 206)
    range_call_result_173780 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), range_173773, *[result_sub_173776, int_173777, int_173778], **kwargs_173779)
    
    # Testing the type of a for loop iterable (line 206)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 8), range_call_result_173780)
    # Getting the type of the for loop variable (line 206)
    for_loop_var_173781 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 8), range_call_result_173780)
    # Assigning a type to the variable 'i' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'i', for_loop_var_173781)
    # SSA begins for a for statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 207):
    
    # Assigning a Name to a Name (line 207):
    # Getting the type of 'c0' (line 207)
    c0_173782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'tmp', c0_173782)
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to polysub(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 208)
    i_173784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'i', False)
    int_173785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 31), 'int')
    # Applying the binary operator '-' (line 208)
    result_sub_173786 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 27), '-', i_173784, int_173785)
    
    # Getting the type of 'c' (line 208)
    c_173787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 25), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___173788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 25), c_173787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_173789 = invoke(stypy.reporting.localization.Localization(__file__, 208, 25), getitem___173788, result_sub_173786)
    
    # Getting the type of 'c1' (line 208)
    c1_173790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 'c1', False)
    # Getting the type of 'i' (line 208)
    i_173791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 40), 'i', False)
    int_173792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'int')
    # Applying the binary operator '-' (line 208)
    result_sub_173793 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 40), '-', i_173791, int_173792)
    
    # Applying the binary operator '*' (line 208)
    result_mul_173794 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 36), '*', c1_173790, result_sub_173793)
    
    # Getting the type of 'i' (line 208)
    i_173795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'i', False)
    # Applying the binary operator 'div' (line 208)
    result_div_173796 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 35), 'div', result_mul_173794, i_173795)
    
    # Processing the call keyword arguments (line 208)
    kwargs_173797 = {}
    # Getting the type of 'polysub' (line 208)
    polysub_173783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'polysub', False)
    # Calling polysub(args, kwargs) (line 208)
    polysub_call_result_173798 = invoke(stypy.reporting.localization.Localization(__file__, 208, 17), polysub_173783, *[subscript_call_result_173789, result_div_173796], **kwargs_173797)
    
    # Assigning a type to the variable 'c0' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'c0', polysub_call_result_173798)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to polyadd(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'tmp' (line 209)
    tmp_173800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'tmp', False)
    
    # Call to polymulx(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'c1' (line 209)
    c1_173802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'c1', False)
    # Processing the call keyword arguments (line 209)
    kwargs_173803 = {}
    # Getting the type of 'polymulx' (line 209)
    polymulx_173801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 31), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 209)
    polymulx_call_result_173804 = invoke(stypy.reporting.localization.Localization(__file__, 209, 31), polymulx_173801, *[c1_173802], **kwargs_173803)
    
    int_173805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 45), 'int')
    # Getting the type of 'i' (line 209)
    i_173806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 47), 'i', False)
    # Applying the binary operator '*' (line 209)
    result_mul_173807 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 45), '*', int_173805, i_173806)
    
    int_173808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 51), 'int')
    # Applying the binary operator '-' (line 209)
    result_sub_173809 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 45), '-', result_mul_173807, int_173808)
    
    # Applying the binary operator '*' (line 209)
    result_mul_173810 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 31), '*', polymulx_call_result_173804, result_sub_173809)
    
    # Getting the type of 'i' (line 209)
    i_173811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 55), 'i', False)
    # Applying the binary operator 'div' (line 209)
    result_div_173812 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 30), 'div', result_mul_173810, i_173811)
    
    # Processing the call keyword arguments (line 209)
    kwargs_173813 = {}
    # Getting the type of 'polyadd' (line 209)
    polyadd_173799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 209)
    polyadd_call_result_173814 = invoke(stypy.reporting.localization.Localization(__file__, 209, 17), polyadd_173799, *[tmp_173800, result_div_173812], **kwargs_173813)
    
    # Assigning a type to the variable 'c1' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'c1', polyadd_call_result_173814)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to polyadd(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'c0' (line 210)
    c0_173816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'c0', False)
    
    # Call to polymulx(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'c1' (line 210)
    c1_173818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'c1', False)
    # Processing the call keyword arguments (line 210)
    kwargs_173819 = {}
    # Getting the type of 'polymulx' (line 210)
    polymulx_173817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 210)
    polymulx_call_result_173820 = invoke(stypy.reporting.localization.Localization(__file__, 210, 27), polymulx_173817, *[c1_173818], **kwargs_173819)
    
    # Processing the call keyword arguments (line 210)
    kwargs_173821 = {}
    # Getting the type of 'polyadd' (line 210)
    polyadd_173815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 210)
    polyadd_call_result_173822 = invoke(stypy.reporting.localization.Localization(__file__, 210, 15), polyadd_173815, *[c0_173816, polymulx_call_result_173820], **kwargs_173821)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'stypy_return_type', polyadd_call_result_173822)
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'leg2poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'leg2poly' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_173823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173823)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'leg2poly'
    return stypy_return_type_173823

# Assigning a type to the variable 'leg2poly' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'leg2poly', leg2poly)

# Assigning a Call to a Name (line 218):

# Assigning a Call to a Name (line 218):

# Call to array(...): (line 218)
# Processing the call arguments (line 218)

# Obtaining an instance of the builtin type 'list' (line 218)
list_173826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 218)
# Adding element type (line 218)
int_173827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_173826, int_173827)
# Adding element type (line 218)
int_173828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_173826, int_173828)

# Processing the call keyword arguments (line 218)
kwargs_173829 = {}
# Getting the type of 'np' (line 218)
np_173824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'np', False)
# Obtaining the member 'array' of a type (line 218)
array_173825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), np_173824, 'array')
# Calling array(args, kwargs) (line 218)
array_call_result_173830 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), array_173825, *[list_173826], **kwargs_173829)

# Assigning a type to the variable 'legdomain' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'legdomain', array_call_result_173830)

# Assigning a Call to a Name (line 221):

# Assigning a Call to a Name (line 221):

# Call to array(...): (line 221)
# Processing the call arguments (line 221)

# Obtaining an instance of the builtin type 'list' (line 221)
list_173833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 221)
# Adding element type (line 221)
int_173834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 19), list_173833, int_173834)

# Processing the call keyword arguments (line 221)
kwargs_173835 = {}
# Getting the type of 'np' (line 221)
np_173831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 10), 'np', False)
# Obtaining the member 'array' of a type (line 221)
array_173832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 10), np_173831, 'array')
# Calling array(args, kwargs) (line 221)
array_call_result_173836 = invoke(stypy.reporting.localization.Localization(__file__, 221, 10), array_173832, *[list_173833], **kwargs_173835)

# Assigning a type to the variable 'legzero' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'legzero', array_call_result_173836)

# Assigning a Call to a Name (line 224):

# Assigning a Call to a Name (line 224):

# Call to array(...): (line 224)
# Processing the call arguments (line 224)

# Obtaining an instance of the builtin type 'list' (line 224)
list_173839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 224)
# Adding element type (line 224)
int_173840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 18), list_173839, int_173840)

# Processing the call keyword arguments (line 224)
kwargs_173841 = {}
# Getting the type of 'np' (line 224)
np_173837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'np', False)
# Obtaining the member 'array' of a type (line 224)
array_173838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 9), np_173837, 'array')
# Calling array(args, kwargs) (line 224)
array_call_result_173842 = invoke(stypy.reporting.localization.Localization(__file__, 224, 9), array_173838, *[list_173839], **kwargs_173841)

# Assigning a type to the variable 'legone' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'legone', array_call_result_173842)

# Assigning a Call to a Name (line 227):

# Assigning a Call to a Name (line 227):

# Call to array(...): (line 227)
# Processing the call arguments (line 227)

# Obtaining an instance of the builtin type 'list' (line 227)
list_173845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 227)
# Adding element type (line 227)
int_173846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), list_173845, int_173846)
# Adding element type (line 227)
int_173847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), list_173845, int_173847)

# Processing the call keyword arguments (line 227)
kwargs_173848 = {}
# Getting the type of 'np' (line 227)
np_173843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'np', False)
# Obtaining the member 'array' of a type (line 227)
array_173844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 7), np_173843, 'array')
# Calling array(args, kwargs) (line 227)
array_call_result_173849 = invoke(stypy.reporting.localization.Localization(__file__, 227, 7), array_173844, *[list_173845], **kwargs_173848)

# Assigning a type to the variable 'legx' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'legx', array_call_result_173849)

@norecursion
def legline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legline'
    module_type_store = module_type_store.open_function_context('legline', 230, 0, False)
    
    # Passed parameters checking function
    legline.stypy_localization = localization
    legline.stypy_type_of_self = None
    legline.stypy_type_store = module_type_store
    legline.stypy_function_name = 'legline'
    legline.stypy_param_names_list = ['off', 'scl']
    legline.stypy_varargs_param_name = None
    legline.stypy_kwargs_param_name = None
    legline.stypy_call_defaults = defaults
    legline.stypy_call_varargs = varargs
    legline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legline', ['off', 'scl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legline', localization, ['off', 'scl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legline(...)' code ##################

    str_173850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', "\n    Legendre series whose graph is a straight line.\n\n\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The specified line is given by ``off + scl*x``.\n\n    Returns\n    -------\n    y : ndarray\n        This module's representation of the Legendre series for\n        ``off + scl*x``.\n\n    See Also\n    --------\n    polyline, chebline\n\n    Examples\n    --------\n    >>> import numpy.polynomial.legendre as L\n    >>> L.legline(3,2)\n    array([3, 2])\n    >>> L.legval(-3, L.legline(3,2)) # should be -3\n    -3.0\n\n    ")
    
    
    # Getting the type of 'scl' (line 260)
    scl_173851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 7), 'scl')
    int_173852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 14), 'int')
    # Applying the binary operator '!=' (line 260)
    result_ne_173853 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), '!=', scl_173851, int_173852)
    
    # Testing the type of an if condition (line 260)
    if_condition_173854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 4), result_ne_173853)
    # Assigning a type to the variable 'if_condition_173854' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'if_condition_173854', if_condition_173854)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 261)
    # Processing the call arguments (line 261)
    
    # Obtaining an instance of the builtin type 'list' (line 261)
    list_173857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 261)
    # Adding element type (line 261)
    # Getting the type of 'off' (line 261)
    off_173858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 24), list_173857, off_173858)
    # Adding element type (line 261)
    # Getting the type of 'scl' (line 261)
    scl_173859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 30), 'scl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 24), list_173857, scl_173859)
    
    # Processing the call keyword arguments (line 261)
    kwargs_173860 = {}
    # Getting the type of 'np' (line 261)
    np_173855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 261)
    array_173856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), np_173855, 'array')
    # Calling array(args, kwargs) (line 261)
    array_call_result_173861 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), array_173856, *[list_173857], **kwargs_173860)
    
    # Assigning a type to the variable 'stypy_return_type' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type', array_call_result_173861)
    # SSA branch for the else part of an if statement (line 260)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 263)
    # Processing the call arguments (line 263)
    
    # Obtaining an instance of the builtin type 'list' (line 263)
    list_173864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 263)
    # Adding element type (line 263)
    # Getting the type of 'off' (line 263)
    off_173865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 24), list_173864, off_173865)
    
    # Processing the call keyword arguments (line 263)
    kwargs_173866 = {}
    # Getting the type of 'np' (line 263)
    np_173862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 263)
    array_173863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), np_173862, 'array')
    # Calling array(args, kwargs) (line 263)
    array_call_result_173867 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), array_173863, *[list_173864], **kwargs_173866)
    
    # Assigning a type to the variable 'stypy_return_type' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'stypy_return_type', array_call_result_173867)
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'legline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legline' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_173868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173868)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legline'
    return stypy_return_type_173868

# Assigning a type to the variable 'legline' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'legline', legline)

@norecursion
def legfromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legfromroots'
    module_type_store = module_type_store.open_function_context('legfromroots', 266, 0, False)
    
    # Passed parameters checking function
    legfromroots.stypy_localization = localization
    legfromroots.stypy_type_of_self = None
    legfromroots.stypy_type_store = module_type_store
    legfromroots.stypy_function_name = 'legfromroots'
    legfromroots.stypy_param_names_list = ['roots']
    legfromroots.stypy_varargs_param_name = None
    legfromroots.stypy_kwargs_param_name = None
    legfromroots.stypy_call_defaults = defaults
    legfromroots.stypy_call_varargs = varargs
    legfromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legfromroots', ['roots'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legfromroots', localization, ['roots'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legfromroots(...)' code ##################

    str_173869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, (-1)), 'str', '\n    Generate a Legendre series with given roots.\n\n    The function returns the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    in Legendre form, where the `r_n` are the roots specified in `roots`.\n    If a zero has multiplicity n, then it must appear in `roots` n times.\n    For instance, if 2 is a root of multiplicity three and 3 is a root of\n    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The\n    roots can appear in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)\n\n    The coefficient of the last term is not generally 1 for monic\n    polynomials in Legendre form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of coefficients.  If all roots are real then `out` is a\n        real array, if some of the roots are complex, then `out` is complex\n        even if all the coefficients in the result are real (see Examples\n        below).\n\n    See Also\n    --------\n    polyfromroots, chebfromroots, lagfromroots, hermfromroots,\n    hermefromroots.\n\n    Examples\n    --------\n    >>> import numpy.polynomial.legendre as L\n    >>> L.legfromroots((-1,0,1)) # x^3 - x relative to the standard basis\n    array([ 0. , -0.4,  0. ,  0.4])\n    >>> j = complex(0,1)\n    >>> L.legfromroots((-j,j)) # x^2 + 1 relative to the standard basis\n    array([ 1.33333333+0.j,  0.00000000+0.j,  0.66666667+0.j])\n\n    ')
    
    
    
    # Call to len(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'roots' (line 315)
    roots_173871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 11), 'roots', False)
    # Processing the call keyword arguments (line 315)
    kwargs_173872 = {}
    # Getting the type of 'len' (line 315)
    len_173870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 7), 'len', False)
    # Calling len(args, kwargs) (line 315)
    len_call_result_173873 = invoke(stypy.reporting.localization.Localization(__file__, 315, 7), len_173870, *[roots_173871], **kwargs_173872)
    
    int_173874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 21), 'int')
    # Applying the binary operator '==' (line 315)
    result_eq_173875 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 7), '==', len_call_result_173873, int_173874)
    
    # Testing the type of an if condition (line 315)
    if_condition_173876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 4), result_eq_173875)
    # Assigning a type to the variable 'if_condition_173876' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'if_condition_173876', if_condition_173876)
    # SSA begins for if statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 316)
    # Processing the call arguments (line 316)
    int_173879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 23), 'int')
    # Processing the call keyword arguments (line 316)
    kwargs_173880 = {}
    # Getting the type of 'np' (line 316)
    np_173877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 316)
    ones_173878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 15), np_173877, 'ones')
    # Calling ones(args, kwargs) (line 316)
    ones_call_result_173881 = invoke(stypy.reporting.localization.Localization(__file__, 316, 15), ones_173878, *[int_173879], **kwargs_173880)
    
    # Assigning a type to the variable 'stypy_return_type' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'stypy_return_type', ones_call_result_173881)
    # SSA branch for the else part of an if statement (line 315)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a List (line 318):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 318)
    # Processing the call arguments (line 318)
    
    # Obtaining an instance of the builtin type 'list' (line 318)
    list_173884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 318)
    # Adding element type (line 318)
    # Getting the type of 'roots' (line 318)
    roots_173885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 31), list_173884, roots_173885)
    
    # Processing the call keyword arguments (line 318)
    # Getting the type of 'False' (line 318)
    False_173886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 45), 'False', False)
    keyword_173887 = False_173886
    kwargs_173888 = {'trim': keyword_173887}
    # Getting the type of 'pu' (line 318)
    pu_173882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 318)
    as_series_173883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 18), pu_173882, 'as_series')
    # Calling as_series(args, kwargs) (line 318)
    as_series_call_result_173889 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), as_series_173883, *[list_173884], **kwargs_173888)
    
    # Assigning a type to the variable 'call_assignment_173607' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'call_assignment_173607', as_series_call_result_173889)
    
    # Assigning a Call to a Name (line 318):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 8), 'int')
    # Processing the call keyword arguments
    kwargs_173893 = {}
    # Getting the type of 'call_assignment_173607' (line 318)
    call_assignment_173607_173890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'call_assignment_173607', False)
    # Obtaining the member '__getitem__' of a type (line 318)
    getitem___173891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), call_assignment_173607_173890, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173894 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173891, *[int_173892], **kwargs_173893)
    
    # Assigning a type to the variable 'call_assignment_173608' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'call_assignment_173608', getitem___call_result_173894)
    
    # Assigning a Name to a Name (line 318):
    # Getting the type of 'call_assignment_173608' (line 318)
    call_assignment_173608_173895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'call_assignment_173608')
    # Assigning a type to the variable 'roots' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 9), 'roots', call_assignment_173608_173895)
    
    # Call to sort(...): (line 319)
    # Processing the call keyword arguments (line 319)
    kwargs_173898 = {}
    # Getting the type of 'roots' (line 319)
    roots_173896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'roots', False)
    # Obtaining the member 'sort' of a type (line 319)
    sort_173897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), roots_173896, 'sort')
    # Calling sort(args, kwargs) (line 319)
    sort_call_result_173899 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), sort_173897, *[], **kwargs_173898)
    
    
    # Assigning a ListComp to a Name (line 320):
    
    # Assigning a ListComp to a Name (line 320):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'roots' (line 320)
    roots_173906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 37), 'roots')
    comprehension_173907 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 13), roots_173906)
    # Assigning a type to the variable 'r' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'r', comprehension_173907)
    
    # Call to legline(...): (line 320)
    # Processing the call arguments (line 320)
    
    # Getting the type of 'r' (line 320)
    r_173901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 22), 'r', False)
    # Applying the 'usub' unary operator (line 320)
    result___neg___173902 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 21), 'usub', r_173901)
    
    int_173903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'int')
    # Processing the call keyword arguments (line 320)
    kwargs_173904 = {}
    # Getting the type of 'legline' (line 320)
    legline_173900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'legline', False)
    # Calling legline(args, kwargs) (line 320)
    legline_call_result_173905 = invoke(stypy.reporting.localization.Localization(__file__, 320, 13), legline_173900, *[result___neg___173902, int_173903], **kwargs_173904)
    
    list_173908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 13), list_173908, legline_call_result_173905)
    # Assigning a type to the variable 'p' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'p', list_173908)
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to len(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'p' (line 321)
    p_173910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'p', False)
    # Processing the call keyword arguments (line 321)
    kwargs_173911 = {}
    # Getting the type of 'len' (line 321)
    len_173909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'len', False)
    # Calling len(args, kwargs) (line 321)
    len_call_result_173912 = invoke(stypy.reporting.localization.Localization(__file__, 321, 12), len_173909, *[p_173910], **kwargs_173911)
    
    # Assigning a type to the variable 'n' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'n', len_call_result_173912)
    
    
    # Getting the type of 'n' (line 322)
    n_173913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 14), 'n')
    int_173914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 18), 'int')
    # Applying the binary operator '>' (line 322)
    result_gt_173915 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 14), '>', n_173913, int_173914)
    
    # Testing the type of an if condition (line 322)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), result_gt_173915)
    # SSA begins for while statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 323):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'n' (line 323)
    n_173917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'n', False)
    int_173918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 29), 'int')
    # Processing the call keyword arguments (line 323)
    kwargs_173919 = {}
    # Getting the type of 'divmod' (line 323)
    divmod_173916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 323)
    divmod_call_result_173920 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), divmod_173916, *[n_173917, int_173918], **kwargs_173919)
    
    # Assigning a type to the variable 'call_assignment_173609' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173609', divmod_call_result_173920)
    
    # Assigning a Call to a Name (line 323):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 12), 'int')
    # Processing the call keyword arguments
    kwargs_173924 = {}
    # Getting the type of 'call_assignment_173609' (line 323)
    call_assignment_173609_173921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173609', False)
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___173922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), call_assignment_173609_173921, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173925 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173922, *[int_173923], **kwargs_173924)
    
    # Assigning a type to the variable 'call_assignment_173610' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173610', getitem___call_result_173925)
    
    # Assigning a Name to a Name (line 323):
    # Getting the type of 'call_assignment_173610' (line 323)
    call_assignment_173610_173926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173610')
    # Assigning a type to the variable 'm' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'm', call_assignment_173610_173926)
    
    # Assigning a Call to a Name (line 323):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 12), 'int')
    # Processing the call keyword arguments
    kwargs_173930 = {}
    # Getting the type of 'call_assignment_173609' (line 323)
    call_assignment_173609_173927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173609', False)
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___173928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), call_assignment_173609_173927, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173931 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173928, *[int_173929], **kwargs_173930)
    
    # Assigning a type to the variable 'call_assignment_173611' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173611', getitem___call_result_173931)
    
    # Assigning a Name to a Name (line 323):
    # Getting the type of 'call_assignment_173611' (line 323)
    call_assignment_173611_173932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'call_assignment_173611')
    # Assigning a type to the variable 'r' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'r', call_assignment_173611_173932)
    
    # Assigning a ListComp to a Name (line 324):
    
    # Assigning a ListComp to a Name (line 324):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'm' (line 324)
    m_173947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 55), 'm', False)
    # Processing the call keyword arguments (line 324)
    kwargs_173948 = {}
    # Getting the type of 'range' (line 324)
    range_173946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 49), 'range', False)
    # Calling range(args, kwargs) (line 324)
    range_call_result_173949 = invoke(stypy.reporting.localization.Localization(__file__, 324, 49), range_173946, *[m_173947], **kwargs_173948)
    
    comprehension_173950 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), range_call_result_173949)
    # Assigning a type to the variable 'i' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'i', comprehension_173950)
    
    # Call to legmul(...): (line 324)
    # Processing the call arguments (line 324)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 324)
    i_173934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'i', False)
    # Getting the type of 'p' (line 324)
    p_173935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 324)
    getitem___173936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 26), p_173935, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 324)
    subscript_call_result_173937 = invoke(stypy.reporting.localization.Localization(__file__, 324, 26), getitem___173936, i_173934)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 324)
    i_173938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'i', False)
    # Getting the type of 'm' (line 324)
    m_173939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 36), 'm', False)
    # Applying the binary operator '+' (line 324)
    result_add_173940 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 34), '+', i_173938, m_173939)
    
    # Getting the type of 'p' (line 324)
    p_173941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 32), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 324)
    getitem___173942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 32), p_173941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 324)
    subscript_call_result_173943 = invoke(stypy.reporting.localization.Localization(__file__, 324, 32), getitem___173942, result_add_173940)
    
    # Processing the call keyword arguments (line 324)
    kwargs_173944 = {}
    # Getting the type of 'legmul' (line 324)
    legmul_173933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'legmul', False)
    # Calling legmul(args, kwargs) (line 324)
    legmul_call_result_173945 = invoke(stypy.reporting.localization.Localization(__file__, 324, 19), legmul_173933, *[subscript_call_result_173937, subscript_call_result_173943], **kwargs_173944)
    
    list_173951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_173951, legmul_call_result_173945)
    # Assigning a type to the variable 'tmp' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'tmp', list_173951)
    
    # Getting the type of 'r' (line 325)
    r_173952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'r')
    # Testing the type of an if condition (line 325)
    if_condition_173953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 12), r_173952)
    # Assigning a type to the variable 'if_condition_173953' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'if_condition_173953', if_condition_173953)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 326):
    
    # Assigning a Call to a Subscript (line 326):
    
    # Call to legmul(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Obtaining the type of the subscript
    int_173955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 36), 'int')
    # Getting the type of 'tmp' (line 326)
    tmp_173956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___173957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), tmp_173956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_173958 = invoke(stypy.reporting.localization.Localization(__file__, 326, 32), getitem___173957, int_173955)
    
    
    # Obtaining the type of the subscript
    int_173959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 42), 'int')
    # Getting the type of 'p' (line 326)
    p_173960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 40), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___173961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 40), p_173960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_173962 = invoke(stypy.reporting.localization.Localization(__file__, 326, 40), getitem___173961, int_173959)
    
    # Processing the call keyword arguments (line 326)
    kwargs_173963 = {}
    # Getting the type of 'legmul' (line 326)
    legmul_173954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'legmul', False)
    # Calling legmul(args, kwargs) (line 326)
    legmul_call_result_173964 = invoke(stypy.reporting.localization.Localization(__file__, 326, 25), legmul_173954, *[subscript_call_result_173958, subscript_call_result_173962], **kwargs_173963)
    
    # Getting the type of 'tmp' (line 326)
    tmp_173965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'tmp')
    int_173966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 20), 'int')
    # Storing an element on a container (line 326)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 16), tmp_173965, (int_173966, legmul_call_result_173964))
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 327):
    
    # Assigning a Name to a Name (line 327):
    # Getting the type of 'tmp' (line 327)
    tmp_173967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'tmp')
    # Assigning a type to the variable 'p' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'p', tmp_173967)
    
    # Assigning a Name to a Name (line 328):
    
    # Assigning a Name to a Name (line 328):
    # Getting the type of 'm' (line 328)
    m_173968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'm')
    # Assigning a type to the variable 'n' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'n', m_173968)
    # SSA join for while statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_173969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 17), 'int')
    # Getting the type of 'p' (line 329)
    p_173970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'p')
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___173971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), p_173970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_173972 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), getitem___173971, int_173969)
    
    # Assigning a type to the variable 'stypy_return_type' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'stypy_return_type', subscript_call_result_173972)
    # SSA join for if statement (line 315)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'legfromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legfromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 266)
    stypy_return_type_173973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173973)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legfromroots'
    return stypy_return_type_173973

# Assigning a type to the variable 'legfromroots' (line 266)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'legfromroots', legfromroots)

@norecursion
def legadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legadd'
    module_type_store = module_type_store.open_function_context('legadd', 332, 0, False)
    
    # Passed parameters checking function
    legadd.stypy_localization = localization
    legadd.stypy_type_of_self = None
    legadd.stypy_type_store = module_type_store
    legadd.stypy_function_name = 'legadd'
    legadd.stypy_param_names_list = ['c1', 'c2']
    legadd.stypy_varargs_param_name = None
    legadd.stypy_kwargs_param_name = None
    legadd.stypy_call_defaults = defaults
    legadd.stypy_call_varargs = varargs
    legadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legadd', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legadd', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legadd(...)' code ##################

    str_173974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, (-1)), 'str', '\n    Add one Legendre series to another.\n\n    Returns the sum of two Legendre series `c1` + `c2`.  The arguments\n    are sequences of coefficients ordered from lowest order term to\n    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Legendre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the Legendre series of their sum.\n\n    See Also\n    --------\n    legsub, legmul, legdiv, legpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the sum of two Legendre series\n    is a Legendre series (without having to "reproject" the result onto\n    the basis set) so addition, just like that of "standard" polynomials,\n    is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial import legendre as L\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> L.legadd(c1,c2)\n    array([ 4.,  4.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 372):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Obtaining an instance of the builtin type 'list' (line 372)
    list_173977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 372)
    # Adding element type (line 372)
    # Getting the type of 'c1' (line 372)
    c1_173978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 28), list_173977, c1_173978)
    # Adding element type (line 372)
    # Getting the type of 'c2' (line 372)
    c2_173979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 28), list_173977, c2_173979)
    
    # Processing the call keyword arguments (line 372)
    kwargs_173980 = {}
    # Getting the type of 'pu' (line 372)
    pu_173975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 372)
    as_series_173976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 15), pu_173975, 'as_series')
    # Calling as_series(args, kwargs) (line 372)
    as_series_call_result_173981 = invoke(stypy.reporting.localization.Localization(__file__, 372, 15), as_series_173976, *[list_173977], **kwargs_173980)
    
    # Assigning a type to the variable 'call_assignment_173612' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173612', as_series_call_result_173981)
    
    # Assigning a Call to a Name (line 372):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173985 = {}
    # Getting the type of 'call_assignment_173612' (line 372)
    call_assignment_173612_173982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173612', False)
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___173983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 4), call_assignment_173612_173982, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173986 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173983, *[int_173984], **kwargs_173985)
    
    # Assigning a type to the variable 'call_assignment_173613' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173613', getitem___call_result_173986)
    
    # Assigning a Name to a Name (line 372):
    # Getting the type of 'call_assignment_173613' (line 372)
    call_assignment_173613_173987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173613')
    # Assigning a type to the variable 'c1' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 5), 'c1', call_assignment_173613_173987)
    
    # Assigning a Call to a Name (line 372):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173991 = {}
    # Getting the type of 'call_assignment_173612' (line 372)
    call_assignment_173612_173988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173612', False)
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___173989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 4), call_assignment_173612_173988, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173992 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173989, *[int_173990], **kwargs_173991)
    
    # Assigning a type to the variable 'call_assignment_173614' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173614', getitem___call_result_173992)
    
    # Assigning a Name to a Name (line 372):
    # Getting the type of 'call_assignment_173614' (line 372)
    call_assignment_173614_173993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'call_assignment_173614')
    # Assigning a type to the variable 'c2' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 9), 'c2', call_assignment_173614_173993)
    
    
    
    # Call to len(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'c1' (line 373)
    c1_173995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'c1', False)
    # Processing the call keyword arguments (line 373)
    kwargs_173996 = {}
    # Getting the type of 'len' (line 373)
    len_173994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 7), 'len', False)
    # Calling len(args, kwargs) (line 373)
    len_call_result_173997 = invoke(stypy.reporting.localization.Localization(__file__, 373, 7), len_173994, *[c1_173995], **kwargs_173996)
    
    
    # Call to len(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'c2' (line 373)
    c2_173999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 21), 'c2', False)
    # Processing the call keyword arguments (line 373)
    kwargs_174000 = {}
    # Getting the type of 'len' (line 373)
    len_173998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'len', False)
    # Calling len(args, kwargs) (line 373)
    len_call_result_174001 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), len_173998, *[c2_173999], **kwargs_174000)
    
    # Applying the binary operator '>' (line 373)
    result_gt_174002 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 7), '>', len_call_result_173997, len_call_result_174001)
    
    # Testing the type of an if condition (line 373)
    if_condition_174003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 4), result_gt_174002)
    # Assigning a type to the variable 'if_condition_174003' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'if_condition_174003', if_condition_174003)
    # SSA begins for if statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 374)
    c1_174004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 374)
    c2_174005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'c2')
    # Obtaining the member 'size' of a type (line 374)
    size_174006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), c2_174005, 'size')
    slice_174007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 374, 8), None, size_174006, None)
    # Getting the type of 'c1' (line 374)
    c1_174008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___174009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), c1_174008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_174010 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), getitem___174009, slice_174007)
    
    # Getting the type of 'c2' (line 374)
    c2_174011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 24), 'c2')
    # Applying the binary operator '+=' (line 374)
    result_iadd_174012 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 8), '+=', subscript_call_result_174010, c2_174011)
    # Getting the type of 'c1' (line 374)
    c1_174013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'c1')
    # Getting the type of 'c2' (line 374)
    c2_174014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'c2')
    # Obtaining the member 'size' of a type (line 374)
    size_174015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), c2_174014, 'size')
    slice_174016 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 374, 8), None, size_174015, None)
    # Storing an element on a container (line 374)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), c1_174013, (slice_174016, result_iadd_174012))
    
    
    # Assigning a Name to a Name (line 375):
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'c1' (line 375)
    c1_174017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'ret', c1_174017)
    # SSA branch for the else part of an if statement (line 373)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'c2' (line 377)
    c2_174018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 377)
    c1_174019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'c1')
    # Obtaining the member 'size' of a type (line 377)
    size_174020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 12), c1_174019, 'size')
    slice_174021 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 377, 8), None, size_174020, None)
    # Getting the type of 'c2' (line 377)
    c2_174022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___174023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), c2_174022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_174024 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), getitem___174023, slice_174021)
    
    # Getting the type of 'c1' (line 377)
    c1_174025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 24), 'c1')
    # Applying the binary operator '+=' (line 377)
    result_iadd_174026 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 8), '+=', subscript_call_result_174024, c1_174025)
    # Getting the type of 'c2' (line 377)
    c2_174027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'c2')
    # Getting the type of 'c1' (line 377)
    c1_174028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'c1')
    # Obtaining the member 'size' of a type (line 377)
    size_174029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 12), c1_174028, 'size')
    slice_174030 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 377, 8), None, size_174029, None)
    # Storing an element on a container (line 377)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 8), c2_174027, (slice_174030, result_iadd_174026))
    
    
    # Assigning a Name to a Name (line 378):
    
    # Assigning a Name to a Name (line 378):
    # Getting the type of 'c2' (line 378)
    c2_174031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'ret', c2_174031)
    # SSA join for if statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'ret' (line 379)
    ret_174034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 22), 'ret', False)
    # Processing the call keyword arguments (line 379)
    kwargs_174035 = {}
    # Getting the type of 'pu' (line 379)
    pu_174032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 379)
    trimseq_174033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), pu_174032, 'trimseq')
    # Calling trimseq(args, kwargs) (line 379)
    trimseq_call_result_174036 = invoke(stypy.reporting.localization.Localization(__file__, 379, 11), trimseq_174033, *[ret_174034], **kwargs_174035)
    
    # Assigning a type to the variable 'stypy_return_type' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type', trimseq_call_result_174036)
    
    # ################# End of 'legadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legadd' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_174037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174037)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legadd'
    return stypy_return_type_174037

# Assigning a type to the variable 'legadd' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'legadd', legadd)

@norecursion
def legsub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legsub'
    module_type_store = module_type_store.open_function_context('legsub', 382, 0, False)
    
    # Passed parameters checking function
    legsub.stypy_localization = localization
    legsub.stypy_type_of_self = None
    legsub.stypy_type_store = module_type_store
    legsub.stypy_function_name = 'legsub'
    legsub.stypy_param_names_list = ['c1', 'c2']
    legsub.stypy_varargs_param_name = None
    legsub.stypy_kwargs_param_name = None
    legsub.stypy_call_defaults = defaults
    legsub.stypy_call_varargs = varargs
    legsub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legsub', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legsub', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legsub(...)' code ##################

    str_174038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'str', '\n    Subtract one Legendre series from another.\n\n    Returns the difference of two Legendre series `c1` - `c2`.  The\n    sequences of coefficients are from lowest order term to highest, i.e.,\n    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Legendre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Legendre series coefficients representing their difference.\n\n    See Also\n    --------\n    legadd, legmul, legdiv, legpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the difference of two Legendre\n    series is a Legendre series (without having to "reproject" the result\n    onto the basis set) so subtraction, just like that of "standard"\n    polynomials, is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial import legendre as L\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> L.legsub(c1,c2)\n    array([-2.,  0.,  2.])\n    >>> L.legsub(c2,c1) # -C.legsub(c1,c2)\n    array([ 2.,  0., -2.])\n\n    ')
    
    # Assigning a Call to a List (line 424):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 424)
    # Processing the call arguments (line 424)
    
    # Obtaining an instance of the builtin type 'list' (line 424)
    list_174041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 424)
    # Adding element type (line 424)
    # Getting the type of 'c1' (line 424)
    c1_174042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 28), list_174041, c1_174042)
    # Adding element type (line 424)
    # Getting the type of 'c2' (line 424)
    c2_174043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 28), list_174041, c2_174043)
    
    # Processing the call keyword arguments (line 424)
    kwargs_174044 = {}
    # Getting the type of 'pu' (line 424)
    pu_174039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 424)
    as_series_174040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 15), pu_174039, 'as_series')
    # Calling as_series(args, kwargs) (line 424)
    as_series_call_result_174045 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), as_series_174040, *[list_174041], **kwargs_174044)
    
    # Assigning a type to the variable 'call_assignment_173615' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173615', as_series_call_result_174045)
    
    # Assigning a Call to a Name (line 424):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174049 = {}
    # Getting the type of 'call_assignment_173615' (line 424)
    call_assignment_173615_174046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173615', False)
    # Obtaining the member '__getitem__' of a type (line 424)
    getitem___174047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 4), call_assignment_173615_174046, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174050 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174047, *[int_174048], **kwargs_174049)
    
    # Assigning a type to the variable 'call_assignment_173616' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173616', getitem___call_result_174050)
    
    # Assigning a Name to a Name (line 424):
    # Getting the type of 'call_assignment_173616' (line 424)
    call_assignment_173616_174051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173616')
    # Assigning a type to the variable 'c1' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 5), 'c1', call_assignment_173616_174051)
    
    # Assigning a Call to a Name (line 424):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174055 = {}
    # Getting the type of 'call_assignment_173615' (line 424)
    call_assignment_173615_174052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173615', False)
    # Obtaining the member '__getitem__' of a type (line 424)
    getitem___174053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 4), call_assignment_173615_174052, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174056 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174053, *[int_174054], **kwargs_174055)
    
    # Assigning a type to the variable 'call_assignment_173617' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173617', getitem___call_result_174056)
    
    # Assigning a Name to a Name (line 424):
    # Getting the type of 'call_assignment_173617' (line 424)
    call_assignment_173617_174057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'call_assignment_173617')
    # Assigning a type to the variable 'c2' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 9), 'c2', call_assignment_173617_174057)
    
    
    
    # Call to len(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'c1' (line 425)
    c1_174059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'c1', False)
    # Processing the call keyword arguments (line 425)
    kwargs_174060 = {}
    # Getting the type of 'len' (line 425)
    len_174058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 7), 'len', False)
    # Calling len(args, kwargs) (line 425)
    len_call_result_174061 = invoke(stypy.reporting.localization.Localization(__file__, 425, 7), len_174058, *[c1_174059], **kwargs_174060)
    
    
    # Call to len(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'c2' (line 425)
    c2_174063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 21), 'c2', False)
    # Processing the call keyword arguments (line 425)
    kwargs_174064 = {}
    # Getting the type of 'len' (line 425)
    len_174062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'len', False)
    # Calling len(args, kwargs) (line 425)
    len_call_result_174065 = invoke(stypy.reporting.localization.Localization(__file__, 425, 17), len_174062, *[c2_174063], **kwargs_174064)
    
    # Applying the binary operator '>' (line 425)
    result_gt_174066 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), '>', len_call_result_174061, len_call_result_174065)
    
    # Testing the type of an if condition (line 425)
    if_condition_174067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 4), result_gt_174066)
    # Assigning a type to the variable 'if_condition_174067' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'if_condition_174067', if_condition_174067)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 426)
    c1_174068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 426)
    c2_174069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'c2')
    # Obtaining the member 'size' of a type (line 426)
    size_174070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), c2_174069, 'size')
    slice_174071 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 426, 8), None, size_174070, None)
    # Getting the type of 'c1' (line 426)
    c1_174072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___174073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), c1_174072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_174074 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), getitem___174073, slice_174071)
    
    # Getting the type of 'c2' (line 426)
    c2_174075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 24), 'c2')
    # Applying the binary operator '-=' (line 426)
    result_isub_174076 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 8), '-=', subscript_call_result_174074, c2_174075)
    # Getting the type of 'c1' (line 426)
    c1_174077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'c1')
    # Getting the type of 'c2' (line 426)
    c2_174078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'c2')
    # Obtaining the member 'size' of a type (line 426)
    size_174079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), c2_174078, 'size')
    slice_174080 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 426, 8), None, size_174079, None)
    # Storing an element on a container (line 426)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 8), c1_174077, (slice_174080, result_isub_174076))
    
    
    # Assigning a Name to a Name (line 427):
    
    # Assigning a Name to a Name (line 427):
    # Getting the type of 'c1' (line 427)
    c1_174081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'ret', c1_174081)
    # SSA branch for the else part of an if statement (line 425)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a UnaryOp to a Name (line 429):
    
    # Assigning a UnaryOp to a Name (line 429):
    
    # Getting the type of 'c2' (line 429)
    c2_174082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 14), 'c2')
    # Applying the 'usub' unary operator (line 429)
    result___neg___174083 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 13), 'usub', c2_174082)
    
    # Assigning a type to the variable 'c2' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'c2', result___neg___174083)
    
    # Getting the type of 'c2' (line 430)
    c2_174084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 430)
    c1_174085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'c1')
    # Obtaining the member 'size' of a type (line 430)
    size_174086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), c1_174085, 'size')
    slice_174087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 430, 8), None, size_174086, None)
    # Getting the type of 'c2' (line 430)
    c2_174088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___174089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), c2_174088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_174090 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), getitem___174089, slice_174087)
    
    # Getting the type of 'c1' (line 430)
    c1_174091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'c1')
    # Applying the binary operator '+=' (line 430)
    result_iadd_174092 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 8), '+=', subscript_call_result_174090, c1_174091)
    # Getting the type of 'c2' (line 430)
    c2_174093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'c2')
    # Getting the type of 'c1' (line 430)
    c1_174094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'c1')
    # Obtaining the member 'size' of a type (line 430)
    size_174095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), c1_174094, 'size')
    slice_174096 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 430, 8), None, size_174095, None)
    # Storing an element on a container (line 430)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 8), c2_174093, (slice_174096, result_iadd_174092))
    
    
    # Assigning a Name to a Name (line 431):
    
    # Assigning a Name to a Name (line 431):
    # Getting the type of 'c2' (line 431)
    c2_174097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'ret', c2_174097)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'ret' (line 432)
    ret_174100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 22), 'ret', False)
    # Processing the call keyword arguments (line 432)
    kwargs_174101 = {}
    # Getting the type of 'pu' (line 432)
    pu_174098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 432)
    trimseq_174099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 11), pu_174098, 'trimseq')
    # Calling trimseq(args, kwargs) (line 432)
    trimseq_call_result_174102 = invoke(stypy.reporting.localization.Localization(__file__, 432, 11), trimseq_174099, *[ret_174100], **kwargs_174101)
    
    # Assigning a type to the variable 'stypy_return_type' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type', trimseq_call_result_174102)
    
    # ################# End of 'legsub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legsub' in the type store
    # Getting the type of 'stypy_return_type' (line 382)
    stypy_return_type_174103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174103)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legsub'
    return stypy_return_type_174103

# Assigning a type to the variable 'legsub' (line 382)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 0), 'legsub', legsub)

@norecursion
def legmulx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legmulx'
    module_type_store = module_type_store.open_function_context('legmulx', 435, 0, False)
    
    # Passed parameters checking function
    legmulx.stypy_localization = localization
    legmulx.stypy_type_of_self = None
    legmulx.stypy_type_store = module_type_store
    legmulx.stypy_function_name = 'legmulx'
    legmulx.stypy_param_names_list = ['c']
    legmulx.stypy_varargs_param_name = None
    legmulx.stypy_kwargs_param_name = None
    legmulx.stypy_call_defaults = defaults
    legmulx.stypy_call_varargs = varargs
    legmulx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legmulx', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legmulx', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legmulx(...)' code ##################

    str_174104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, (-1)), 'str', 'Multiply a Legendre series by x.\n\n    Multiply the Legendre series `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Legendre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    Notes\n    -----\n    The multiplication uses the recursion relationship for Legendre\n    polynomials in the form\n\n    .. math::\n\n      xP_i(x) = ((i + 1)*P_{i + 1}(x) + i*P_{i - 1}(x))/(2i + 1)\n\n    ')
    
    # Assigning a Call to a List (line 464):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 464)
    # Processing the call arguments (line 464)
    
    # Obtaining an instance of the builtin type 'list' (line 464)
    list_174107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 464)
    # Adding element type (line 464)
    # Getting the type of 'c' (line 464)
    c_174108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 23), list_174107, c_174108)
    
    # Processing the call keyword arguments (line 464)
    kwargs_174109 = {}
    # Getting the type of 'pu' (line 464)
    pu_174105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 464)
    as_series_174106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 10), pu_174105, 'as_series')
    # Calling as_series(args, kwargs) (line 464)
    as_series_call_result_174110 = invoke(stypy.reporting.localization.Localization(__file__, 464, 10), as_series_174106, *[list_174107], **kwargs_174109)
    
    # Assigning a type to the variable 'call_assignment_173618' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'call_assignment_173618', as_series_call_result_174110)
    
    # Assigning a Call to a Name (line 464):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174114 = {}
    # Getting the type of 'call_assignment_173618' (line 464)
    call_assignment_173618_174111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'call_assignment_173618', False)
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___174112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 4), call_assignment_173618_174111, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174115 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174112, *[int_174113], **kwargs_174114)
    
    # Assigning a type to the variable 'call_assignment_173619' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'call_assignment_173619', getitem___call_result_174115)
    
    # Assigning a Name to a Name (line 464):
    # Getting the type of 'call_assignment_173619' (line 464)
    call_assignment_173619_174116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'call_assignment_173619')
    # Assigning a type to the variable 'c' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 5), 'c', call_assignment_173619_174116)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'c' (line 466)
    c_174118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 11), 'c', False)
    # Processing the call keyword arguments (line 466)
    kwargs_174119 = {}
    # Getting the type of 'len' (line 466)
    len_174117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 7), 'len', False)
    # Calling len(args, kwargs) (line 466)
    len_call_result_174120 = invoke(stypy.reporting.localization.Localization(__file__, 466, 7), len_174117, *[c_174118], **kwargs_174119)
    
    int_174121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 17), 'int')
    # Applying the binary operator '==' (line 466)
    result_eq_174122 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), '==', len_call_result_174120, int_174121)
    
    
    
    # Obtaining the type of the subscript
    int_174123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 25), 'int')
    # Getting the type of 'c' (line 466)
    c_174124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 23), 'c')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___174125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 23), c_174124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_174126 = invoke(stypy.reporting.localization.Localization(__file__, 466, 23), getitem___174125, int_174123)
    
    int_174127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 31), 'int')
    # Applying the binary operator '==' (line 466)
    result_eq_174128 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 23), '==', subscript_call_result_174126, int_174127)
    
    # Applying the binary operator 'and' (line 466)
    result_and_keyword_174129 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), 'and', result_eq_174122, result_eq_174128)
    
    # Testing the type of an if condition (line 466)
    if_condition_174130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 4), result_and_keyword_174129)
    # Assigning a type to the variable 'if_condition_174130' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'if_condition_174130', if_condition_174130)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 467)
    c_174131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'stypy_return_type', c_174131)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to empty(...): (line 469)
    # Processing the call arguments (line 469)
    
    # Call to len(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'c' (line 469)
    c_174135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 23), 'c', False)
    # Processing the call keyword arguments (line 469)
    kwargs_174136 = {}
    # Getting the type of 'len' (line 469)
    len_174134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'len', False)
    # Calling len(args, kwargs) (line 469)
    len_call_result_174137 = invoke(stypy.reporting.localization.Localization(__file__, 469, 19), len_174134, *[c_174135], **kwargs_174136)
    
    int_174138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 28), 'int')
    # Applying the binary operator '+' (line 469)
    result_add_174139 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 19), '+', len_call_result_174137, int_174138)
    
    # Processing the call keyword arguments (line 469)
    # Getting the type of 'c' (line 469)
    c_174140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), 'c', False)
    # Obtaining the member 'dtype' of a type (line 469)
    dtype_174141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 37), c_174140, 'dtype')
    keyword_174142 = dtype_174141
    kwargs_174143 = {'dtype': keyword_174142}
    # Getting the type of 'np' (line 469)
    np_174132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 469)
    empty_174133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 10), np_174132, 'empty')
    # Calling empty(args, kwargs) (line 469)
    empty_call_result_174144 = invoke(stypy.reporting.localization.Localization(__file__, 469, 10), empty_174133, *[result_add_174139], **kwargs_174143)
    
    # Assigning a type to the variable 'prd' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'prd', empty_call_result_174144)
    
    # Assigning a BinOp to a Subscript (line 470):
    
    # Assigning a BinOp to a Subscript (line 470):
    
    # Obtaining the type of the subscript
    int_174145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 15), 'int')
    # Getting the type of 'c' (line 470)
    c_174146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___174147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 13), c_174146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_174148 = invoke(stypy.reporting.localization.Localization(__file__, 470, 13), getitem___174147, int_174145)
    
    int_174149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 18), 'int')
    # Applying the binary operator '*' (line 470)
    result_mul_174150 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 13), '*', subscript_call_result_174148, int_174149)
    
    # Getting the type of 'prd' (line 470)
    prd_174151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'prd')
    int_174152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 8), 'int')
    # Storing an element on a container (line 470)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 4), prd_174151, (int_174152, result_mul_174150))
    
    # Assigning a Subscript to a Subscript (line 471):
    
    # Assigning a Subscript to a Subscript (line 471):
    
    # Obtaining the type of the subscript
    int_174153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 15), 'int')
    # Getting the type of 'c' (line 471)
    c_174154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___174155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 13), c_174154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_174156 = invoke(stypy.reporting.localization.Localization(__file__, 471, 13), getitem___174155, int_174153)
    
    # Getting the type of 'prd' (line 471)
    prd_174157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'prd')
    int_174158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 8), 'int')
    # Storing an element on a container (line 471)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 4), prd_174157, (int_174158, subscript_call_result_174156))
    
    
    # Call to range(...): (line 472)
    # Processing the call arguments (line 472)
    int_174160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 19), 'int')
    
    # Call to len(...): (line 472)
    # Processing the call arguments (line 472)
    # Getting the type of 'c' (line 472)
    c_174162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 26), 'c', False)
    # Processing the call keyword arguments (line 472)
    kwargs_174163 = {}
    # Getting the type of 'len' (line 472)
    len_174161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'len', False)
    # Calling len(args, kwargs) (line 472)
    len_call_result_174164 = invoke(stypy.reporting.localization.Localization(__file__, 472, 22), len_174161, *[c_174162], **kwargs_174163)
    
    # Processing the call keyword arguments (line 472)
    kwargs_174165 = {}
    # Getting the type of 'range' (line 472)
    range_174159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 13), 'range', False)
    # Calling range(args, kwargs) (line 472)
    range_call_result_174166 = invoke(stypy.reporting.localization.Localization(__file__, 472, 13), range_174159, *[int_174160, len_call_result_174164], **kwargs_174165)
    
    # Testing the type of a for loop iterable (line 472)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 472, 4), range_call_result_174166)
    # Getting the type of the for loop variable (line 472)
    for_loop_var_174167 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 472, 4), range_call_result_174166)
    # Assigning a type to the variable 'i' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'i', for_loop_var_174167)
    # SSA begins for a for statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 473):
    
    # Assigning a BinOp to a Name (line 473):
    # Getting the type of 'i' (line 473)
    i_174168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'i')
    int_174169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 16), 'int')
    # Applying the binary operator '+' (line 473)
    result_add_174170 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 12), '+', i_174168, int_174169)
    
    # Assigning a type to the variable 'j' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'j', result_add_174170)
    
    # Assigning a BinOp to a Name (line 474):
    
    # Assigning a BinOp to a Name (line 474):
    # Getting the type of 'i' (line 474)
    i_174171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'i')
    int_174172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 16), 'int')
    # Applying the binary operator '-' (line 474)
    result_sub_174173 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 12), '-', i_174171, int_174172)
    
    # Assigning a type to the variable 'k' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'k', result_sub_174173)
    
    # Assigning a BinOp to a Name (line 475):
    
    # Assigning a BinOp to a Name (line 475):
    # Getting the type of 'i' (line 475)
    i_174174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'i')
    # Getting the type of 'j' (line 475)
    j_174175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'j')
    # Applying the binary operator '+' (line 475)
    result_add_174176 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 12), '+', i_174174, j_174175)
    
    # Assigning a type to the variable 's' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 's', result_add_174176)
    
    # Assigning a BinOp to a Subscript (line 476):
    
    # Assigning a BinOp to a Subscript (line 476):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 476)
    i_174177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 20), 'i')
    # Getting the type of 'c' (line 476)
    c_174178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'c')
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___174179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 18), c_174178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_174180 = invoke(stypy.reporting.localization.Localization(__file__, 476, 18), getitem___174179, i_174177)
    
    # Getting the type of 'j' (line 476)
    j_174181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 23), 'j')
    # Applying the binary operator '*' (line 476)
    result_mul_174182 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 18), '*', subscript_call_result_174180, j_174181)
    
    # Getting the type of 's' (line 476)
    s_174183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 's')
    # Applying the binary operator 'div' (line 476)
    result_div_174184 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 17), 'div', result_mul_174182, s_174183)
    
    # Getting the type of 'prd' (line 476)
    prd_174185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'prd')
    # Getting the type of 'j' (line 476)
    j_174186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'j')
    # Storing an element on a container (line 476)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 8), prd_174185, (j_174186, result_div_174184))
    
    # Getting the type of 'prd' (line 477)
    prd_174187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'prd')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 477)
    k_174188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'k')
    # Getting the type of 'prd' (line 477)
    prd_174189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'prd')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___174190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), prd_174189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_174191 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___174190, k_174188)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 477)
    i_174192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 21), 'i')
    # Getting the type of 'c' (line 477)
    c_174193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___174194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 19), c_174193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_174195 = invoke(stypy.reporting.localization.Localization(__file__, 477, 19), getitem___174194, i_174192)
    
    # Getting the type of 'i' (line 477)
    i_174196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 24), 'i')
    # Applying the binary operator '*' (line 477)
    result_mul_174197 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 19), '*', subscript_call_result_174195, i_174196)
    
    # Getting the type of 's' (line 477)
    s_174198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 27), 's')
    # Applying the binary operator 'div' (line 477)
    result_div_174199 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 18), 'div', result_mul_174197, s_174198)
    
    # Applying the binary operator '+=' (line 477)
    result_iadd_174200 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 8), '+=', subscript_call_result_174191, result_div_174199)
    # Getting the type of 'prd' (line 477)
    prd_174201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'prd')
    # Getting the type of 'k' (line 477)
    k_174202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'k')
    # Storing an element on a container (line 477)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 8), prd_174201, (k_174202, result_iadd_174200))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 478)
    prd_174203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type', prd_174203)
    
    # ################# End of 'legmulx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legmulx' in the type store
    # Getting the type of 'stypy_return_type' (line 435)
    stypy_return_type_174204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legmulx'
    return stypy_return_type_174204

# Assigning a type to the variable 'legmulx' (line 435)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'legmulx', legmulx)

@norecursion
def legmul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legmul'
    module_type_store = module_type_store.open_function_context('legmul', 481, 0, False)
    
    # Passed parameters checking function
    legmul.stypy_localization = localization
    legmul.stypy_type_of_self = None
    legmul.stypy_type_store = module_type_store
    legmul.stypy_function_name = 'legmul'
    legmul.stypy_param_names_list = ['c1', 'c2']
    legmul.stypy_varargs_param_name = None
    legmul.stypy_kwargs_param_name = None
    legmul.stypy_call_defaults = defaults
    legmul.stypy_call_varargs = varargs
    legmul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legmul', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legmul', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legmul(...)' code ##################

    str_174205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, (-1)), 'str', '\n    Multiply one Legendre series by another.\n\n    Returns the product of two Legendre series `c1` * `c2`.  The arguments\n    are sequences of coefficients, from lowest order "term" to highest,\n    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Legendre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Legendre series coefficients representing their product.\n\n    See Also\n    --------\n    legadd, legsub, legdiv, legpow\n\n    Notes\n    -----\n    In general, the (polynomial) product of two C-series results in terms\n    that are not in the Legendre polynomial basis set.  Thus, to express\n    the product as a Legendre series, it is necessary to "reproject" the\n    product onto said basis set, which may produce "unintuitive" (but\n    correct) results; see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import legendre as L\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2)\n    >>> P.legmul(c1,c2) # multiplication requires "reprojection"\n    array([  4.33333333,  10.4       ,  11.66666667,   3.6       ])\n\n    ')
    
    # Assigning a Call to a List (line 522):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 522)
    # Processing the call arguments (line 522)
    
    # Obtaining an instance of the builtin type 'list' (line 522)
    list_174208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 522)
    # Adding element type (line 522)
    # Getting the type of 'c1' (line 522)
    c1_174209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 28), list_174208, c1_174209)
    # Adding element type (line 522)
    # Getting the type of 'c2' (line 522)
    c2_174210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 28), list_174208, c2_174210)
    
    # Processing the call keyword arguments (line 522)
    kwargs_174211 = {}
    # Getting the type of 'pu' (line 522)
    pu_174206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 522)
    as_series_174207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), pu_174206, 'as_series')
    # Calling as_series(args, kwargs) (line 522)
    as_series_call_result_174212 = invoke(stypy.reporting.localization.Localization(__file__, 522, 15), as_series_174207, *[list_174208], **kwargs_174211)
    
    # Assigning a type to the variable 'call_assignment_173620' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173620', as_series_call_result_174212)
    
    # Assigning a Call to a Name (line 522):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174216 = {}
    # Getting the type of 'call_assignment_173620' (line 522)
    call_assignment_173620_174213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173620', False)
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___174214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 4), call_assignment_173620_174213, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174217 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174214, *[int_174215], **kwargs_174216)
    
    # Assigning a type to the variable 'call_assignment_173621' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173621', getitem___call_result_174217)
    
    # Assigning a Name to a Name (line 522):
    # Getting the type of 'call_assignment_173621' (line 522)
    call_assignment_173621_174218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173621')
    # Assigning a type to the variable 'c1' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 5), 'c1', call_assignment_173621_174218)
    
    # Assigning a Call to a Name (line 522):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174222 = {}
    # Getting the type of 'call_assignment_173620' (line 522)
    call_assignment_173620_174219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173620', False)
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___174220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 4), call_assignment_173620_174219, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174223 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174220, *[int_174221], **kwargs_174222)
    
    # Assigning a type to the variable 'call_assignment_173622' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173622', getitem___call_result_174223)
    
    # Assigning a Name to a Name (line 522):
    # Getting the type of 'call_assignment_173622' (line 522)
    call_assignment_173622_174224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'call_assignment_173622')
    # Assigning a type to the variable 'c2' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 9), 'c2', call_assignment_173622_174224)
    
    
    
    # Call to len(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'c1' (line 524)
    c1_174226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 11), 'c1', False)
    # Processing the call keyword arguments (line 524)
    kwargs_174227 = {}
    # Getting the type of 'len' (line 524)
    len_174225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 7), 'len', False)
    # Calling len(args, kwargs) (line 524)
    len_call_result_174228 = invoke(stypy.reporting.localization.Localization(__file__, 524, 7), len_174225, *[c1_174226], **kwargs_174227)
    
    
    # Call to len(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'c2' (line 524)
    c2_174230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 21), 'c2', False)
    # Processing the call keyword arguments (line 524)
    kwargs_174231 = {}
    # Getting the type of 'len' (line 524)
    len_174229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 17), 'len', False)
    # Calling len(args, kwargs) (line 524)
    len_call_result_174232 = invoke(stypy.reporting.localization.Localization(__file__, 524, 17), len_174229, *[c2_174230], **kwargs_174231)
    
    # Applying the binary operator '>' (line 524)
    result_gt_174233 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 7), '>', len_call_result_174228, len_call_result_174232)
    
    # Testing the type of an if condition (line 524)
    if_condition_174234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 4), result_gt_174233)
    # Assigning a type to the variable 'if_condition_174234' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'if_condition_174234', if_condition_174234)
    # SSA begins for if statement (line 524)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 525):
    
    # Assigning a Name to a Name (line 525):
    # Getting the type of 'c2' (line 525)
    c2_174235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'c2')
    # Assigning a type to the variable 'c' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'c', c2_174235)
    
    # Assigning a Name to a Name (line 526):
    
    # Assigning a Name to a Name (line 526):
    # Getting the type of 'c1' (line 526)
    c1_174236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 13), 'c1')
    # Assigning a type to the variable 'xs' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'xs', c1_174236)
    # SSA branch for the else part of an if statement (line 524)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 528):
    
    # Assigning a Name to a Name (line 528):
    # Getting the type of 'c1' (line 528)
    c1_174237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'c1')
    # Assigning a type to the variable 'c' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'c', c1_174237)
    
    # Assigning a Name to a Name (line 529):
    
    # Assigning a Name to a Name (line 529):
    # Getting the type of 'c2' (line 529)
    c2_174238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 13), 'c2')
    # Assigning a type to the variable 'xs' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'xs', c2_174238)
    # SSA join for if statement (line 524)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 531)
    # Processing the call arguments (line 531)
    # Getting the type of 'c' (line 531)
    c_174240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 11), 'c', False)
    # Processing the call keyword arguments (line 531)
    kwargs_174241 = {}
    # Getting the type of 'len' (line 531)
    len_174239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 7), 'len', False)
    # Calling len(args, kwargs) (line 531)
    len_call_result_174242 = invoke(stypy.reporting.localization.Localization(__file__, 531, 7), len_174239, *[c_174240], **kwargs_174241)
    
    int_174243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 17), 'int')
    # Applying the binary operator '==' (line 531)
    result_eq_174244 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 7), '==', len_call_result_174242, int_174243)
    
    # Testing the type of an if condition (line 531)
    if_condition_174245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 4), result_eq_174244)
    # Assigning a type to the variable 'if_condition_174245' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'if_condition_174245', if_condition_174245)
    # SSA begins for if statement (line 531)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 532):
    
    # Assigning a BinOp to a Name (line 532):
    
    # Obtaining the type of the subscript
    int_174246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 15), 'int')
    # Getting the type of 'c' (line 532)
    c_174247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 532)
    getitem___174248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 13), c_174247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 532)
    subscript_call_result_174249 = invoke(stypy.reporting.localization.Localization(__file__, 532, 13), getitem___174248, int_174246)
    
    # Getting the type of 'xs' (line 532)
    xs_174250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 18), 'xs')
    # Applying the binary operator '*' (line 532)
    result_mul_174251 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 13), '*', subscript_call_result_174249, xs_174250)
    
    # Assigning a type to the variable 'c0' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'c0', result_mul_174251)
    
    # Assigning a Num to a Name (line 533):
    
    # Assigning a Num to a Name (line 533):
    int_174252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 13), 'int')
    # Assigning a type to the variable 'c1' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'c1', int_174252)
    # SSA branch for the else part of an if statement (line 531)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'c' (line 534)
    c_174254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'c', False)
    # Processing the call keyword arguments (line 534)
    kwargs_174255 = {}
    # Getting the type of 'len' (line 534)
    len_174253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 9), 'len', False)
    # Calling len(args, kwargs) (line 534)
    len_call_result_174256 = invoke(stypy.reporting.localization.Localization(__file__, 534, 9), len_174253, *[c_174254], **kwargs_174255)
    
    int_174257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 19), 'int')
    # Applying the binary operator '==' (line 534)
    result_eq_174258 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 9), '==', len_call_result_174256, int_174257)
    
    # Testing the type of an if condition (line 534)
    if_condition_174259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 9), result_eq_174258)
    # Assigning a type to the variable 'if_condition_174259' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 9), 'if_condition_174259', if_condition_174259)
    # SSA begins for if statement (line 534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 535):
    
    # Assigning a BinOp to a Name (line 535):
    
    # Obtaining the type of the subscript
    int_174260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 15), 'int')
    # Getting the type of 'c' (line 535)
    c_174261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 535)
    getitem___174262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 13), c_174261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 535)
    subscript_call_result_174263 = invoke(stypy.reporting.localization.Localization(__file__, 535, 13), getitem___174262, int_174260)
    
    # Getting the type of 'xs' (line 535)
    xs_174264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 18), 'xs')
    # Applying the binary operator '*' (line 535)
    result_mul_174265 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 13), '*', subscript_call_result_174263, xs_174264)
    
    # Assigning a type to the variable 'c0' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'c0', result_mul_174265)
    
    # Assigning a BinOp to a Name (line 536):
    
    # Assigning a BinOp to a Name (line 536):
    
    # Obtaining the type of the subscript
    int_174266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 15), 'int')
    # Getting the type of 'c' (line 536)
    c_174267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___174268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 13), c_174267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 536)
    subscript_call_result_174269 = invoke(stypy.reporting.localization.Localization(__file__, 536, 13), getitem___174268, int_174266)
    
    # Getting the type of 'xs' (line 536)
    xs_174270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 18), 'xs')
    # Applying the binary operator '*' (line 536)
    result_mul_174271 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 13), '*', subscript_call_result_174269, xs_174270)
    
    # Assigning a type to the variable 'c1' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'c1', result_mul_174271)
    # SSA branch for the else part of an if statement (line 534)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 538):
    
    # Assigning a Call to a Name (line 538):
    
    # Call to len(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'c' (line 538)
    c_174273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 17), 'c', False)
    # Processing the call keyword arguments (line 538)
    kwargs_174274 = {}
    # Getting the type of 'len' (line 538)
    len_174272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 13), 'len', False)
    # Calling len(args, kwargs) (line 538)
    len_call_result_174275 = invoke(stypy.reporting.localization.Localization(__file__, 538, 13), len_174272, *[c_174273], **kwargs_174274)
    
    # Assigning a type to the variable 'nd' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'nd', len_call_result_174275)
    
    # Assigning a BinOp to a Name (line 539):
    
    # Assigning a BinOp to a Name (line 539):
    
    # Obtaining the type of the subscript
    int_174276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 15), 'int')
    # Getting the type of 'c' (line 539)
    c_174277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___174278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 13), c_174277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 539)
    subscript_call_result_174279 = invoke(stypy.reporting.localization.Localization(__file__, 539, 13), getitem___174278, int_174276)
    
    # Getting the type of 'xs' (line 539)
    xs_174280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), 'xs')
    # Applying the binary operator '*' (line 539)
    result_mul_174281 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 13), '*', subscript_call_result_174279, xs_174280)
    
    # Assigning a type to the variable 'c0' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'c0', result_mul_174281)
    
    # Assigning a BinOp to a Name (line 540):
    
    # Assigning a BinOp to a Name (line 540):
    
    # Obtaining the type of the subscript
    int_174282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 15), 'int')
    # Getting the type of 'c' (line 540)
    c_174283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___174284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 13), c_174283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_174285 = invoke(stypy.reporting.localization.Localization(__file__, 540, 13), getitem___174284, int_174282)
    
    # Getting the type of 'xs' (line 540)
    xs_174286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 19), 'xs')
    # Applying the binary operator '*' (line 540)
    result_mul_174287 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 13), '*', subscript_call_result_174285, xs_174286)
    
    # Assigning a type to the variable 'c1' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'c1', result_mul_174287)
    
    
    # Call to range(...): (line 541)
    # Processing the call arguments (line 541)
    int_174289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 23), 'int')
    
    # Call to len(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'c' (line 541)
    c_174291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 30), 'c', False)
    # Processing the call keyword arguments (line 541)
    kwargs_174292 = {}
    # Getting the type of 'len' (line 541)
    len_174290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 26), 'len', False)
    # Calling len(args, kwargs) (line 541)
    len_call_result_174293 = invoke(stypy.reporting.localization.Localization(__file__, 541, 26), len_174290, *[c_174291], **kwargs_174292)
    
    int_174294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 35), 'int')
    # Applying the binary operator '+' (line 541)
    result_add_174295 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 26), '+', len_call_result_174293, int_174294)
    
    # Processing the call keyword arguments (line 541)
    kwargs_174296 = {}
    # Getting the type of 'range' (line 541)
    range_174288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 17), 'range', False)
    # Calling range(args, kwargs) (line 541)
    range_call_result_174297 = invoke(stypy.reporting.localization.Localization(__file__, 541, 17), range_174288, *[int_174289, result_add_174295], **kwargs_174296)
    
    # Testing the type of a for loop iterable (line 541)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 541, 8), range_call_result_174297)
    # Getting the type of the for loop variable (line 541)
    for_loop_var_174298 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 541, 8), range_call_result_174297)
    # Assigning a type to the variable 'i' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'i', for_loop_var_174298)
    # SSA begins for a for statement (line 541)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 542):
    
    # Assigning a Name to a Name (line 542):
    # Getting the type of 'c0' (line 542)
    c0_174299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'tmp', c0_174299)
    
    # Assigning a BinOp to a Name (line 543):
    
    # Assigning a BinOp to a Name (line 543):
    # Getting the type of 'nd' (line 543)
    nd_174300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 17), 'nd')
    int_174301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 22), 'int')
    # Applying the binary operator '-' (line 543)
    result_sub_174302 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 17), '-', nd_174300, int_174301)
    
    # Assigning a type to the variable 'nd' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'nd', result_sub_174302)
    
    # Assigning a Call to a Name (line 544):
    
    # Assigning a Call to a Name (line 544):
    
    # Call to legsub(...): (line 544)
    # Processing the call arguments (line 544)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 544)
    i_174304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 27), 'i', False)
    # Applying the 'usub' unary operator (line 544)
    result___neg___174305 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 26), 'usub', i_174304)
    
    # Getting the type of 'c' (line 544)
    c_174306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___174307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 24), c_174306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_174308 = invoke(stypy.reporting.localization.Localization(__file__, 544, 24), getitem___174307, result___neg___174305)
    
    # Getting the type of 'xs' (line 544)
    xs_174309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 30), 'xs', False)
    # Applying the binary operator '*' (line 544)
    result_mul_174310 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 24), '*', subscript_call_result_174308, xs_174309)
    
    # Getting the type of 'c1' (line 544)
    c1_174311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 35), 'c1', False)
    # Getting the type of 'nd' (line 544)
    nd_174312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 39), 'nd', False)
    int_174313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 44), 'int')
    # Applying the binary operator '-' (line 544)
    result_sub_174314 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 39), '-', nd_174312, int_174313)
    
    # Applying the binary operator '*' (line 544)
    result_mul_174315 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 35), '*', c1_174311, result_sub_174314)
    
    # Getting the type of 'nd' (line 544)
    nd_174316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 48), 'nd', False)
    # Applying the binary operator 'div' (line 544)
    result_div_174317 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 34), 'div', result_mul_174315, nd_174316)
    
    # Processing the call keyword arguments (line 544)
    kwargs_174318 = {}
    # Getting the type of 'legsub' (line 544)
    legsub_174303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 17), 'legsub', False)
    # Calling legsub(args, kwargs) (line 544)
    legsub_call_result_174319 = invoke(stypy.reporting.localization.Localization(__file__, 544, 17), legsub_174303, *[result_mul_174310, result_div_174317], **kwargs_174318)
    
    # Assigning a type to the variable 'c0' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'c0', legsub_call_result_174319)
    
    # Assigning a Call to a Name (line 545):
    
    # Assigning a Call to a Name (line 545):
    
    # Call to legadd(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'tmp' (line 545)
    tmp_174321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'tmp', False)
    
    # Call to legmulx(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'c1' (line 545)
    c1_174323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 38), 'c1', False)
    # Processing the call keyword arguments (line 545)
    kwargs_174324 = {}
    # Getting the type of 'legmulx' (line 545)
    legmulx_174322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 30), 'legmulx', False)
    # Calling legmulx(args, kwargs) (line 545)
    legmulx_call_result_174325 = invoke(stypy.reporting.localization.Localization(__file__, 545, 30), legmulx_174322, *[c1_174323], **kwargs_174324)
    
    int_174326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 43), 'int')
    # Getting the type of 'nd' (line 545)
    nd_174327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 45), 'nd', False)
    # Applying the binary operator '*' (line 545)
    result_mul_174328 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 43), '*', int_174326, nd_174327)
    
    int_174329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 50), 'int')
    # Applying the binary operator '-' (line 545)
    result_sub_174330 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 43), '-', result_mul_174328, int_174329)
    
    # Applying the binary operator '*' (line 545)
    result_mul_174331 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 30), '*', legmulx_call_result_174325, result_sub_174330)
    
    # Getting the type of 'nd' (line 545)
    nd_174332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 54), 'nd', False)
    # Applying the binary operator 'div' (line 545)
    result_div_174333 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 29), 'div', result_mul_174331, nd_174332)
    
    # Processing the call keyword arguments (line 545)
    kwargs_174334 = {}
    # Getting the type of 'legadd' (line 545)
    legadd_174320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 17), 'legadd', False)
    # Calling legadd(args, kwargs) (line 545)
    legadd_call_result_174335 = invoke(stypy.reporting.localization.Localization(__file__, 545, 17), legadd_174320, *[tmp_174321, result_div_174333], **kwargs_174334)
    
    # Assigning a type to the variable 'c1' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'c1', legadd_call_result_174335)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 534)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 531)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to legadd(...): (line 546)
    # Processing the call arguments (line 546)
    # Getting the type of 'c0' (line 546)
    c0_174337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 18), 'c0', False)
    
    # Call to legmulx(...): (line 546)
    # Processing the call arguments (line 546)
    # Getting the type of 'c1' (line 546)
    c1_174339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 30), 'c1', False)
    # Processing the call keyword arguments (line 546)
    kwargs_174340 = {}
    # Getting the type of 'legmulx' (line 546)
    legmulx_174338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 22), 'legmulx', False)
    # Calling legmulx(args, kwargs) (line 546)
    legmulx_call_result_174341 = invoke(stypy.reporting.localization.Localization(__file__, 546, 22), legmulx_174338, *[c1_174339], **kwargs_174340)
    
    # Processing the call keyword arguments (line 546)
    kwargs_174342 = {}
    # Getting the type of 'legadd' (line 546)
    legadd_174336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 11), 'legadd', False)
    # Calling legadd(args, kwargs) (line 546)
    legadd_call_result_174343 = invoke(stypy.reporting.localization.Localization(__file__, 546, 11), legadd_174336, *[c0_174337, legmulx_call_result_174341], **kwargs_174342)
    
    # Assigning a type to the variable 'stypy_return_type' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'stypy_return_type', legadd_call_result_174343)
    
    # ################# End of 'legmul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legmul' in the type store
    # Getting the type of 'stypy_return_type' (line 481)
    stypy_return_type_174344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174344)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legmul'
    return stypy_return_type_174344

# Assigning a type to the variable 'legmul' (line 481)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'legmul', legmul)

@norecursion
def legdiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legdiv'
    module_type_store = module_type_store.open_function_context('legdiv', 549, 0, False)
    
    # Passed parameters checking function
    legdiv.stypy_localization = localization
    legdiv.stypy_type_of_self = None
    legdiv.stypy_type_store = module_type_store
    legdiv.stypy_function_name = 'legdiv'
    legdiv.stypy_param_names_list = ['c1', 'c2']
    legdiv.stypy_varargs_param_name = None
    legdiv.stypy_kwargs_param_name = None
    legdiv.stypy_call_defaults = defaults
    legdiv.stypy_call_varargs = varargs
    legdiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legdiv', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legdiv', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legdiv(...)' code ##################

    str_174345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, (-1)), 'str', '\n    Divide one Legendre series by another.\n\n    Returns the quotient-with-remainder of two Legendre series\n    `c1` / `c2`.  The arguments are sequences of coefficients from lowest\n    order "term" to highest, e.g., [1,2,3] represents the series\n    ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Legendre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    quo, rem : ndarrays\n        Of Legendre series coefficients representing the quotient and\n        remainder.\n\n    See Also\n    --------\n    legadd, legsub, legmul, legpow\n\n    Notes\n    -----\n    In general, the (polynomial) division of one Legendre series by another\n    results in quotient and remainder terms that are not in the Legendre\n    polynomial basis set.  Thus, to express these results as a Legendre\n    series, it is necessary to "reproject" the results onto the Legendre\n    basis set, which may produce "unintuitive" (but correct) results; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import legendre as L\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> L.legdiv(c1,c2) # quotient "intuitive," remainder not\n    (array([ 3.]), array([-8., -4.]))\n    >>> c2 = (0,1,2,3)\n    >>> L.legdiv(c2,c1) # neither "intuitive"\n    (array([-0.07407407,  1.66666667]), array([-1.03703704, -2.51851852]))\n\n    ')
    
    # Assigning a Call to a List (line 596):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 596)
    # Processing the call arguments (line 596)
    
    # Obtaining an instance of the builtin type 'list' (line 596)
    list_174348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 596)
    # Adding element type (line 596)
    # Getting the type of 'c1' (line 596)
    c1_174349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 28), list_174348, c1_174349)
    # Adding element type (line 596)
    # Getting the type of 'c2' (line 596)
    c2_174350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 28), list_174348, c2_174350)
    
    # Processing the call keyword arguments (line 596)
    kwargs_174351 = {}
    # Getting the type of 'pu' (line 596)
    pu_174346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 596)
    as_series_174347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 15), pu_174346, 'as_series')
    # Calling as_series(args, kwargs) (line 596)
    as_series_call_result_174352 = invoke(stypy.reporting.localization.Localization(__file__, 596, 15), as_series_174347, *[list_174348], **kwargs_174351)
    
    # Assigning a type to the variable 'call_assignment_173623' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173623', as_series_call_result_174352)
    
    # Assigning a Call to a Name (line 596):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174356 = {}
    # Getting the type of 'call_assignment_173623' (line 596)
    call_assignment_173623_174353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173623', False)
    # Obtaining the member '__getitem__' of a type (line 596)
    getitem___174354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 4), call_assignment_173623_174353, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174357 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174354, *[int_174355], **kwargs_174356)
    
    # Assigning a type to the variable 'call_assignment_173624' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173624', getitem___call_result_174357)
    
    # Assigning a Name to a Name (line 596):
    # Getting the type of 'call_assignment_173624' (line 596)
    call_assignment_173624_174358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173624')
    # Assigning a type to the variable 'c1' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 5), 'c1', call_assignment_173624_174358)
    
    # Assigning a Call to a Name (line 596):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174362 = {}
    # Getting the type of 'call_assignment_173623' (line 596)
    call_assignment_173623_174359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173623', False)
    # Obtaining the member '__getitem__' of a type (line 596)
    getitem___174360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 4), call_assignment_173623_174359, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174363 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174360, *[int_174361], **kwargs_174362)
    
    # Assigning a type to the variable 'call_assignment_173625' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173625', getitem___call_result_174363)
    
    # Assigning a Name to a Name (line 596):
    # Getting the type of 'call_assignment_173625' (line 596)
    call_assignment_173625_174364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'call_assignment_173625')
    # Assigning a type to the variable 'c2' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 9), 'c2', call_assignment_173625_174364)
    
    
    
    # Obtaining the type of the subscript
    int_174365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 10), 'int')
    # Getting the type of 'c2' (line 597)
    c2_174366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 7), 'c2')
    # Obtaining the member '__getitem__' of a type (line 597)
    getitem___174367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 7), c2_174366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 597)
    subscript_call_result_174368 = invoke(stypy.reporting.localization.Localization(__file__, 597, 7), getitem___174367, int_174365)
    
    int_174369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 17), 'int')
    # Applying the binary operator '==' (line 597)
    result_eq_174370 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 7), '==', subscript_call_result_174368, int_174369)
    
    # Testing the type of an if condition (line 597)
    if_condition_174371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 4), result_eq_174370)
    # Assigning a type to the variable 'if_condition_174371' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'if_condition_174371', if_condition_174371)
    # SSA begins for if statement (line 597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ZeroDivisionError(...): (line 598)
    # Processing the call keyword arguments (line 598)
    kwargs_174373 = {}
    # Getting the type of 'ZeroDivisionError' (line 598)
    ZeroDivisionError_174372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 14), 'ZeroDivisionError', False)
    # Calling ZeroDivisionError(args, kwargs) (line 598)
    ZeroDivisionError_call_result_174374 = invoke(stypy.reporting.localization.Localization(__file__, 598, 14), ZeroDivisionError_174372, *[], **kwargs_174373)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 598, 8), ZeroDivisionError_call_result_174374, 'raise parameter', BaseException)
    # SSA join for if statement (line 597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 600):
    
    # Assigning a Call to a Name (line 600):
    
    # Call to len(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'c1' (line 600)
    c1_174376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 14), 'c1', False)
    # Processing the call keyword arguments (line 600)
    kwargs_174377 = {}
    # Getting the type of 'len' (line 600)
    len_174375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 10), 'len', False)
    # Calling len(args, kwargs) (line 600)
    len_call_result_174378 = invoke(stypy.reporting.localization.Localization(__file__, 600, 10), len_174375, *[c1_174376], **kwargs_174377)
    
    # Assigning a type to the variable 'lc1' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'lc1', len_call_result_174378)
    
    # Assigning a Call to a Name (line 601):
    
    # Assigning a Call to a Name (line 601):
    
    # Call to len(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'c2' (line 601)
    c2_174380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 14), 'c2', False)
    # Processing the call keyword arguments (line 601)
    kwargs_174381 = {}
    # Getting the type of 'len' (line 601)
    len_174379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 10), 'len', False)
    # Calling len(args, kwargs) (line 601)
    len_call_result_174382 = invoke(stypy.reporting.localization.Localization(__file__, 601, 10), len_174379, *[c2_174380], **kwargs_174381)
    
    # Assigning a type to the variable 'lc2' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'lc2', len_call_result_174382)
    
    
    # Getting the type of 'lc1' (line 602)
    lc1_174383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 7), 'lc1')
    # Getting the type of 'lc2' (line 602)
    lc2_174384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 13), 'lc2')
    # Applying the binary operator '<' (line 602)
    result_lt_174385 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 7), '<', lc1_174383, lc2_174384)
    
    # Testing the type of an if condition (line 602)
    if_condition_174386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 602, 4), result_lt_174385)
    # Assigning a type to the variable 'if_condition_174386' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'if_condition_174386', if_condition_174386)
    # SSA begins for if statement (line 602)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 603)
    tuple_174387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 603)
    # Adding element type (line 603)
    
    # Obtaining the type of the subscript
    int_174388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 19), 'int')
    slice_174389 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 603, 15), None, int_174388, None)
    # Getting the type of 'c1' (line 603)
    c1_174390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___174391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 15), c1_174390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_174392 = invoke(stypy.reporting.localization.Localization(__file__, 603, 15), getitem___174391, slice_174389)
    
    int_174393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 22), 'int')
    # Applying the binary operator '*' (line 603)
    result_mul_174394 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 15), '*', subscript_call_result_174392, int_174393)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), tuple_174387, result_mul_174394)
    # Adding element type (line 603)
    # Getting the type of 'c1' (line 603)
    c1_174395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 25), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), tuple_174387, c1_174395)
    
    # Assigning a type to the variable 'stypy_return_type' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'stypy_return_type', tuple_174387)
    # SSA branch for the else part of an if statement (line 602)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lc2' (line 604)
    lc2_174396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 9), 'lc2')
    int_174397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 16), 'int')
    # Applying the binary operator '==' (line 604)
    result_eq_174398 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 9), '==', lc2_174396, int_174397)
    
    # Testing the type of an if condition (line 604)
    if_condition_174399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 9), result_eq_174398)
    # Assigning a type to the variable 'if_condition_174399' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 9), 'if_condition_174399', if_condition_174399)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 605)
    tuple_174400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 605)
    # Adding element type (line 605)
    # Getting the type of 'c1' (line 605)
    c1_174401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'c1')
    
    # Obtaining the type of the subscript
    int_174402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 21), 'int')
    # Getting the type of 'c2' (line 605)
    c2_174403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 18), 'c2')
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___174404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 18), c2_174403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_174405 = invoke(stypy.reporting.localization.Localization(__file__, 605, 18), getitem___174404, int_174402)
    
    # Applying the binary operator 'div' (line 605)
    result_div_174406 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 15), 'div', c1_174401, subscript_call_result_174405)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 15), tuple_174400, result_div_174406)
    # Adding element type (line 605)
    
    # Obtaining the type of the subscript
    int_174407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 30), 'int')
    slice_174408 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 605, 26), None, int_174407, None)
    # Getting the type of 'c1' (line 605)
    c1_174409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___174410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 26), c1_174409, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_174411 = invoke(stypy.reporting.localization.Localization(__file__, 605, 26), getitem___174410, slice_174408)
    
    int_174412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 33), 'int')
    # Applying the binary operator '*' (line 605)
    result_mul_174413 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 26), '*', subscript_call_result_174411, int_174412)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 15), tuple_174400, result_mul_174413)
    
    # Assigning a type to the variable 'stypy_return_type' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'stypy_return_type', tuple_174400)
    # SSA branch for the else part of an if statement (line 604)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 607):
    
    # Assigning a Call to a Name (line 607):
    
    # Call to empty(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of 'lc1' (line 607)
    lc1_174416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 607)
    lc2_174417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 29), 'lc2', False)
    # Applying the binary operator '-' (line 607)
    result_sub_174418 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 23), '-', lc1_174416, lc2_174417)
    
    int_174419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 35), 'int')
    # Applying the binary operator '+' (line 607)
    result_add_174420 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 33), '+', result_sub_174418, int_174419)
    
    # Processing the call keyword arguments (line 607)
    # Getting the type of 'c1' (line 607)
    c1_174421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 44), 'c1', False)
    # Obtaining the member 'dtype' of a type (line 607)
    dtype_174422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 44), c1_174421, 'dtype')
    keyword_174423 = dtype_174422
    kwargs_174424 = {'dtype': keyword_174423}
    # Getting the type of 'np' (line 607)
    np_174414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 607)
    empty_174415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 14), np_174414, 'empty')
    # Calling empty(args, kwargs) (line 607)
    empty_call_result_174425 = invoke(stypy.reporting.localization.Localization(__file__, 607, 14), empty_174415, *[result_add_174420], **kwargs_174424)
    
    # Assigning a type to the variable 'quo' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'quo', empty_call_result_174425)
    
    # Assigning a Name to a Name (line 608):
    
    # Assigning a Name to a Name (line 608):
    # Getting the type of 'c1' (line 608)
    c1_174426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 14), 'c1')
    # Assigning a type to the variable 'rem' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'rem', c1_174426)
    
    
    # Call to range(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'lc1' (line 609)
    lc1_174428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 609)
    lc2_174429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 29), 'lc2', False)
    # Applying the binary operator '-' (line 609)
    result_sub_174430 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 23), '-', lc1_174428, lc2_174429)
    
    int_174431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 34), 'int')
    int_174432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 39), 'int')
    # Processing the call keyword arguments (line 609)
    kwargs_174433 = {}
    # Getting the type of 'range' (line 609)
    range_174427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 17), 'range', False)
    # Calling range(args, kwargs) (line 609)
    range_call_result_174434 = invoke(stypy.reporting.localization.Localization(__file__, 609, 17), range_174427, *[result_sub_174430, int_174431, int_174432], **kwargs_174433)
    
    # Testing the type of a for loop iterable (line 609)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 609, 8), range_call_result_174434)
    # Getting the type of the for loop variable (line 609)
    for_loop_var_174435 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 609, 8), range_call_result_174434)
    # Assigning a type to the variable 'i' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'i', for_loop_var_174435)
    # SSA begins for a for statement (line 609)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 610):
    
    # Assigning a Call to a Name (line 610):
    
    # Call to legmul(...): (line 610)
    # Processing the call arguments (line 610)
    
    # Obtaining an instance of the builtin type 'list' (line 610)
    list_174437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 610)
    # Adding element type (line 610)
    int_174438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 23), list_174437, int_174438)
    
    # Getting the type of 'i' (line 610)
    i_174439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 27), 'i', False)
    # Applying the binary operator '*' (line 610)
    result_mul_174440 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 23), '*', list_174437, i_174439)
    
    
    # Obtaining an instance of the builtin type 'list' (line 610)
    list_174441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 610)
    # Adding element type (line 610)
    int_174442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 31), list_174441, int_174442)
    
    # Applying the binary operator '+' (line 610)
    result_add_174443 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 23), '+', result_mul_174440, list_174441)
    
    # Getting the type of 'c2' (line 610)
    c2_174444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 36), 'c2', False)
    # Processing the call keyword arguments (line 610)
    kwargs_174445 = {}
    # Getting the type of 'legmul' (line 610)
    legmul_174436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 16), 'legmul', False)
    # Calling legmul(args, kwargs) (line 610)
    legmul_call_result_174446 = invoke(stypy.reporting.localization.Localization(__file__, 610, 16), legmul_174436, *[result_add_174443, c2_174444], **kwargs_174445)
    
    # Assigning a type to the variable 'p' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'p', legmul_call_result_174446)
    
    # Assigning a BinOp to a Name (line 611):
    
    # Assigning a BinOp to a Name (line 611):
    
    # Obtaining the type of the subscript
    int_174447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 20), 'int')
    # Getting the type of 'rem' (line 611)
    rem_174448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'rem')
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___174449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), rem_174448, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_174450 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), getitem___174449, int_174447)
    
    
    # Obtaining the type of the subscript
    int_174451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 26), 'int')
    # Getting the type of 'p' (line 611)
    p_174452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 24), 'p')
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___174453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 24), p_174452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_174454 = invoke(stypy.reporting.localization.Localization(__file__, 611, 24), getitem___174453, int_174451)
    
    # Applying the binary operator 'div' (line 611)
    result_div_174455 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 16), 'div', subscript_call_result_174450, subscript_call_result_174454)
    
    # Assigning a type to the variable 'q' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'q', result_div_174455)
    
    # Assigning a BinOp to a Name (line 612):
    
    # Assigning a BinOp to a Name (line 612):
    
    # Obtaining the type of the subscript
    int_174456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 23), 'int')
    slice_174457 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 612, 18), None, int_174456, None)
    # Getting the type of 'rem' (line 612)
    rem_174458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 18), 'rem')
    # Obtaining the member '__getitem__' of a type (line 612)
    getitem___174459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 18), rem_174458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 612)
    subscript_call_result_174460 = invoke(stypy.reporting.localization.Localization(__file__, 612, 18), getitem___174459, slice_174457)
    
    # Getting the type of 'q' (line 612)
    q_174461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 29), 'q')
    
    # Obtaining the type of the subscript
    int_174462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 34), 'int')
    slice_174463 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 612, 31), None, int_174462, None)
    # Getting the type of 'p' (line 612)
    p_174464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 31), 'p')
    # Obtaining the member '__getitem__' of a type (line 612)
    getitem___174465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 31), p_174464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 612)
    subscript_call_result_174466 = invoke(stypy.reporting.localization.Localization(__file__, 612, 31), getitem___174465, slice_174463)
    
    # Applying the binary operator '*' (line 612)
    result_mul_174467 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 29), '*', q_174461, subscript_call_result_174466)
    
    # Applying the binary operator '-' (line 612)
    result_sub_174468 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 18), '-', subscript_call_result_174460, result_mul_174467)
    
    # Assigning a type to the variable 'rem' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'rem', result_sub_174468)
    
    # Assigning a Name to a Subscript (line 613):
    
    # Assigning a Name to a Subscript (line 613):
    # Getting the type of 'q' (line 613)
    q_174469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 21), 'q')
    # Getting the type of 'quo' (line 613)
    quo_174470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'quo')
    # Getting the type of 'i' (line 613)
    i_174471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 16), 'i')
    # Storing an element on a container (line 613)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 12), quo_174470, (i_174471, q_174469))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 614)
    tuple_174472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 614)
    # Adding element type (line 614)
    # Getting the type of 'quo' (line 614)
    quo_174473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 15), 'quo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 15), tuple_174472, quo_174473)
    # Adding element type (line 614)
    
    # Call to trimseq(...): (line 614)
    # Processing the call arguments (line 614)
    # Getting the type of 'rem' (line 614)
    rem_174476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 31), 'rem', False)
    # Processing the call keyword arguments (line 614)
    kwargs_174477 = {}
    # Getting the type of 'pu' (line 614)
    pu_174474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 614)
    trimseq_174475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 20), pu_174474, 'trimseq')
    # Calling trimseq(args, kwargs) (line 614)
    trimseq_call_result_174478 = invoke(stypy.reporting.localization.Localization(__file__, 614, 20), trimseq_174475, *[rem_174476], **kwargs_174477)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 15), tuple_174472, trimseq_call_result_174478)
    
    # Assigning a type to the variable 'stypy_return_type' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'stypy_return_type', tuple_174472)
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 602)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'legdiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legdiv' in the type store
    # Getting the type of 'stypy_return_type' (line 549)
    stypy_return_type_174479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174479)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legdiv'
    return stypy_return_type_174479

# Assigning a type to the variable 'legdiv' (line 549)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'legdiv', legdiv)

@norecursion
def legpow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_174480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 28), 'int')
    defaults = [int_174480]
    # Create a new context for function 'legpow'
    module_type_store = module_type_store.open_function_context('legpow', 617, 0, False)
    
    # Passed parameters checking function
    legpow.stypy_localization = localization
    legpow.stypy_type_of_self = None
    legpow.stypy_type_store = module_type_store
    legpow.stypy_function_name = 'legpow'
    legpow.stypy_param_names_list = ['c', 'pow', 'maxpower']
    legpow.stypy_varargs_param_name = None
    legpow.stypy_kwargs_param_name = None
    legpow.stypy_call_defaults = defaults
    legpow.stypy_call_varargs = varargs
    legpow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legpow', ['c', 'pow', 'maxpower'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legpow', localization, ['c', 'pow', 'maxpower'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legpow(...)' code ##################

    str_174481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, (-1)), 'str', 'Raise a Legendre series to a power.\n\n    Returns the Legendre series `c` raised to the power `pow`. The\n    arguement `c` is a sequence of coefficients ordered from low to high.\n    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Legendre series coefficients ordered from low to\n        high.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Legendre series of power.\n\n    See Also\n    --------\n    legadd, legsub, legmul, legdiv\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a Call to a List (line 649):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 649)
    # Processing the call arguments (line 649)
    
    # Obtaining an instance of the builtin type 'list' (line 649)
    list_174484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 649)
    # Adding element type (line 649)
    # Getting the type of 'c' (line 649)
    c_174485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 23), list_174484, c_174485)
    
    # Processing the call keyword arguments (line 649)
    kwargs_174486 = {}
    # Getting the type of 'pu' (line 649)
    pu_174482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 649)
    as_series_174483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 10), pu_174482, 'as_series')
    # Calling as_series(args, kwargs) (line 649)
    as_series_call_result_174487 = invoke(stypy.reporting.localization.Localization(__file__, 649, 10), as_series_174483, *[list_174484], **kwargs_174486)
    
    # Assigning a type to the variable 'call_assignment_173626' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'call_assignment_173626', as_series_call_result_174487)
    
    # Assigning a Call to a Name (line 649):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_174490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    # Processing the call keyword arguments
    kwargs_174491 = {}
    # Getting the type of 'call_assignment_173626' (line 649)
    call_assignment_173626_174488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'call_assignment_173626', False)
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___174489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), call_assignment_173626_174488, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_174492 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___174489, *[int_174490], **kwargs_174491)
    
    # Assigning a type to the variable 'call_assignment_173627' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'call_assignment_173627', getitem___call_result_174492)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'call_assignment_173627' (line 649)
    call_assignment_173627_174493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'call_assignment_173627')
    # Assigning a type to the variable 'c' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 5), 'c', call_assignment_173627_174493)
    
    # Assigning a Call to a Name (line 650):
    
    # Assigning a Call to a Name (line 650):
    
    # Call to int(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'pow' (line 650)
    pow_174495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'pow', False)
    # Processing the call keyword arguments (line 650)
    kwargs_174496 = {}
    # Getting the type of 'int' (line 650)
    int_174494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'int', False)
    # Calling int(args, kwargs) (line 650)
    int_call_result_174497 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), int_174494, *[pow_174495], **kwargs_174496)
    
    # Assigning a type to the variable 'power' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'power', int_call_result_174497)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'power' (line 651)
    power_174498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 7), 'power')
    # Getting the type of 'pow' (line 651)
    pow_174499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'pow')
    # Applying the binary operator '!=' (line 651)
    result_ne_174500 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 7), '!=', power_174498, pow_174499)
    
    
    # Getting the type of 'power' (line 651)
    power_174501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 23), 'power')
    int_174502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 31), 'int')
    # Applying the binary operator '<' (line 651)
    result_lt_174503 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 23), '<', power_174501, int_174502)
    
    # Applying the binary operator 'or' (line 651)
    result_or_keyword_174504 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 7), 'or', result_ne_174500, result_lt_174503)
    
    # Testing the type of an if condition (line 651)
    if_condition_174505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 4), result_or_keyword_174504)
    # Assigning a type to the variable 'if_condition_174505' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'if_condition_174505', if_condition_174505)
    # SSA begins for if statement (line 651)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 652)
    # Processing the call arguments (line 652)
    str_174507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 25), 'str', 'Power must be a non-negative integer.')
    # Processing the call keyword arguments (line 652)
    kwargs_174508 = {}
    # Getting the type of 'ValueError' (line 652)
    ValueError_174506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 652)
    ValueError_call_result_174509 = invoke(stypy.reporting.localization.Localization(__file__, 652, 14), ValueError_174506, *[str_174507], **kwargs_174508)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 652, 8), ValueError_call_result_174509, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 651)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxpower' (line 653)
    maxpower_174510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 9), 'maxpower')
    # Getting the type of 'None' (line 653)
    None_174511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 25), 'None')
    # Applying the binary operator 'isnot' (line 653)
    result_is_not_174512 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 9), 'isnot', maxpower_174510, None_174511)
    
    
    # Getting the type of 'power' (line 653)
    power_174513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 34), 'power')
    # Getting the type of 'maxpower' (line 653)
    maxpower_174514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 42), 'maxpower')
    # Applying the binary operator '>' (line 653)
    result_gt_174515 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 34), '>', power_174513, maxpower_174514)
    
    # Applying the binary operator 'and' (line 653)
    result_and_keyword_174516 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 9), 'and', result_is_not_174512, result_gt_174515)
    
    # Testing the type of an if condition (line 653)
    if_condition_174517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 653, 9), result_and_keyword_174516)
    # Assigning a type to the variable 'if_condition_174517' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 9), 'if_condition_174517', if_condition_174517)
    # SSA begins for if statement (line 653)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 654)
    # Processing the call arguments (line 654)
    str_174519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 25), 'str', 'Power is too large')
    # Processing the call keyword arguments (line 654)
    kwargs_174520 = {}
    # Getting the type of 'ValueError' (line 654)
    ValueError_174518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 654)
    ValueError_call_result_174521 = invoke(stypy.reporting.localization.Localization(__file__, 654, 14), ValueError_174518, *[str_174519], **kwargs_174520)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 654, 8), ValueError_call_result_174521, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 653)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 655)
    power_174522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 9), 'power')
    int_174523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 18), 'int')
    # Applying the binary operator '==' (line 655)
    result_eq_174524 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 9), '==', power_174522, int_174523)
    
    # Testing the type of an if condition (line 655)
    if_condition_174525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 655, 9), result_eq_174524)
    # Assigning a type to the variable 'if_condition_174525' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 9), 'if_condition_174525', if_condition_174525)
    # SSA begins for if statement (line 655)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 656)
    # Processing the call arguments (line 656)
    
    # Obtaining an instance of the builtin type 'list' (line 656)
    list_174528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 656)
    # Adding element type (line 656)
    int_174529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 24), list_174528, int_174529)
    
    # Processing the call keyword arguments (line 656)
    # Getting the type of 'c' (line 656)
    c_174530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 35), 'c', False)
    # Obtaining the member 'dtype' of a type (line 656)
    dtype_174531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 35), c_174530, 'dtype')
    keyword_174532 = dtype_174531
    kwargs_174533 = {'dtype': keyword_174532}
    # Getting the type of 'np' (line 656)
    np_174526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 656)
    array_174527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 15), np_174526, 'array')
    # Calling array(args, kwargs) (line 656)
    array_call_result_174534 = invoke(stypy.reporting.localization.Localization(__file__, 656, 15), array_174527, *[list_174528], **kwargs_174533)
    
    # Assigning a type to the variable 'stypy_return_type' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'stypy_return_type', array_call_result_174534)
    # SSA branch for the else part of an if statement (line 655)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 657)
    power_174535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 9), 'power')
    int_174536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 18), 'int')
    # Applying the binary operator '==' (line 657)
    result_eq_174537 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 9), '==', power_174535, int_174536)
    
    # Testing the type of an if condition (line 657)
    if_condition_174538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 9), result_eq_174537)
    # Assigning a type to the variable 'if_condition_174538' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 9), 'if_condition_174538', if_condition_174538)
    # SSA begins for if statement (line 657)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 658)
    c_174539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'stypy_return_type', c_174539)
    # SSA branch for the else part of an if statement (line 657)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 662):
    
    # Assigning a Name to a Name (line 662):
    # Getting the type of 'c' (line 662)
    c_174540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 14), 'c')
    # Assigning a type to the variable 'prd' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'prd', c_174540)
    
    
    # Call to range(...): (line 663)
    # Processing the call arguments (line 663)
    int_174542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 23), 'int')
    # Getting the type of 'power' (line 663)
    power_174543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 26), 'power', False)
    int_174544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 34), 'int')
    # Applying the binary operator '+' (line 663)
    result_add_174545 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 26), '+', power_174543, int_174544)
    
    # Processing the call keyword arguments (line 663)
    kwargs_174546 = {}
    # Getting the type of 'range' (line 663)
    range_174541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 17), 'range', False)
    # Calling range(args, kwargs) (line 663)
    range_call_result_174547 = invoke(stypy.reporting.localization.Localization(__file__, 663, 17), range_174541, *[int_174542, result_add_174545], **kwargs_174546)
    
    # Testing the type of a for loop iterable (line 663)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 663, 8), range_call_result_174547)
    # Getting the type of the for loop variable (line 663)
    for_loop_var_174548 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 663, 8), range_call_result_174547)
    # Assigning a type to the variable 'i' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'i', for_loop_var_174548)
    # SSA begins for a for statement (line 663)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 664):
    
    # Assigning a Call to a Name (line 664):
    
    # Call to legmul(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 'prd' (line 664)
    prd_174550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 25), 'prd', False)
    # Getting the type of 'c' (line 664)
    c_174551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 30), 'c', False)
    # Processing the call keyword arguments (line 664)
    kwargs_174552 = {}
    # Getting the type of 'legmul' (line 664)
    legmul_174549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 18), 'legmul', False)
    # Calling legmul(args, kwargs) (line 664)
    legmul_call_result_174553 = invoke(stypy.reporting.localization.Localization(__file__, 664, 18), legmul_174549, *[prd_174550, c_174551], **kwargs_174552)
    
    # Assigning a type to the variable 'prd' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'prd', legmul_call_result_174553)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 665)
    prd_174554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 15), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), 'stypy_return_type', prd_174554)
    # SSA join for if statement (line 657)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 655)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 653)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 651)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'legpow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legpow' in the type store
    # Getting the type of 'stypy_return_type' (line 617)
    stypy_return_type_174555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174555)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legpow'
    return stypy_return_type_174555

# Assigning a type to the variable 'legpow' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'legpow', legpow)

@norecursion
def legder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_174556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 16), 'int')
    int_174557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 23), 'int')
    int_174558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 31), 'int')
    defaults = [int_174556, int_174557, int_174558]
    # Create a new context for function 'legder'
    module_type_store = module_type_store.open_function_context('legder', 668, 0, False)
    
    # Passed parameters checking function
    legder.stypy_localization = localization
    legder.stypy_type_of_self = None
    legder.stypy_type_store = module_type_store
    legder.stypy_function_name = 'legder'
    legder.stypy_param_names_list = ['c', 'm', 'scl', 'axis']
    legder.stypy_varargs_param_name = None
    legder.stypy_kwargs_param_name = None
    legder.stypy_call_defaults = defaults
    legder.stypy_call_varargs = varargs
    legder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legder', ['c', 'm', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legder', localization, ['c', 'm', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legder(...)' code ##################

    str_174559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, (-1)), 'str', '\n    Differentiate a Legendre series.\n\n    Returns the Legendre series coefficients `c` differentiated `m` times\n    along `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable). The argument\n    `c` is an array of coefficients from low to high degree along each\n    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``\n    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +\n    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is\n    ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Legendre series coefficients. If c is multidimensional the\n        different axis correspond to different variables with the degree in\n        each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change of\n        variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Legendre series of the derivative.\n\n    See Also\n    --------\n    legint\n\n    Notes\n    -----\n    In general, the result of differentiating a Legendre series does not\n    resemble the same operation on a power series. Thus the result of this\n    function may be "unintuitive," albeit correct; see Examples section\n    below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import legendre as L\n    >>> c = (1,2,3,4)\n    >>> L.legder(c)\n    array([  6.,   9.,  20.])\n    >>> L.legder(c, 3)\n    array([ 60.])\n    >>> L.legder(c, scl=-1)\n    array([ -6.,  -9., -20.])\n    >>> L.legder(c, 2,-1)\n    array([  9.,  60.])\n\n    ')
    
    # Assigning a Call to a Name (line 728):
    
    # Assigning a Call to a Name (line 728):
    
    # Call to array(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'c' (line 728)
    c_174562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 17), 'c', False)
    # Processing the call keyword arguments (line 728)
    int_174563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 26), 'int')
    keyword_174564 = int_174563
    int_174565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 34), 'int')
    keyword_174566 = int_174565
    kwargs_174567 = {'copy': keyword_174566, 'ndmin': keyword_174564}
    # Getting the type of 'np' (line 728)
    np_174560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 728)
    array_174561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 8), np_174560, 'array')
    # Calling array(args, kwargs) (line 728)
    array_call_result_174568 = invoke(stypy.reporting.localization.Localization(__file__, 728, 8), array_174561, *[c_174562], **kwargs_174567)
    
    # Assigning a type to the variable 'c' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'c', array_call_result_174568)
    
    
    # Getting the type of 'c' (line 729)
    c_174569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 729)
    dtype_174570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 7), c_174569, 'dtype')
    # Obtaining the member 'char' of a type (line 729)
    char_174571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 7), dtype_174570, 'char')
    str_174572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 729)
    result_contains_174573 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 7), 'in', char_174571, str_174572)
    
    # Testing the type of an if condition (line 729)
    if_condition_174574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 4), result_contains_174573)
    # Assigning a type to the variable 'if_condition_174574' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'if_condition_174574', if_condition_174574)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 730):
    
    # Assigning a Call to a Name (line 730):
    
    # Call to astype(...): (line 730)
    # Processing the call arguments (line 730)
    # Getting the type of 'np' (line 730)
    np_174577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 730)
    double_174578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 21), np_174577, 'double')
    # Processing the call keyword arguments (line 730)
    kwargs_174579 = {}
    # Getting the type of 'c' (line 730)
    c_174575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 730)
    astype_174576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 12), c_174575, 'astype')
    # Calling astype(args, kwargs) (line 730)
    astype_call_result_174580 = invoke(stypy.reporting.localization.Localization(__file__, 730, 12), astype_174576, *[double_174578], **kwargs_174579)
    
    # Assigning a type to the variable 'c' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'c', astype_call_result_174580)
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 731):
    
    # Assigning a Subscript to a Name (line 731):
    
    # Obtaining the type of the subscript
    int_174581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 731)
    list_174586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 731)
    # Adding element type (line 731)
    # Getting the type of 'm' (line 731)
    m_174587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 34), list_174586, m_174587)
    # Adding element type (line 731)
    # Getting the type of 'axis' (line 731)
    axis_174588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 34), list_174586, axis_174588)
    
    comprehension_174589 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 18), list_174586)
    # Assigning a type to the variable 't' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 18), 't', comprehension_174589)
    
    # Call to int(...): (line 731)
    # Processing the call arguments (line 731)
    # Getting the type of 't' (line 731)
    t_174583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 22), 't', False)
    # Processing the call keyword arguments (line 731)
    kwargs_174584 = {}
    # Getting the type of 'int' (line 731)
    int_174582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 18), 'int', False)
    # Calling int(args, kwargs) (line 731)
    int_call_result_174585 = invoke(stypy.reporting.localization.Localization(__file__, 731, 18), int_174582, *[t_174583], **kwargs_174584)
    
    list_174590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 18), list_174590, int_call_result_174585)
    # Obtaining the member '__getitem__' of a type (line 731)
    getitem___174591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 4), list_174590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 731)
    subscript_call_result_174592 = invoke(stypy.reporting.localization.Localization(__file__, 731, 4), getitem___174591, int_174581)
    
    # Assigning a type to the variable 'tuple_var_assignment_173628' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'tuple_var_assignment_173628', subscript_call_result_174592)
    
    # Assigning a Subscript to a Name (line 731):
    
    # Obtaining the type of the subscript
    int_174593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 731)
    list_174598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 731)
    # Adding element type (line 731)
    # Getting the type of 'm' (line 731)
    m_174599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 34), list_174598, m_174599)
    # Adding element type (line 731)
    # Getting the type of 'axis' (line 731)
    axis_174600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 34), list_174598, axis_174600)
    
    comprehension_174601 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 18), list_174598)
    # Assigning a type to the variable 't' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 18), 't', comprehension_174601)
    
    # Call to int(...): (line 731)
    # Processing the call arguments (line 731)
    # Getting the type of 't' (line 731)
    t_174595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 22), 't', False)
    # Processing the call keyword arguments (line 731)
    kwargs_174596 = {}
    # Getting the type of 'int' (line 731)
    int_174594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 18), 'int', False)
    # Calling int(args, kwargs) (line 731)
    int_call_result_174597 = invoke(stypy.reporting.localization.Localization(__file__, 731, 18), int_174594, *[t_174595], **kwargs_174596)
    
    list_174602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 18), list_174602, int_call_result_174597)
    # Obtaining the member '__getitem__' of a type (line 731)
    getitem___174603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 4), list_174602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 731)
    subscript_call_result_174604 = invoke(stypy.reporting.localization.Localization(__file__, 731, 4), getitem___174603, int_174593)
    
    # Assigning a type to the variable 'tuple_var_assignment_173629' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'tuple_var_assignment_173629', subscript_call_result_174604)
    
    # Assigning a Name to a Name (line 731):
    # Getting the type of 'tuple_var_assignment_173628' (line 731)
    tuple_var_assignment_173628_174605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'tuple_var_assignment_173628')
    # Assigning a type to the variable 'cnt' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'cnt', tuple_var_assignment_173628_174605)
    
    # Assigning a Name to a Name (line 731):
    # Getting the type of 'tuple_var_assignment_173629' (line 731)
    tuple_var_assignment_173629_174606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'tuple_var_assignment_173629')
    # Assigning a type to the variable 'iaxis' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 9), 'iaxis', tuple_var_assignment_173629_174606)
    
    
    # Getting the type of 'cnt' (line 733)
    cnt_174607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 7), 'cnt')
    # Getting the type of 'm' (line 733)
    m_174608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 14), 'm')
    # Applying the binary operator '!=' (line 733)
    result_ne_174609 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 7), '!=', cnt_174607, m_174608)
    
    # Testing the type of an if condition (line 733)
    if_condition_174610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 733, 4), result_ne_174609)
    # Assigning a type to the variable 'if_condition_174610' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'if_condition_174610', if_condition_174610)
    # SSA begins for if statement (line 733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 734)
    # Processing the call arguments (line 734)
    str_174612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 25), 'str', 'The order of derivation must be integer')
    # Processing the call keyword arguments (line 734)
    kwargs_174613 = {}
    # Getting the type of 'ValueError' (line 734)
    ValueError_174611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 734)
    ValueError_call_result_174614 = invoke(stypy.reporting.localization.Localization(__file__, 734, 14), ValueError_174611, *[str_174612], **kwargs_174613)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 734, 8), ValueError_call_result_174614, 'raise parameter', BaseException)
    # SSA join for if statement (line 733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 735)
    cnt_174615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 7), 'cnt')
    int_174616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 13), 'int')
    # Applying the binary operator '<' (line 735)
    result_lt_174617 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 7), '<', cnt_174615, int_174616)
    
    # Testing the type of an if condition (line 735)
    if_condition_174618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 735, 4), result_lt_174617)
    # Assigning a type to the variable 'if_condition_174618' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'if_condition_174618', if_condition_174618)
    # SSA begins for if statement (line 735)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 736)
    # Processing the call arguments (line 736)
    str_174620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 25), 'str', 'The order of derivation must be non-negative')
    # Processing the call keyword arguments (line 736)
    kwargs_174621 = {}
    # Getting the type of 'ValueError' (line 736)
    ValueError_174619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 736)
    ValueError_call_result_174622 = invoke(stypy.reporting.localization.Localization(__file__, 736, 14), ValueError_174619, *[str_174620], **kwargs_174621)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 736, 8), ValueError_call_result_174622, 'raise parameter', BaseException)
    # SSA join for if statement (line 735)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 737)
    iaxis_174623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 7), 'iaxis')
    # Getting the type of 'axis' (line 737)
    axis_174624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 16), 'axis')
    # Applying the binary operator '!=' (line 737)
    result_ne_174625 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 7), '!=', iaxis_174623, axis_174624)
    
    # Testing the type of an if condition (line 737)
    if_condition_174626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 737, 4), result_ne_174625)
    # Assigning a type to the variable 'if_condition_174626' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'if_condition_174626', if_condition_174626)
    # SSA begins for if statement (line 737)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 738)
    # Processing the call arguments (line 738)
    str_174628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 738)
    kwargs_174629 = {}
    # Getting the type of 'ValueError' (line 738)
    ValueError_174627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 738)
    ValueError_call_result_174630 = invoke(stypy.reporting.localization.Localization(__file__, 738, 14), ValueError_174627, *[str_174628], **kwargs_174629)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 738, 8), ValueError_call_result_174630, 'raise parameter', BaseException)
    # SSA join for if statement (line 737)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 739)
    c_174631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 739)
    ndim_174632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 12), c_174631, 'ndim')
    # Applying the 'usub' unary operator (line 739)
    result___neg___174633 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 11), 'usub', ndim_174632)
    
    # Getting the type of 'iaxis' (line 739)
    iaxis_174634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 22), 'iaxis')
    # Applying the binary operator '<=' (line 739)
    result_le_174635 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 11), '<=', result___neg___174633, iaxis_174634)
    # Getting the type of 'c' (line 739)
    c_174636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 739)
    ndim_174637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 30), c_174636, 'ndim')
    # Applying the binary operator '<' (line 739)
    result_lt_174638 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 11), '<', iaxis_174634, ndim_174637)
    # Applying the binary operator '&' (line 739)
    result_and__174639 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 11), '&', result_le_174635, result_lt_174638)
    
    # Applying the 'not' unary operator (line 739)
    result_not__174640 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 7), 'not', result_and__174639)
    
    # Testing the type of an if condition (line 739)
    if_condition_174641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 4), result_not__174640)
    # Assigning a type to the variable 'if_condition_174641' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'if_condition_174641', if_condition_174641)
    # SSA begins for if statement (line 739)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 740)
    # Processing the call arguments (line 740)
    str_174643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 740)
    kwargs_174644 = {}
    # Getting the type of 'ValueError' (line 740)
    ValueError_174642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 740)
    ValueError_call_result_174645 = invoke(stypy.reporting.localization.Localization(__file__, 740, 14), ValueError_174642, *[str_174643], **kwargs_174644)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 740, 8), ValueError_call_result_174645, 'raise parameter', BaseException)
    # SSA join for if statement (line 739)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 741)
    iaxis_174646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 7), 'iaxis')
    int_174647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 15), 'int')
    # Applying the binary operator '<' (line 741)
    result_lt_174648 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 7), '<', iaxis_174646, int_174647)
    
    # Testing the type of an if condition (line 741)
    if_condition_174649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 741, 4), result_lt_174648)
    # Assigning a type to the variable 'if_condition_174649' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'if_condition_174649', if_condition_174649)
    # SSA begins for if statement (line 741)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 742)
    iaxis_174650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'iaxis')
    # Getting the type of 'c' (line 742)
    c_174651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 742)
    ndim_174652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 17), c_174651, 'ndim')
    # Applying the binary operator '+=' (line 742)
    result_iadd_174653 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 8), '+=', iaxis_174650, ndim_174652)
    # Assigning a type to the variable 'iaxis' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'iaxis', result_iadd_174653)
    
    # SSA join for if statement (line 741)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 744)
    cnt_174654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 7), 'cnt')
    int_174655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 14), 'int')
    # Applying the binary operator '==' (line 744)
    result_eq_174656 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 7), '==', cnt_174654, int_174655)
    
    # Testing the type of an if condition (line 744)
    if_condition_174657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 4), result_eq_174656)
    # Assigning a type to the variable 'if_condition_174657' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'if_condition_174657', if_condition_174657)
    # SSA begins for if statement (line 744)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 745)
    c_174658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'stypy_return_type', c_174658)
    # SSA join for if statement (line 744)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 747):
    
    # Assigning a Call to a Name (line 747):
    
    # Call to rollaxis(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'c' (line 747)
    c_174661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'c', False)
    # Getting the type of 'iaxis' (line 747)
    iaxis_174662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 747)
    kwargs_174663 = {}
    # Getting the type of 'np' (line 747)
    np_174659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 747)
    rollaxis_174660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 8), np_174659, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 747)
    rollaxis_call_result_174664 = invoke(stypy.reporting.localization.Localization(__file__, 747, 8), rollaxis_174660, *[c_174661, iaxis_174662], **kwargs_174663)
    
    # Assigning a type to the variable 'c' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'c', rollaxis_call_result_174664)
    
    # Assigning a Call to a Name (line 748):
    
    # Assigning a Call to a Name (line 748):
    
    # Call to len(...): (line 748)
    # Processing the call arguments (line 748)
    # Getting the type of 'c' (line 748)
    c_174666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'c', False)
    # Processing the call keyword arguments (line 748)
    kwargs_174667 = {}
    # Getting the type of 'len' (line 748)
    len_174665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'len', False)
    # Calling len(args, kwargs) (line 748)
    len_call_result_174668 = invoke(stypy.reporting.localization.Localization(__file__, 748, 8), len_174665, *[c_174666], **kwargs_174667)
    
    # Assigning a type to the variable 'n' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'n', len_call_result_174668)
    
    
    # Getting the type of 'cnt' (line 749)
    cnt_174669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 7), 'cnt')
    # Getting the type of 'n' (line 749)
    n_174670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 14), 'n')
    # Applying the binary operator '>=' (line 749)
    result_ge_174671 = python_operator(stypy.reporting.localization.Localization(__file__, 749, 7), '>=', cnt_174669, n_174670)
    
    # Testing the type of an if condition (line 749)
    if_condition_174672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 749, 4), result_ge_174671)
    # Assigning a type to the variable 'if_condition_174672' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'if_condition_174672', if_condition_174672)
    # SSA begins for if statement (line 749)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 750):
    
    # Assigning a BinOp to a Name (line 750):
    
    # Obtaining the type of the subscript
    int_174673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 15), 'int')
    slice_174674 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 750, 12), None, int_174673, None)
    # Getting the type of 'c' (line 750)
    c_174675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 750)
    getitem___174676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 12), c_174675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 750)
    subscript_call_result_174677 = invoke(stypy.reporting.localization.Localization(__file__, 750, 12), getitem___174676, slice_174674)
    
    int_174678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 18), 'int')
    # Applying the binary operator '*' (line 750)
    result_mul_174679 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 12), '*', subscript_call_result_174677, int_174678)
    
    # Assigning a type to the variable 'c' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'c', result_mul_174679)
    # SSA branch for the else part of an if statement (line 749)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 752)
    # Processing the call arguments (line 752)
    # Getting the type of 'cnt' (line 752)
    cnt_174681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 23), 'cnt', False)
    # Processing the call keyword arguments (line 752)
    kwargs_174682 = {}
    # Getting the type of 'range' (line 752)
    range_174680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 17), 'range', False)
    # Calling range(args, kwargs) (line 752)
    range_call_result_174683 = invoke(stypy.reporting.localization.Localization(__file__, 752, 17), range_174680, *[cnt_174681], **kwargs_174682)
    
    # Testing the type of a for loop iterable (line 752)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 752, 8), range_call_result_174683)
    # Getting the type of the for loop variable (line 752)
    for_loop_var_174684 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 752, 8), range_call_result_174683)
    # Assigning a type to the variable 'i' (line 752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'i', for_loop_var_174684)
    # SSA begins for a for statement (line 752)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 753):
    
    # Assigning a BinOp to a Name (line 753):
    # Getting the type of 'n' (line 753)
    n_174685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 16), 'n')
    int_174686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 20), 'int')
    # Applying the binary operator '-' (line 753)
    result_sub_174687 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 16), '-', n_174685, int_174686)
    
    # Assigning a type to the variable 'n' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), 'n', result_sub_174687)
    
    # Getting the type of 'c' (line 754)
    c_174688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'c')
    # Getting the type of 'scl' (line 754)
    scl_174689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 17), 'scl')
    # Applying the binary operator '*=' (line 754)
    result_imul_174690 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 12), '*=', c_174688, scl_174689)
    # Assigning a type to the variable 'c' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'c', result_imul_174690)
    
    
    # Assigning a Call to a Name (line 755):
    
    # Assigning a Call to a Name (line 755):
    
    # Call to empty(...): (line 755)
    # Processing the call arguments (line 755)
    
    # Obtaining an instance of the builtin type 'tuple' (line 755)
    tuple_174693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 755)
    # Adding element type (line 755)
    # Getting the type of 'n' (line 755)
    n_174694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 28), tuple_174693, n_174694)
    
    
    # Obtaining the type of the subscript
    int_174695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 42), 'int')
    slice_174696 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 755, 34), int_174695, None, None)
    # Getting the type of 'c' (line 755)
    c_174697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 755)
    shape_174698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 34), c_174697, 'shape')
    # Obtaining the member '__getitem__' of a type (line 755)
    getitem___174699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 34), shape_174698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 755)
    subscript_call_result_174700 = invoke(stypy.reporting.localization.Localization(__file__, 755, 34), getitem___174699, slice_174696)
    
    # Applying the binary operator '+' (line 755)
    result_add_174701 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 27), '+', tuple_174693, subscript_call_result_174700)
    
    # Processing the call keyword arguments (line 755)
    # Getting the type of 'c' (line 755)
    c_174702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 755)
    dtype_174703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 53), c_174702, 'dtype')
    keyword_174704 = dtype_174703
    kwargs_174705 = {'dtype': keyword_174704}
    # Getting the type of 'np' (line 755)
    np_174691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 755)
    empty_174692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 18), np_174691, 'empty')
    # Calling empty(args, kwargs) (line 755)
    empty_call_result_174706 = invoke(stypy.reporting.localization.Localization(__file__, 755, 18), empty_174692, *[result_add_174701], **kwargs_174705)
    
    # Assigning a type to the variable 'der' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'der', empty_call_result_174706)
    
    
    # Call to range(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'n' (line 756)
    n_174708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 27), 'n', False)
    int_174709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 30), 'int')
    int_174710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 33), 'int')
    # Processing the call keyword arguments (line 756)
    kwargs_174711 = {}
    # Getting the type of 'range' (line 756)
    range_174707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 21), 'range', False)
    # Calling range(args, kwargs) (line 756)
    range_call_result_174712 = invoke(stypy.reporting.localization.Localization(__file__, 756, 21), range_174707, *[n_174708, int_174709, int_174710], **kwargs_174711)
    
    # Testing the type of a for loop iterable (line 756)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 756, 12), range_call_result_174712)
    # Getting the type of the for loop variable (line 756)
    for_loop_var_174713 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 756, 12), range_call_result_174712)
    # Assigning a type to the variable 'j' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'j', for_loop_var_174713)
    # SSA begins for a for statement (line 756)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 757):
    
    # Assigning a BinOp to a Subscript (line 757):
    int_174714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 30), 'int')
    # Getting the type of 'j' (line 757)
    j_174715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 32), 'j')
    # Applying the binary operator '*' (line 757)
    result_mul_174716 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 30), '*', int_174714, j_174715)
    
    int_174717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 36), 'int')
    # Applying the binary operator '-' (line 757)
    result_sub_174718 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 30), '-', result_mul_174716, int_174717)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 757)
    j_174719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 41), 'j')
    # Getting the type of 'c' (line 757)
    c_174720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 39), 'c')
    # Obtaining the member '__getitem__' of a type (line 757)
    getitem___174721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 39), c_174720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 757)
    subscript_call_result_174722 = invoke(stypy.reporting.localization.Localization(__file__, 757, 39), getitem___174721, j_174719)
    
    # Applying the binary operator '*' (line 757)
    result_mul_174723 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 29), '*', result_sub_174718, subscript_call_result_174722)
    
    # Getting the type of 'der' (line 757)
    der_174724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 16), 'der')
    # Getting the type of 'j' (line 757)
    j_174725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 20), 'j')
    int_174726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 24), 'int')
    # Applying the binary operator '-' (line 757)
    result_sub_174727 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 20), '-', j_174725, int_174726)
    
    # Storing an element on a container (line 757)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 16), der_174724, (result_sub_174727, result_mul_174723))
    
    # Getting the type of 'c' (line 758)
    c_174728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'c')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 758)
    j_174729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 18), 'j')
    int_174730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 22), 'int')
    # Applying the binary operator '-' (line 758)
    result_sub_174731 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 18), '-', j_174729, int_174730)
    
    # Getting the type of 'c' (line 758)
    c_174732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'c')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___174733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 16), c_174732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_174734 = invoke(stypy.reporting.localization.Localization(__file__, 758, 16), getitem___174733, result_sub_174731)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 758)
    j_174735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 30), 'j')
    # Getting the type of 'c' (line 758)
    c_174736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 28), 'c')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___174737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 28), c_174736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_174738 = invoke(stypy.reporting.localization.Localization(__file__, 758, 28), getitem___174737, j_174735)
    
    # Applying the binary operator '+=' (line 758)
    result_iadd_174739 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 16), '+=', subscript_call_result_174734, subscript_call_result_174738)
    # Getting the type of 'c' (line 758)
    c_174740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'c')
    # Getting the type of 'j' (line 758)
    j_174741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 18), 'j')
    int_174742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 22), 'int')
    # Applying the binary operator '-' (line 758)
    result_sub_174743 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 18), '-', j_174741, int_174742)
    
    # Storing an element on a container (line 758)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 16), c_174740, (result_sub_174743, result_iadd_174739))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 759)
    n_174744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), 'n')
    int_174745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 19), 'int')
    # Applying the binary operator '>' (line 759)
    result_gt_174746 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 15), '>', n_174744, int_174745)
    
    # Testing the type of an if condition (line 759)
    if_condition_174747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 759, 12), result_gt_174746)
    # Assigning a type to the variable 'if_condition_174747' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'if_condition_174747', if_condition_174747)
    # SSA begins for if statement (line 759)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 760):
    
    # Assigning a BinOp to a Subscript (line 760):
    int_174748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 25), 'int')
    
    # Obtaining the type of the subscript
    int_174749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 29), 'int')
    # Getting the type of 'c' (line 760)
    c_174750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 27), 'c')
    # Obtaining the member '__getitem__' of a type (line 760)
    getitem___174751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 27), c_174750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 760)
    subscript_call_result_174752 = invoke(stypy.reporting.localization.Localization(__file__, 760, 27), getitem___174751, int_174749)
    
    # Applying the binary operator '*' (line 760)
    result_mul_174753 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 25), '*', int_174748, subscript_call_result_174752)
    
    # Getting the type of 'der' (line 760)
    der_174754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'der')
    int_174755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 20), 'int')
    # Storing an element on a container (line 760)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 16), der_174754, (int_174755, result_mul_174753))
    # SSA join for if statement (line 759)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 761):
    
    # Assigning a Subscript to a Subscript (line 761):
    
    # Obtaining the type of the subscript
    int_174756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 23), 'int')
    # Getting the type of 'c' (line 761)
    c_174757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___174758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 21), c_174757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_174759 = invoke(stypy.reporting.localization.Localization(__file__, 761, 21), getitem___174758, int_174756)
    
    # Getting the type of 'der' (line 761)
    der_174760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'der')
    int_174761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 16), 'int')
    # Storing an element on a container (line 761)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 12), der_174760, (int_174761, subscript_call_result_174759))
    
    # Assigning a Name to a Name (line 762):
    
    # Assigning a Name to a Name (line 762):
    # Getting the type of 'der' (line 762)
    der_174762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 16), 'der')
    # Assigning a type to the variable 'c' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'c', der_174762)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 749)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 763):
    
    # Assigning a Call to a Name (line 763):
    
    # Call to rollaxis(...): (line 763)
    # Processing the call arguments (line 763)
    # Getting the type of 'c' (line 763)
    c_174765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 20), 'c', False)
    int_174766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 23), 'int')
    # Getting the type of 'iaxis' (line 763)
    iaxis_174767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 26), 'iaxis', False)
    int_174768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 34), 'int')
    # Applying the binary operator '+' (line 763)
    result_add_174769 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 26), '+', iaxis_174767, int_174768)
    
    # Processing the call keyword arguments (line 763)
    kwargs_174770 = {}
    # Getting the type of 'np' (line 763)
    np_174763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 763)
    rollaxis_174764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 8), np_174763, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 763)
    rollaxis_call_result_174771 = invoke(stypy.reporting.localization.Localization(__file__, 763, 8), rollaxis_174764, *[c_174765, int_174766, result_add_174769], **kwargs_174770)
    
    # Assigning a type to the variable 'c' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'c', rollaxis_call_result_174771)
    # Getting the type of 'c' (line 764)
    c_174772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'stypy_return_type', c_174772)
    
    # ################# End of 'legder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legder' in the type store
    # Getting the type of 'stypy_return_type' (line 668)
    stypy_return_type_174773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_174773)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legder'
    return stypy_return_type_174773

# Assigning a type to the variable 'legder' (line 668)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 0), 'legder', legder)

@norecursion
def legint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_174774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 16), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 767)
    list_174775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 767)
    
    int_174776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 30), 'int')
    int_174777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 37), 'int')
    int_174778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 45), 'int')
    defaults = [int_174774, list_174775, int_174776, int_174777, int_174778]
    # Create a new context for function 'legint'
    module_type_store = module_type_store.open_function_context('legint', 767, 0, False)
    
    # Passed parameters checking function
    legint.stypy_localization = localization
    legint.stypy_type_of_self = None
    legint.stypy_type_store = module_type_store
    legint.stypy_function_name = 'legint'
    legint.stypy_param_names_list = ['c', 'm', 'k', 'lbnd', 'scl', 'axis']
    legint.stypy_varargs_param_name = None
    legint.stypy_kwargs_param_name = None
    legint.stypy_call_defaults = defaults
    legint.stypy_call_varargs = varargs
    legint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legint', ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legint', localization, ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legint(...)' code ##################

    str_174779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, (-1)), 'str', '\n    Integrate a Legendre series.\n\n    Returns the Legendre series coefficients `c` integrated `m` times from\n    `lbnd` along `axis`. At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.)  The argument `c` is an array of\n    coefficients from low to high degree along each axis, e.g., [1,2,3]\n    represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]\n    represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +\n    2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Legendre series coefficients. If c is multidimensional the\n        different axis correspond to different variables with the degree in\n        each axis given by the corresponding index.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at\n        ``lbnd`` is the first value in the list, the value of the second\n        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the\n        default), all constants are set to zero.  If ``m == 1``, a single\n        scalar can be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        Legendre series coefficient array of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or\n        ``np.isscalar(scl) == False``.\n\n    See Also\n    --------\n    legder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.\n    Why is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`.  Then\n    .. math::`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a` - perhaps not what one would have first thought.\n\n    Also note that, in general, the result of integrating a C-series needs\n    to be "reprojected" onto the C-series basis set.  Thus, typically,\n    the result of this function is "unintuitive," albeit correct; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import legendre as L\n    >>> c = (1,2,3)\n    >>> L.legint(c)\n    array([ 0.33333333,  0.4       ,  0.66666667,  0.6       ])\n    >>> L.legint(c, 3)\n    array([  1.66666667e-02,  -1.78571429e-02,   4.76190476e-02,\n            -1.73472348e-18,   1.90476190e-02,   9.52380952e-03])\n    >>> L.legint(c, k=3)\n    array([ 3.33333333,  0.4       ,  0.66666667,  0.6       ])\n    >>> L.legint(c, lbnd=-2)\n    array([ 7.33333333,  0.4       ,  0.66666667,  0.6       ])\n    >>> L.legint(c, scl=2)\n    array([ 0.66666667,  0.8       ,  1.33333333,  1.2       ])\n\n    ')
    
    # Assigning a Call to a Name (line 852):
    
    # Assigning a Call to a Name (line 852):
    
    # Call to array(...): (line 852)
    # Processing the call arguments (line 852)
    # Getting the type of 'c' (line 852)
    c_174782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 17), 'c', False)
    # Processing the call keyword arguments (line 852)
    int_174783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 26), 'int')
    keyword_174784 = int_174783
    int_174785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 34), 'int')
    keyword_174786 = int_174785
    kwargs_174787 = {'copy': keyword_174786, 'ndmin': keyword_174784}
    # Getting the type of 'np' (line 852)
    np_174780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 852)
    array_174781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 8), np_174780, 'array')
    # Calling array(args, kwargs) (line 852)
    array_call_result_174788 = invoke(stypy.reporting.localization.Localization(__file__, 852, 8), array_174781, *[c_174782], **kwargs_174787)
    
    # Assigning a type to the variable 'c' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 4), 'c', array_call_result_174788)
    
    
    # Getting the type of 'c' (line 853)
    c_174789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 853)
    dtype_174790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 7), c_174789, 'dtype')
    # Obtaining the member 'char' of a type (line 853)
    char_174791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 7), dtype_174790, 'char')
    str_174792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 853)
    result_contains_174793 = python_operator(stypy.reporting.localization.Localization(__file__, 853, 7), 'in', char_174791, str_174792)
    
    # Testing the type of an if condition (line 853)
    if_condition_174794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 853, 4), result_contains_174793)
    # Assigning a type to the variable 'if_condition_174794' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'if_condition_174794', if_condition_174794)
    # SSA begins for if statement (line 853)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 854):
    
    # Assigning a Call to a Name (line 854):
    
    # Call to astype(...): (line 854)
    # Processing the call arguments (line 854)
    # Getting the type of 'np' (line 854)
    np_174797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 854)
    double_174798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 21), np_174797, 'double')
    # Processing the call keyword arguments (line 854)
    kwargs_174799 = {}
    # Getting the type of 'c' (line 854)
    c_174795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 854)
    astype_174796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 12), c_174795, 'astype')
    # Calling astype(args, kwargs) (line 854)
    astype_call_result_174800 = invoke(stypy.reporting.localization.Localization(__file__, 854, 12), astype_174796, *[double_174798], **kwargs_174799)
    
    # Assigning a type to the variable 'c' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'c', astype_call_result_174800)
    # SSA join for if statement (line 853)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to iterable(...): (line 855)
    # Processing the call arguments (line 855)
    # Getting the type of 'k' (line 855)
    k_174803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 23), 'k', False)
    # Processing the call keyword arguments (line 855)
    kwargs_174804 = {}
    # Getting the type of 'np' (line 855)
    np_174801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 11), 'np', False)
    # Obtaining the member 'iterable' of a type (line 855)
    iterable_174802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 11), np_174801, 'iterable')
    # Calling iterable(args, kwargs) (line 855)
    iterable_call_result_174805 = invoke(stypy.reporting.localization.Localization(__file__, 855, 11), iterable_174802, *[k_174803], **kwargs_174804)
    
    # Applying the 'not' unary operator (line 855)
    result_not__174806 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 7), 'not', iterable_call_result_174805)
    
    # Testing the type of an if condition (line 855)
    if_condition_174807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 855, 4), result_not__174806)
    # Assigning a type to the variable 'if_condition_174807' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 4), 'if_condition_174807', if_condition_174807)
    # SSA begins for if statement (line 855)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 856):
    
    # Assigning a List to a Name (line 856):
    
    # Obtaining an instance of the builtin type 'list' (line 856)
    list_174808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 856)
    # Adding element type (line 856)
    # Getting the type of 'k' (line 856)
    k_174809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 856, 12), list_174808, k_174809)
    
    # Assigning a type to the variable 'k' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 8), 'k', list_174808)
    # SSA join for if statement (line 855)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 857):
    
    # Assigning a Subscript to a Name (line 857):
    
    # Obtaining the type of the subscript
    int_174810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 857)
    list_174815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 857)
    # Adding element type (line 857)
    # Getting the type of 'm' (line 857)
    m_174816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 34), list_174815, m_174816)
    # Adding element type (line 857)
    # Getting the type of 'axis' (line 857)
    axis_174817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 34), list_174815, axis_174817)
    
    comprehension_174818 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 18), list_174815)
    # Assigning a type to the variable 't' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 18), 't', comprehension_174818)
    
    # Call to int(...): (line 857)
    # Processing the call arguments (line 857)
    # Getting the type of 't' (line 857)
    t_174812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 22), 't', False)
    # Processing the call keyword arguments (line 857)
    kwargs_174813 = {}
    # Getting the type of 'int' (line 857)
    int_174811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 18), 'int', False)
    # Calling int(args, kwargs) (line 857)
    int_call_result_174814 = invoke(stypy.reporting.localization.Localization(__file__, 857, 18), int_174811, *[t_174812], **kwargs_174813)
    
    list_174819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 18), list_174819, int_call_result_174814)
    # Obtaining the member '__getitem__' of a type (line 857)
    getitem___174820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 4), list_174819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 857)
    subscript_call_result_174821 = invoke(stypy.reporting.localization.Localization(__file__, 857, 4), getitem___174820, int_174810)
    
    # Assigning a type to the variable 'tuple_var_assignment_173630' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'tuple_var_assignment_173630', subscript_call_result_174821)
    
    # Assigning a Subscript to a Name (line 857):
    
    # Obtaining the type of the subscript
    int_174822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 857)
    list_174827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 857)
    # Adding element type (line 857)
    # Getting the type of 'm' (line 857)
    m_174828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 34), list_174827, m_174828)
    # Adding element type (line 857)
    # Getting the type of 'axis' (line 857)
    axis_174829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 34), list_174827, axis_174829)
    
    comprehension_174830 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 18), list_174827)
    # Assigning a type to the variable 't' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 18), 't', comprehension_174830)
    
    # Call to int(...): (line 857)
    # Processing the call arguments (line 857)
    # Getting the type of 't' (line 857)
    t_174824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 22), 't', False)
    # Processing the call keyword arguments (line 857)
    kwargs_174825 = {}
    # Getting the type of 'int' (line 857)
    int_174823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 18), 'int', False)
    # Calling int(args, kwargs) (line 857)
    int_call_result_174826 = invoke(stypy.reporting.localization.Localization(__file__, 857, 18), int_174823, *[t_174824], **kwargs_174825)
    
    list_174831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 18), list_174831, int_call_result_174826)
    # Obtaining the member '__getitem__' of a type (line 857)
    getitem___174832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 4), list_174831, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 857)
    subscript_call_result_174833 = invoke(stypy.reporting.localization.Localization(__file__, 857, 4), getitem___174832, int_174822)
    
    # Assigning a type to the variable 'tuple_var_assignment_173631' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'tuple_var_assignment_173631', subscript_call_result_174833)
    
    # Assigning a Name to a Name (line 857):
    # Getting the type of 'tuple_var_assignment_173630' (line 857)
    tuple_var_assignment_173630_174834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'tuple_var_assignment_173630')
    # Assigning a type to the variable 'cnt' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'cnt', tuple_var_assignment_173630_174834)
    
    # Assigning a Name to a Name (line 857):
    # Getting the type of 'tuple_var_assignment_173631' (line 857)
    tuple_var_assignment_173631_174835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'tuple_var_assignment_173631')
    # Assigning a type to the variable 'iaxis' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 9), 'iaxis', tuple_var_assignment_173631_174835)
    
    
    # Getting the type of 'cnt' (line 859)
    cnt_174836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 7), 'cnt')
    # Getting the type of 'm' (line 859)
    m_174837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 14), 'm')
    # Applying the binary operator '!=' (line 859)
    result_ne_174838 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 7), '!=', cnt_174836, m_174837)
    
    # Testing the type of an if condition (line 859)
    if_condition_174839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 859, 4), result_ne_174838)
    # Assigning a type to the variable 'if_condition_174839' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'if_condition_174839', if_condition_174839)
    # SSA begins for if statement (line 859)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 860)
    # Processing the call arguments (line 860)
    str_174841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 25), 'str', 'The order of integration must be integer')
    # Processing the call keyword arguments (line 860)
    kwargs_174842 = {}
    # Getting the type of 'ValueError' (line 860)
    ValueError_174840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 860)
    ValueError_call_result_174843 = invoke(stypy.reporting.localization.Localization(__file__, 860, 14), ValueError_174840, *[str_174841], **kwargs_174842)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 860, 8), ValueError_call_result_174843, 'raise parameter', BaseException)
    # SSA join for if statement (line 859)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 861)
    cnt_174844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 7), 'cnt')
    int_174845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 13), 'int')
    # Applying the binary operator '<' (line 861)
    result_lt_174846 = python_operator(stypy.reporting.localization.Localization(__file__, 861, 7), '<', cnt_174844, int_174845)
    
    # Testing the type of an if condition (line 861)
    if_condition_174847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 861, 4), result_lt_174846)
    # Assigning a type to the variable 'if_condition_174847' (line 861)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 4), 'if_condition_174847', if_condition_174847)
    # SSA begins for if statement (line 861)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 862)
    # Processing the call arguments (line 862)
    str_174849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 25), 'str', 'The order of integration must be non-negative')
    # Processing the call keyword arguments (line 862)
    kwargs_174850 = {}
    # Getting the type of 'ValueError' (line 862)
    ValueError_174848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 862)
    ValueError_call_result_174851 = invoke(stypy.reporting.localization.Localization(__file__, 862, 14), ValueError_174848, *[str_174849], **kwargs_174850)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 862, 8), ValueError_call_result_174851, 'raise parameter', BaseException)
    # SSA join for if statement (line 861)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 863)
    # Processing the call arguments (line 863)
    # Getting the type of 'k' (line 863)
    k_174853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 11), 'k', False)
    # Processing the call keyword arguments (line 863)
    kwargs_174854 = {}
    # Getting the type of 'len' (line 863)
    len_174852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 7), 'len', False)
    # Calling len(args, kwargs) (line 863)
    len_call_result_174855 = invoke(stypy.reporting.localization.Localization(__file__, 863, 7), len_174852, *[k_174853], **kwargs_174854)
    
    # Getting the type of 'cnt' (line 863)
    cnt_174856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 16), 'cnt')
    # Applying the binary operator '>' (line 863)
    result_gt_174857 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 7), '>', len_call_result_174855, cnt_174856)
    
    # Testing the type of an if condition (line 863)
    if_condition_174858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 863, 4), result_gt_174857)
    # Assigning a type to the variable 'if_condition_174858' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'if_condition_174858', if_condition_174858)
    # SSA begins for if statement (line 863)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 864)
    # Processing the call arguments (line 864)
    str_174860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 25), 'str', 'Too many integration constants')
    # Processing the call keyword arguments (line 864)
    kwargs_174861 = {}
    # Getting the type of 'ValueError' (line 864)
    ValueError_174859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 864)
    ValueError_call_result_174862 = invoke(stypy.reporting.localization.Localization(__file__, 864, 14), ValueError_174859, *[str_174860], **kwargs_174861)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 864, 8), ValueError_call_result_174862, 'raise parameter', BaseException)
    # SSA join for if statement (line 863)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 865)
    iaxis_174863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 7), 'iaxis')
    # Getting the type of 'axis' (line 865)
    axis_174864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 16), 'axis')
    # Applying the binary operator '!=' (line 865)
    result_ne_174865 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 7), '!=', iaxis_174863, axis_174864)
    
    # Testing the type of an if condition (line 865)
    if_condition_174866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 865, 4), result_ne_174865)
    # Assigning a type to the variable 'if_condition_174866' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'if_condition_174866', if_condition_174866)
    # SSA begins for if statement (line 865)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 866)
    # Processing the call arguments (line 866)
    str_174868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 866)
    kwargs_174869 = {}
    # Getting the type of 'ValueError' (line 866)
    ValueError_174867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 866)
    ValueError_call_result_174870 = invoke(stypy.reporting.localization.Localization(__file__, 866, 14), ValueError_174867, *[str_174868], **kwargs_174869)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 866, 8), ValueError_call_result_174870, 'raise parameter', BaseException)
    # SSA join for if statement (line 865)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 867)
    c_174871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 867)
    ndim_174872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 12), c_174871, 'ndim')
    # Applying the 'usub' unary operator (line 867)
    result___neg___174873 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), 'usub', ndim_174872)
    
    # Getting the type of 'iaxis' (line 867)
    iaxis_174874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 22), 'iaxis')
    # Applying the binary operator '<=' (line 867)
    result_le_174875 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), '<=', result___neg___174873, iaxis_174874)
    # Getting the type of 'c' (line 867)
    c_174876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 867)
    ndim_174877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 30), c_174876, 'ndim')
    # Applying the binary operator '<' (line 867)
    result_lt_174878 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), '<', iaxis_174874, ndim_174877)
    # Applying the binary operator '&' (line 867)
    result_and__174879 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), '&', result_le_174875, result_lt_174878)
    
    # Applying the 'not' unary operator (line 867)
    result_not__174880 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 7), 'not', result_and__174879)
    
    # Testing the type of an if condition (line 867)
    if_condition_174881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 867, 4), result_not__174880)
    # Assigning a type to the variable 'if_condition_174881' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 4), 'if_condition_174881', if_condition_174881)
    # SSA begins for if statement (line 867)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 868)
    # Processing the call arguments (line 868)
    str_174883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 868)
    kwargs_174884 = {}
    # Getting the type of 'ValueError' (line 868)
    ValueError_174882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 868)
    ValueError_call_result_174885 = invoke(stypy.reporting.localization.Localization(__file__, 868, 14), ValueError_174882, *[str_174883], **kwargs_174884)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 868, 8), ValueError_call_result_174885, 'raise parameter', BaseException)
    # SSA join for if statement (line 867)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 869)
    iaxis_174886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 7), 'iaxis')
    int_174887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 15), 'int')
    # Applying the binary operator '<' (line 869)
    result_lt_174888 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 7), '<', iaxis_174886, int_174887)
    
    # Testing the type of an if condition (line 869)
    if_condition_174889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 869, 4), result_lt_174888)
    # Assigning a type to the variable 'if_condition_174889' (line 869)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 4), 'if_condition_174889', if_condition_174889)
    # SSA begins for if statement (line 869)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 870)
    iaxis_174890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'iaxis')
    # Getting the type of 'c' (line 870)
    c_174891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 870)
    ndim_174892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 17), c_174891, 'ndim')
    # Applying the binary operator '+=' (line 870)
    result_iadd_174893 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 8), '+=', iaxis_174890, ndim_174892)
    # Assigning a type to the variable 'iaxis' (line 870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'iaxis', result_iadd_174893)
    
    # SSA join for if statement (line 869)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 872)
    cnt_174894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 7), 'cnt')
    int_174895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 14), 'int')
    # Applying the binary operator '==' (line 872)
    result_eq_174896 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 7), '==', cnt_174894, int_174895)
    
    # Testing the type of an if condition (line 872)
    if_condition_174897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 4), result_eq_174896)
    # Assigning a type to the variable 'if_condition_174897' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 4), 'if_condition_174897', if_condition_174897)
    # SSA begins for if statement (line 872)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 873)
    c_174898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 8), 'stypy_return_type', c_174898)
    # SSA join for if statement (line 872)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 875):
    
    # Assigning a Call to a Name (line 875):
    
    # Call to rollaxis(...): (line 875)
    # Processing the call arguments (line 875)
    # Getting the type of 'c' (line 875)
    c_174901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 20), 'c', False)
    # Getting the type of 'iaxis' (line 875)
    iaxis_174902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 875)
    kwargs_174903 = {}
    # Getting the type of 'np' (line 875)
    np_174899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 875)
    rollaxis_174900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 8), np_174899, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 875)
    rollaxis_call_result_174904 = invoke(stypy.reporting.localization.Localization(__file__, 875, 8), rollaxis_174900, *[c_174901, iaxis_174902], **kwargs_174903)
    
    # Assigning a type to the variable 'c' (line 875)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), 'c', rollaxis_call_result_174904)
    
    # Assigning a BinOp to a Name (line 876):
    
    # Assigning a BinOp to a Name (line 876):
    
    # Call to list(...): (line 876)
    # Processing the call arguments (line 876)
    # Getting the type of 'k' (line 876)
    k_174906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 13), 'k', False)
    # Processing the call keyword arguments (line 876)
    kwargs_174907 = {}
    # Getting the type of 'list' (line 876)
    list_174905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'list', False)
    # Calling list(args, kwargs) (line 876)
    list_call_result_174908 = invoke(stypy.reporting.localization.Localization(__file__, 876, 8), list_174905, *[k_174906], **kwargs_174907)
    
    
    # Obtaining an instance of the builtin type 'list' (line 876)
    list_174909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 876)
    # Adding element type (line 876)
    int_174910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 876, 18), list_174909, int_174910)
    
    # Getting the type of 'cnt' (line 876)
    cnt_174911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 23), 'cnt')
    
    # Call to len(...): (line 876)
    # Processing the call arguments (line 876)
    # Getting the type of 'k' (line 876)
    k_174913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 33), 'k', False)
    # Processing the call keyword arguments (line 876)
    kwargs_174914 = {}
    # Getting the type of 'len' (line 876)
    len_174912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 29), 'len', False)
    # Calling len(args, kwargs) (line 876)
    len_call_result_174915 = invoke(stypy.reporting.localization.Localization(__file__, 876, 29), len_174912, *[k_174913], **kwargs_174914)
    
    # Applying the binary operator '-' (line 876)
    result_sub_174916 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 23), '-', cnt_174911, len_call_result_174915)
    
    # Applying the binary operator '*' (line 876)
    result_mul_174917 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 18), '*', list_174909, result_sub_174916)
    
    # Applying the binary operator '+' (line 876)
    result_add_174918 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 8), '+', list_call_result_174908, result_mul_174917)
    
    # Assigning a type to the variable 'k' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 4), 'k', result_add_174918)
    
    
    # Call to range(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'cnt' (line 877)
    cnt_174920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 19), 'cnt', False)
    # Processing the call keyword arguments (line 877)
    kwargs_174921 = {}
    # Getting the type of 'range' (line 877)
    range_174919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 13), 'range', False)
    # Calling range(args, kwargs) (line 877)
    range_call_result_174922 = invoke(stypy.reporting.localization.Localization(__file__, 877, 13), range_174919, *[cnt_174920], **kwargs_174921)
    
    # Testing the type of a for loop iterable (line 877)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 877, 4), range_call_result_174922)
    # Getting the type of the for loop variable (line 877)
    for_loop_var_174923 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 877, 4), range_call_result_174922)
    # Assigning a type to the variable 'i' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 4), 'i', for_loop_var_174923)
    # SSA begins for a for statement (line 877)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 878):
    
    # Assigning a Call to a Name (line 878):
    
    # Call to len(...): (line 878)
    # Processing the call arguments (line 878)
    # Getting the type of 'c' (line 878)
    c_174925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 16), 'c', False)
    # Processing the call keyword arguments (line 878)
    kwargs_174926 = {}
    # Getting the type of 'len' (line 878)
    len_174924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 12), 'len', False)
    # Calling len(args, kwargs) (line 878)
    len_call_result_174927 = invoke(stypy.reporting.localization.Localization(__file__, 878, 12), len_174924, *[c_174925], **kwargs_174926)
    
    # Assigning a type to the variable 'n' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 8), 'n', len_call_result_174927)
    
    # Getting the type of 'c' (line 879)
    c_174928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'c')
    # Getting the type of 'scl' (line 879)
    scl_174929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 13), 'scl')
    # Applying the binary operator '*=' (line 879)
    result_imul_174930 = python_operator(stypy.reporting.localization.Localization(__file__, 879, 8), '*=', c_174928, scl_174929)
    # Assigning a type to the variable 'c' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'c', result_imul_174930)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 880)
    n_174931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 11), 'n')
    int_174932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 16), 'int')
    # Applying the binary operator '==' (line 880)
    result_eq_174933 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 11), '==', n_174931, int_174932)
    
    
    # Call to all(...): (line 880)
    # Processing the call arguments (line 880)
    
    
    # Obtaining the type of the subscript
    int_174936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 31), 'int')
    # Getting the type of 'c' (line 880)
    c_174937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 880)
    getitem___174938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 29), c_174937, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 880)
    subscript_call_result_174939 = invoke(stypy.reporting.localization.Localization(__file__, 880, 29), getitem___174938, int_174936)
    
    int_174940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 37), 'int')
    # Applying the binary operator '==' (line 880)
    result_eq_174941 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 29), '==', subscript_call_result_174939, int_174940)
    
    # Processing the call keyword arguments (line 880)
    kwargs_174942 = {}
    # Getting the type of 'np' (line 880)
    np_174934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 22), 'np', False)
    # Obtaining the member 'all' of a type (line 880)
    all_174935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 22), np_174934, 'all')
    # Calling all(args, kwargs) (line 880)
    all_call_result_174943 = invoke(stypy.reporting.localization.Localization(__file__, 880, 22), all_174935, *[result_eq_174941], **kwargs_174942)
    
    # Applying the binary operator 'and' (line 880)
    result_and_keyword_174944 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 11), 'and', result_eq_174933, all_call_result_174943)
    
    # Testing the type of an if condition (line 880)
    if_condition_174945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 880, 8), result_and_keyword_174944)
    # Assigning a type to the variable 'if_condition_174945' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 8), 'if_condition_174945', if_condition_174945)
    # SSA begins for if statement (line 880)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 881)
    c_174946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'c')
    
    # Obtaining the type of the subscript
    int_174947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 14), 'int')
    # Getting the type of 'c' (line 881)
    c_174948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 881)
    getitem___174949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 12), c_174948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 881)
    subscript_call_result_174950 = invoke(stypy.reporting.localization.Localization(__file__, 881, 12), getitem___174949, int_174947)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 881)
    i_174951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 22), 'i')
    # Getting the type of 'k' (line 881)
    k_174952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'k')
    # Obtaining the member '__getitem__' of a type (line 881)
    getitem___174953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), k_174952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 881)
    subscript_call_result_174954 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), getitem___174953, i_174951)
    
    # Applying the binary operator '+=' (line 881)
    result_iadd_174955 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 12), '+=', subscript_call_result_174950, subscript_call_result_174954)
    # Getting the type of 'c' (line 881)
    c_174956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'c')
    int_174957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 14), 'int')
    # Storing an element on a container (line 881)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 12), c_174956, (int_174957, result_iadd_174955))
    
    # SSA branch for the else part of an if statement (line 880)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 883):
    
    # Assigning a Call to a Name (line 883):
    
    # Call to empty(...): (line 883)
    # Processing the call arguments (line 883)
    
    # Obtaining an instance of the builtin type 'tuple' (line 883)
    tuple_174960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 883)
    # Adding element type (line 883)
    # Getting the type of 'n' (line 883)
    n_174961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 28), 'n', False)
    int_174962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 32), 'int')
    # Applying the binary operator '+' (line 883)
    result_add_174963 = python_operator(stypy.reporting.localization.Localization(__file__, 883, 28), '+', n_174961, int_174962)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 28), tuple_174960, result_add_174963)
    
    
    # Obtaining the type of the subscript
    int_174964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 46), 'int')
    slice_174965 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 883, 38), int_174964, None, None)
    # Getting the type of 'c' (line 883)
    c_174966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 883)
    shape_174967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 38), c_174966, 'shape')
    # Obtaining the member '__getitem__' of a type (line 883)
    getitem___174968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 38), shape_174967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 883)
    subscript_call_result_174969 = invoke(stypy.reporting.localization.Localization(__file__, 883, 38), getitem___174968, slice_174965)
    
    # Applying the binary operator '+' (line 883)
    result_add_174970 = python_operator(stypy.reporting.localization.Localization(__file__, 883, 27), '+', tuple_174960, subscript_call_result_174969)
    
    # Processing the call keyword arguments (line 883)
    # Getting the type of 'c' (line 883)
    c_174971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 57), 'c', False)
    # Obtaining the member 'dtype' of a type (line 883)
    dtype_174972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 57), c_174971, 'dtype')
    keyword_174973 = dtype_174972
    kwargs_174974 = {'dtype': keyword_174973}
    # Getting the type of 'np' (line 883)
    np_174958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 883)
    empty_174959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 18), np_174958, 'empty')
    # Calling empty(args, kwargs) (line 883)
    empty_call_result_174975 = invoke(stypy.reporting.localization.Localization(__file__, 883, 18), empty_174959, *[result_add_174970], **kwargs_174974)
    
    # Assigning a type to the variable 'tmp' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 12), 'tmp', empty_call_result_174975)
    
    # Assigning a BinOp to a Subscript (line 884):
    
    # Assigning a BinOp to a Subscript (line 884):
    
    # Obtaining the type of the subscript
    int_174976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 23), 'int')
    # Getting the type of 'c' (line 884)
    c_174977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 884)
    getitem___174978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 21), c_174977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 884)
    subscript_call_result_174979 = invoke(stypy.reporting.localization.Localization(__file__, 884, 21), getitem___174978, int_174976)
    
    int_174980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 26), 'int')
    # Applying the binary operator '*' (line 884)
    result_mul_174981 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 21), '*', subscript_call_result_174979, int_174980)
    
    # Getting the type of 'tmp' (line 884)
    tmp_174982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 12), 'tmp')
    int_174983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 16), 'int')
    # Storing an element on a container (line 884)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 884, 12), tmp_174982, (int_174983, result_mul_174981))
    
    # Assigning a Subscript to a Subscript (line 885):
    
    # Assigning a Subscript to a Subscript (line 885):
    
    # Obtaining the type of the subscript
    int_174984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 23), 'int')
    # Getting the type of 'c' (line 885)
    c_174985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 885)
    getitem___174986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 21), c_174985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 885)
    subscript_call_result_174987 = invoke(stypy.reporting.localization.Localization(__file__, 885, 21), getitem___174986, int_174984)
    
    # Getting the type of 'tmp' (line 885)
    tmp_174988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'tmp')
    int_174989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 16), 'int')
    # Storing an element on a container (line 885)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 885, 12), tmp_174988, (int_174989, subscript_call_result_174987))
    
    
    # Getting the type of 'n' (line 886)
    n_174990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 15), 'n')
    int_174991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 19), 'int')
    # Applying the binary operator '>' (line 886)
    result_gt_174992 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 15), '>', n_174990, int_174991)
    
    # Testing the type of an if condition (line 886)
    if_condition_174993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 886, 12), result_gt_174992)
    # Assigning a type to the variable 'if_condition_174993' (line 886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 12), 'if_condition_174993', if_condition_174993)
    # SSA begins for if statement (line 886)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 887):
    
    # Assigning a BinOp to a Subscript (line 887):
    
    # Obtaining the type of the subscript
    int_174994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 27), 'int')
    # Getting the type of 'c' (line 887)
    c_174995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 25), 'c')
    # Obtaining the member '__getitem__' of a type (line 887)
    getitem___174996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 25), c_174995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 887)
    subscript_call_result_174997 = invoke(stypy.reporting.localization.Localization(__file__, 887, 25), getitem___174996, int_174994)
    
    int_174998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 30), 'int')
    # Applying the binary operator 'div' (line 887)
    result_div_174999 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 25), 'div', subscript_call_result_174997, int_174998)
    
    # Getting the type of 'tmp' (line 887)
    tmp_175000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 16), 'tmp')
    int_175001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 20), 'int')
    # Storing an element on a container (line 887)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 887, 16), tmp_175000, (int_175001, result_div_174999))
    # SSA join for if statement (line 886)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 888)
    # Processing the call arguments (line 888)
    int_175003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 27), 'int')
    # Getting the type of 'n' (line 888)
    n_175004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 30), 'n', False)
    # Processing the call keyword arguments (line 888)
    kwargs_175005 = {}
    # Getting the type of 'range' (line 888)
    range_175002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 21), 'range', False)
    # Calling range(args, kwargs) (line 888)
    range_call_result_175006 = invoke(stypy.reporting.localization.Localization(__file__, 888, 21), range_175002, *[int_175003, n_175004], **kwargs_175005)
    
    # Testing the type of a for loop iterable (line 888)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 888, 12), range_call_result_175006)
    # Getting the type of the for loop variable (line 888)
    for_loop_var_175007 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 888, 12), range_call_result_175006)
    # Assigning a type to the variable 'j' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 12), 'j', for_loop_var_175007)
    # SSA begins for a for statement (line 888)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 889):
    
    # Assigning a BinOp to a Name (line 889):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 889)
    j_175008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 22), 'j')
    # Getting the type of 'c' (line 889)
    c_175009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 20), 'c')
    # Obtaining the member '__getitem__' of a type (line 889)
    getitem___175010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 20), c_175009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 889)
    subscript_call_result_175011 = invoke(stypy.reporting.localization.Localization(__file__, 889, 20), getitem___175010, j_175008)
    
    int_175012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 26), 'int')
    # Getting the type of 'j' (line 889)
    j_175013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 28), 'j')
    # Applying the binary operator '*' (line 889)
    result_mul_175014 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 26), '*', int_175012, j_175013)
    
    int_175015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 32), 'int')
    # Applying the binary operator '+' (line 889)
    result_add_175016 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 26), '+', result_mul_175014, int_175015)
    
    # Applying the binary operator 'div' (line 889)
    result_div_175017 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 20), 'div', subscript_call_result_175011, result_add_175016)
    
    # Assigning a type to the variable 't' (line 889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 16), 't', result_div_175017)
    
    # Assigning a Name to a Subscript (line 890):
    
    # Assigning a Name to a Subscript (line 890):
    # Getting the type of 't' (line 890)
    t_175018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 29), 't')
    # Getting the type of 'tmp' (line 890)
    tmp_175019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 16), 'tmp')
    # Getting the type of 'j' (line 890)
    j_175020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 20), 'j')
    int_175021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 24), 'int')
    # Applying the binary operator '+' (line 890)
    result_add_175022 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 20), '+', j_175020, int_175021)
    
    # Storing an element on a container (line 890)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 890, 16), tmp_175019, (result_add_175022, t_175018))
    
    # Getting the type of 'tmp' (line 891)
    tmp_175023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'tmp')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 891)
    j_175024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 20), 'j')
    int_175025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 24), 'int')
    # Applying the binary operator '-' (line 891)
    result_sub_175026 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 20), '-', j_175024, int_175025)
    
    # Getting the type of 'tmp' (line 891)
    tmp_175027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 891)
    getitem___175028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 16), tmp_175027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 891)
    subscript_call_result_175029 = invoke(stypy.reporting.localization.Localization(__file__, 891, 16), getitem___175028, result_sub_175026)
    
    # Getting the type of 't' (line 891)
    t_175030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 30), 't')
    # Applying the binary operator '-=' (line 891)
    result_isub_175031 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 16), '-=', subscript_call_result_175029, t_175030)
    # Getting the type of 'tmp' (line 891)
    tmp_175032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'tmp')
    # Getting the type of 'j' (line 891)
    j_175033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 20), 'j')
    int_175034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 24), 'int')
    # Applying the binary operator '-' (line 891)
    result_sub_175035 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 20), '-', j_175033, int_175034)
    
    # Storing an element on a container (line 891)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 891, 16), tmp_175032, (result_sub_175035, result_isub_175031))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tmp' (line 892)
    tmp_175036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'tmp')
    
    # Obtaining the type of the subscript
    int_175037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 16), 'int')
    # Getting the type of 'tmp' (line 892)
    tmp_175038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 892)
    getitem___175039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 12), tmp_175038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 892)
    subscript_call_result_175040 = invoke(stypy.reporting.localization.Localization(__file__, 892, 12), getitem___175039, int_175037)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 892)
    i_175041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 24), 'i')
    # Getting the type of 'k' (line 892)
    k_175042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 22), 'k')
    # Obtaining the member '__getitem__' of a type (line 892)
    getitem___175043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 22), k_175042, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 892)
    subscript_call_result_175044 = invoke(stypy.reporting.localization.Localization(__file__, 892, 22), getitem___175043, i_175041)
    
    
    # Call to legval(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'lbnd' (line 892)
    lbnd_175046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 36), 'lbnd', False)
    # Getting the type of 'tmp' (line 892)
    tmp_175047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 42), 'tmp', False)
    # Processing the call keyword arguments (line 892)
    kwargs_175048 = {}
    # Getting the type of 'legval' (line 892)
    legval_175045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 29), 'legval', False)
    # Calling legval(args, kwargs) (line 892)
    legval_call_result_175049 = invoke(stypy.reporting.localization.Localization(__file__, 892, 29), legval_175045, *[lbnd_175046, tmp_175047], **kwargs_175048)
    
    # Applying the binary operator '-' (line 892)
    result_sub_175050 = python_operator(stypy.reporting.localization.Localization(__file__, 892, 22), '-', subscript_call_result_175044, legval_call_result_175049)
    
    # Applying the binary operator '+=' (line 892)
    result_iadd_175051 = python_operator(stypy.reporting.localization.Localization(__file__, 892, 12), '+=', subscript_call_result_175040, result_sub_175050)
    # Getting the type of 'tmp' (line 892)
    tmp_175052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'tmp')
    int_175053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 16), 'int')
    # Storing an element on a container (line 892)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 892, 12), tmp_175052, (int_175053, result_iadd_175051))
    
    
    # Assigning a Name to a Name (line 893):
    
    # Assigning a Name to a Name (line 893):
    # Getting the type of 'tmp' (line 893)
    tmp_175054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 16), 'tmp')
    # Assigning a type to the variable 'c' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 12), 'c', tmp_175054)
    # SSA join for if statement (line 880)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 894):
    
    # Assigning a Call to a Name (line 894):
    
    # Call to rollaxis(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'c' (line 894)
    c_175057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'c', False)
    int_175058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 23), 'int')
    # Getting the type of 'iaxis' (line 894)
    iaxis_175059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 26), 'iaxis', False)
    int_175060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 34), 'int')
    # Applying the binary operator '+' (line 894)
    result_add_175061 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 26), '+', iaxis_175059, int_175060)
    
    # Processing the call keyword arguments (line 894)
    kwargs_175062 = {}
    # Getting the type of 'np' (line 894)
    np_175055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 894)
    rollaxis_175056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 8), np_175055, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 894)
    rollaxis_call_result_175063 = invoke(stypy.reporting.localization.Localization(__file__, 894, 8), rollaxis_175056, *[c_175057, int_175058, result_add_175061], **kwargs_175062)
    
    # Assigning a type to the variable 'c' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'c', rollaxis_call_result_175063)
    # Getting the type of 'c' (line 895)
    c_175064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 895)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 4), 'stypy_return_type', c_175064)
    
    # ################# End of 'legint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legint' in the type store
    # Getting the type of 'stypy_return_type' (line 767)
    stypy_return_type_175065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legint'
    return stypy_return_type_175065

# Assigning a type to the variable 'legint' (line 767)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 0), 'legint', legint)

@norecursion
def legval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 898)
    True_175066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 24), 'True')
    defaults = [True_175066]
    # Create a new context for function 'legval'
    module_type_store = module_type_store.open_function_context('legval', 898, 0, False)
    
    # Passed parameters checking function
    legval.stypy_localization = localization
    legval.stypy_type_of_self = None
    legval.stypy_type_store = module_type_store
    legval.stypy_function_name = 'legval'
    legval.stypy_param_names_list = ['x', 'c', 'tensor']
    legval.stypy_varargs_param_name = None
    legval.stypy_kwargs_param_name = None
    legval.stypy_call_defaults = defaults
    legval.stypy_call_varargs = varargs
    legval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legval', ['x', 'c', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legval', localization, ['x', 'c', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legval(...)' code ##################

    str_175067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, (-1)), 'str', '\n    Evaluate a Legendre series at points x.\n\n    If `c` is of length `n + 1`, this function returns the value:\n\n    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, algebra_like\n        The shape of the return value is described above.\n\n    See Also\n    --------\n    legval2d, leggrid2d, legval3d, leggrid3d\n\n    Notes\n    -----\n    The evaluation uses Clenshaw recursion, aka synthetic division.\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a Call to a Name (line 960):
    
    # Assigning a Call to a Name (line 960):
    
    # Call to array(...): (line 960)
    # Processing the call arguments (line 960)
    # Getting the type of 'c' (line 960)
    c_175070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 17), 'c', False)
    # Processing the call keyword arguments (line 960)
    int_175071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 26), 'int')
    keyword_175072 = int_175071
    int_175073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 34), 'int')
    keyword_175074 = int_175073
    kwargs_175075 = {'copy': keyword_175074, 'ndmin': keyword_175072}
    # Getting the type of 'np' (line 960)
    np_175068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 960)
    array_175069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 8), np_175068, 'array')
    # Calling array(args, kwargs) (line 960)
    array_call_result_175076 = invoke(stypy.reporting.localization.Localization(__file__, 960, 8), array_175069, *[c_175070], **kwargs_175075)
    
    # Assigning a type to the variable 'c' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'c', array_call_result_175076)
    
    
    # Getting the type of 'c' (line 961)
    c_175077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 961)
    dtype_175078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 7), c_175077, 'dtype')
    # Obtaining the member 'char' of a type (line 961)
    char_175079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 7), dtype_175078, 'char')
    str_175080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 961)
    result_contains_175081 = python_operator(stypy.reporting.localization.Localization(__file__, 961, 7), 'in', char_175079, str_175080)
    
    # Testing the type of an if condition (line 961)
    if_condition_175082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 961, 4), result_contains_175081)
    # Assigning a type to the variable 'if_condition_175082' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'if_condition_175082', if_condition_175082)
    # SSA begins for if statement (line 961)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 962):
    
    # Assigning a Call to a Name (line 962):
    
    # Call to astype(...): (line 962)
    # Processing the call arguments (line 962)
    # Getting the type of 'np' (line 962)
    np_175085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 962)
    double_175086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 21), np_175085, 'double')
    # Processing the call keyword arguments (line 962)
    kwargs_175087 = {}
    # Getting the type of 'c' (line 962)
    c_175083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 962)
    astype_175084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 12), c_175083, 'astype')
    # Calling astype(args, kwargs) (line 962)
    astype_call_result_175088 = invoke(stypy.reporting.localization.Localization(__file__, 962, 12), astype_175084, *[double_175086], **kwargs_175087)
    
    # Assigning a type to the variable 'c' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 8), 'c', astype_call_result_175088)
    # SSA join for if statement (line 961)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'x' (line 963)
    x_175090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 18), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 963)
    tuple_175091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 963)
    # Adding element type (line 963)
    # Getting the type of 'tuple' (line 963)
    tuple_175092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 22), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 963, 22), tuple_175091, tuple_175092)
    # Adding element type (line 963)
    # Getting the type of 'list' (line 963)
    list_175093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 963, 22), tuple_175091, list_175093)
    
    # Processing the call keyword arguments (line 963)
    kwargs_175094 = {}
    # Getting the type of 'isinstance' (line 963)
    isinstance_175089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 963)
    isinstance_call_result_175095 = invoke(stypy.reporting.localization.Localization(__file__, 963, 7), isinstance_175089, *[x_175090, tuple_175091], **kwargs_175094)
    
    # Testing the type of an if condition (line 963)
    if_condition_175096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 963, 4), isinstance_call_result_175095)
    # Assigning a type to the variable 'if_condition_175096' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'if_condition_175096', if_condition_175096)
    # SSA begins for if statement (line 963)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 964):
    
    # Assigning a Call to a Name (line 964):
    
    # Call to asarray(...): (line 964)
    # Processing the call arguments (line 964)
    # Getting the type of 'x' (line 964)
    x_175099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 23), 'x', False)
    # Processing the call keyword arguments (line 964)
    kwargs_175100 = {}
    # Getting the type of 'np' (line 964)
    np_175097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 964)
    asarray_175098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 12), np_175097, 'asarray')
    # Calling asarray(args, kwargs) (line 964)
    asarray_call_result_175101 = invoke(stypy.reporting.localization.Localization(__file__, 964, 12), asarray_175098, *[x_175099], **kwargs_175100)
    
    # Assigning a type to the variable 'x' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'x', asarray_call_result_175101)
    # SSA join for if statement (line 963)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'x' (line 965)
    x_175103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 18), 'x', False)
    # Getting the type of 'np' (line 965)
    np_175104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 965)
    ndarray_175105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 21), np_175104, 'ndarray')
    # Processing the call keyword arguments (line 965)
    kwargs_175106 = {}
    # Getting the type of 'isinstance' (line 965)
    isinstance_175102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 965)
    isinstance_call_result_175107 = invoke(stypy.reporting.localization.Localization(__file__, 965, 7), isinstance_175102, *[x_175103, ndarray_175105], **kwargs_175106)
    
    # Getting the type of 'tensor' (line 965)
    tensor_175108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 37), 'tensor')
    # Applying the binary operator 'and' (line 965)
    result_and_keyword_175109 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 7), 'and', isinstance_call_result_175107, tensor_175108)
    
    # Testing the type of an if condition (line 965)
    if_condition_175110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 965, 4), result_and_keyword_175109)
    # Assigning a type to the variable 'if_condition_175110' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'if_condition_175110', if_condition_175110)
    # SSA begins for if statement (line 965)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 966):
    
    # Assigning a Call to a Name (line 966):
    
    # Call to reshape(...): (line 966)
    # Processing the call arguments (line 966)
    # Getting the type of 'c' (line 966)
    c_175113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 22), 'c', False)
    # Obtaining the member 'shape' of a type (line 966)
    shape_175114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 22), c_175113, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 966)
    tuple_175115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 966)
    # Adding element type (line 966)
    int_175116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 966, 33), tuple_175115, int_175116)
    
    # Getting the type of 'x' (line 966)
    x_175117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 37), 'x', False)
    # Obtaining the member 'ndim' of a type (line 966)
    ndim_175118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 37), x_175117, 'ndim')
    # Applying the binary operator '*' (line 966)
    result_mul_175119 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 32), '*', tuple_175115, ndim_175118)
    
    # Applying the binary operator '+' (line 966)
    result_add_175120 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 22), '+', shape_175114, result_mul_175119)
    
    # Processing the call keyword arguments (line 966)
    kwargs_175121 = {}
    # Getting the type of 'c' (line 966)
    c_175111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 966)
    reshape_175112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 12), c_175111, 'reshape')
    # Calling reshape(args, kwargs) (line 966)
    reshape_call_result_175122 = invoke(stypy.reporting.localization.Localization(__file__, 966, 12), reshape_175112, *[result_add_175120], **kwargs_175121)
    
    # Assigning a type to the variable 'c' (line 966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'c', reshape_call_result_175122)
    # SSA join for if statement (line 965)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'c' (line 968)
    c_175124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 11), 'c', False)
    # Processing the call keyword arguments (line 968)
    kwargs_175125 = {}
    # Getting the type of 'len' (line 968)
    len_175123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 7), 'len', False)
    # Calling len(args, kwargs) (line 968)
    len_call_result_175126 = invoke(stypy.reporting.localization.Localization(__file__, 968, 7), len_175123, *[c_175124], **kwargs_175125)
    
    int_175127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 17), 'int')
    # Applying the binary operator '==' (line 968)
    result_eq_175128 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 7), '==', len_call_result_175126, int_175127)
    
    # Testing the type of an if condition (line 968)
    if_condition_175129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 968, 4), result_eq_175128)
    # Assigning a type to the variable 'if_condition_175129' (line 968)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'if_condition_175129', if_condition_175129)
    # SSA begins for if statement (line 968)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 969):
    
    # Assigning a Subscript to a Name (line 969):
    
    # Obtaining the type of the subscript
    int_175130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 15), 'int')
    # Getting the type of 'c' (line 969)
    c_175131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 969)
    getitem___175132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 13), c_175131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 969)
    subscript_call_result_175133 = invoke(stypy.reporting.localization.Localization(__file__, 969, 13), getitem___175132, int_175130)
    
    # Assigning a type to the variable 'c0' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 8), 'c0', subscript_call_result_175133)
    
    # Assigning a Num to a Name (line 970):
    
    # Assigning a Num to a Name (line 970):
    int_175134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 13), 'int')
    # Assigning a type to the variable 'c1' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 8), 'c1', int_175134)
    # SSA branch for the else part of an if statement (line 968)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 971)
    # Processing the call arguments (line 971)
    # Getting the type of 'c' (line 971)
    c_175136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 13), 'c', False)
    # Processing the call keyword arguments (line 971)
    kwargs_175137 = {}
    # Getting the type of 'len' (line 971)
    len_175135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 9), 'len', False)
    # Calling len(args, kwargs) (line 971)
    len_call_result_175138 = invoke(stypy.reporting.localization.Localization(__file__, 971, 9), len_175135, *[c_175136], **kwargs_175137)
    
    int_175139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 19), 'int')
    # Applying the binary operator '==' (line 971)
    result_eq_175140 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 9), '==', len_call_result_175138, int_175139)
    
    # Testing the type of an if condition (line 971)
    if_condition_175141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 9), result_eq_175140)
    # Assigning a type to the variable 'if_condition_175141' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 9), 'if_condition_175141', if_condition_175141)
    # SSA begins for if statement (line 971)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 972):
    
    # Assigning a Subscript to a Name (line 972):
    
    # Obtaining the type of the subscript
    int_175142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 15), 'int')
    # Getting the type of 'c' (line 972)
    c_175143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 972)
    getitem___175144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 13), c_175143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 972)
    subscript_call_result_175145 = invoke(stypy.reporting.localization.Localization(__file__, 972, 13), getitem___175144, int_175142)
    
    # Assigning a type to the variable 'c0' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'c0', subscript_call_result_175145)
    
    # Assigning a Subscript to a Name (line 973):
    
    # Assigning a Subscript to a Name (line 973):
    
    # Obtaining the type of the subscript
    int_175146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 15), 'int')
    # Getting the type of 'c' (line 973)
    c_175147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 973)
    getitem___175148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 13), c_175147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 973)
    subscript_call_result_175149 = invoke(stypy.reporting.localization.Localization(__file__, 973, 13), getitem___175148, int_175146)
    
    # Assigning a type to the variable 'c1' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'c1', subscript_call_result_175149)
    # SSA branch for the else part of an if statement (line 971)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 975):
    
    # Assigning a Call to a Name (line 975):
    
    # Call to len(...): (line 975)
    # Processing the call arguments (line 975)
    # Getting the type of 'c' (line 975)
    c_175151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 17), 'c', False)
    # Processing the call keyword arguments (line 975)
    kwargs_175152 = {}
    # Getting the type of 'len' (line 975)
    len_175150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 13), 'len', False)
    # Calling len(args, kwargs) (line 975)
    len_call_result_175153 = invoke(stypy.reporting.localization.Localization(__file__, 975, 13), len_175150, *[c_175151], **kwargs_175152)
    
    # Assigning a type to the variable 'nd' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 8), 'nd', len_call_result_175153)
    
    # Assigning a Subscript to a Name (line 976):
    
    # Assigning a Subscript to a Name (line 976):
    
    # Obtaining the type of the subscript
    int_175154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 15), 'int')
    # Getting the type of 'c' (line 976)
    c_175155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 976)
    getitem___175156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 13), c_175155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 976)
    subscript_call_result_175157 = invoke(stypy.reporting.localization.Localization(__file__, 976, 13), getitem___175156, int_175154)
    
    # Assigning a type to the variable 'c0' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'c0', subscript_call_result_175157)
    
    # Assigning a Subscript to a Name (line 977):
    
    # Assigning a Subscript to a Name (line 977):
    
    # Obtaining the type of the subscript
    int_175158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 15), 'int')
    # Getting the type of 'c' (line 977)
    c_175159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 977)
    getitem___175160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 13), c_175159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 977)
    subscript_call_result_175161 = invoke(stypy.reporting.localization.Localization(__file__, 977, 13), getitem___175160, int_175158)
    
    # Assigning a type to the variable 'c1' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 8), 'c1', subscript_call_result_175161)
    
    
    # Call to range(...): (line 978)
    # Processing the call arguments (line 978)
    int_175163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 23), 'int')
    
    # Call to len(...): (line 978)
    # Processing the call arguments (line 978)
    # Getting the type of 'c' (line 978)
    c_175165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 30), 'c', False)
    # Processing the call keyword arguments (line 978)
    kwargs_175166 = {}
    # Getting the type of 'len' (line 978)
    len_175164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 26), 'len', False)
    # Calling len(args, kwargs) (line 978)
    len_call_result_175167 = invoke(stypy.reporting.localization.Localization(__file__, 978, 26), len_175164, *[c_175165], **kwargs_175166)
    
    int_175168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 35), 'int')
    # Applying the binary operator '+' (line 978)
    result_add_175169 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 26), '+', len_call_result_175167, int_175168)
    
    # Processing the call keyword arguments (line 978)
    kwargs_175170 = {}
    # Getting the type of 'range' (line 978)
    range_175162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 17), 'range', False)
    # Calling range(args, kwargs) (line 978)
    range_call_result_175171 = invoke(stypy.reporting.localization.Localization(__file__, 978, 17), range_175162, *[int_175163, result_add_175169], **kwargs_175170)
    
    # Testing the type of a for loop iterable (line 978)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 978, 8), range_call_result_175171)
    # Getting the type of the for loop variable (line 978)
    for_loop_var_175172 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 978, 8), range_call_result_175171)
    # Assigning a type to the variable 'i' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 8), 'i', for_loop_var_175172)
    # SSA begins for a for statement (line 978)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 979):
    
    # Assigning a Name to a Name (line 979):
    # Getting the type of 'c0' (line 979)
    c0_175173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 979)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 12), 'tmp', c0_175173)
    
    # Assigning a BinOp to a Name (line 980):
    
    # Assigning a BinOp to a Name (line 980):
    # Getting the type of 'nd' (line 980)
    nd_175174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 17), 'nd')
    int_175175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 22), 'int')
    # Applying the binary operator '-' (line 980)
    result_sub_175176 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 17), '-', nd_175174, int_175175)
    
    # Assigning a type to the variable 'nd' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 12), 'nd', result_sub_175176)
    
    # Assigning a BinOp to a Name (line 981):
    
    # Assigning a BinOp to a Name (line 981):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 981)
    i_175177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 20), 'i')
    # Applying the 'usub' unary operator (line 981)
    result___neg___175178 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 19), 'usub', i_175177)
    
    # Getting the type of 'c' (line 981)
    c_175179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 17), 'c')
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___175180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 17), c_175179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_175181 = invoke(stypy.reporting.localization.Localization(__file__, 981, 17), getitem___175180, result___neg___175178)
    
    # Getting the type of 'c1' (line 981)
    c1_175182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 26), 'c1')
    # Getting the type of 'nd' (line 981)
    nd_175183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 30), 'nd')
    int_175184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 35), 'int')
    # Applying the binary operator '-' (line 981)
    result_sub_175185 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 30), '-', nd_175183, int_175184)
    
    # Applying the binary operator '*' (line 981)
    result_mul_175186 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 26), '*', c1_175182, result_sub_175185)
    
    # Getting the type of 'nd' (line 981)
    nd_175187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 39), 'nd')
    # Applying the binary operator 'div' (line 981)
    result_div_175188 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 25), 'div', result_mul_175186, nd_175187)
    
    # Applying the binary operator '-' (line 981)
    result_sub_175189 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 17), '-', subscript_call_result_175181, result_div_175188)
    
    # Assigning a type to the variable 'c0' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 12), 'c0', result_sub_175189)
    
    # Assigning a BinOp to a Name (line 982):
    
    # Assigning a BinOp to a Name (line 982):
    # Getting the type of 'tmp' (line 982)
    tmp_175190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 17), 'tmp')
    # Getting the type of 'c1' (line 982)
    c1_175191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 24), 'c1')
    # Getting the type of 'x' (line 982)
    x_175192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 27), 'x')
    # Applying the binary operator '*' (line 982)
    result_mul_175193 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 24), '*', c1_175191, x_175192)
    
    int_175194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 30), 'int')
    # Getting the type of 'nd' (line 982)
    nd_175195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 32), 'nd')
    # Applying the binary operator '*' (line 982)
    result_mul_175196 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 30), '*', int_175194, nd_175195)
    
    int_175197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 37), 'int')
    # Applying the binary operator '-' (line 982)
    result_sub_175198 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 30), '-', result_mul_175196, int_175197)
    
    # Applying the binary operator '*' (line 982)
    result_mul_175199 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 28), '*', result_mul_175193, result_sub_175198)
    
    # Getting the type of 'nd' (line 982)
    nd_175200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 41), 'nd')
    # Applying the binary operator 'div' (line 982)
    result_div_175201 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 23), 'div', result_mul_175199, nd_175200)
    
    # Applying the binary operator '+' (line 982)
    result_add_175202 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 17), '+', tmp_175190, result_div_175201)
    
    # Assigning a type to the variable 'c1' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 12), 'c1', result_add_175202)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 971)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 968)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 983)
    c0_175203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 11), 'c0')
    # Getting the type of 'c1' (line 983)
    c1_175204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 16), 'c1')
    # Getting the type of 'x' (line 983)
    x_175205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 19), 'x')
    # Applying the binary operator '*' (line 983)
    result_mul_175206 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 16), '*', c1_175204, x_175205)
    
    # Applying the binary operator '+' (line 983)
    result_add_175207 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 11), '+', c0_175203, result_mul_175206)
    
    # Assigning a type to the variable 'stypy_return_type' (line 983)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 4), 'stypy_return_type', result_add_175207)
    
    # ################# End of 'legval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legval' in the type store
    # Getting the type of 'stypy_return_type' (line 898)
    stypy_return_type_175208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175208)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legval'
    return stypy_return_type_175208

# Assigning a type to the variable 'legval' (line 898)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 0), 'legval', legval)

@norecursion
def legval2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legval2d'
    module_type_store = module_type_store.open_function_context('legval2d', 986, 0, False)
    
    # Passed parameters checking function
    legval2d.stypy_localization = localization
    legval2d.stypy_type_of_self = None
    legval2d.stypy_type_store = module_type_store
    legval2d.stypy_function_name = 'legval2d'
    legval2d.stypy_param_names_list = ['x', 'y', 'c']
    legval2d.stypy_varargs_param_name = None
    legval2d.stypy_kwargs_param_name = None
    legval2d.stypy_call_defaults = defaults
    legval2d.stypy_call_varargs = varargs
    legval2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legval2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legval2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legval2d(...)' code ##################

    str_175209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, (-1)), 'str', "\n    Evaluate a 2-D Legendre series at points (x, y).\n\n    This function returns the values:\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` is a 1-D array a one is implicitly appended to its shape to make\n    it 2-D. The shape of the result will be c.shape[2:] + x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and if it isn't an ndarray it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in ``c[i,j]``. If `c` has\n        dimension greater than two the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional Legendre series at points formed\n        from pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    legval, leggrid2d, legval3d, leggrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1032)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1033):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1033)
    # Processing the call arguments (line 1033)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1033)
    tuple_175212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1033)
    # Adding element type (line 1033)
    # Getting the type of 'x' (line 1033)
    x_175213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1033, 25), tuple_175212, x_175213)
    # Adding element type (line 1033)
    # Getting the type of 'y' (line 1033)
    y_175214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 28), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1033, 25), tuple_175212, y_175214)
    
    # Processing the call keyword arguments (line 1033)
    int_175215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 37), 'int')
    keyword_175216 = int_175215
    kwargs_175217 = {'copy': keyword_175216}
    # Getting the type of 'np' (line 1033)
    np_175210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1033)
    array_175211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 15), np_175210, 'array')
    # Calling array(args, kwargs) (line 1033)
    array_call_result_175218 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 15), array_175211, *[tuple_175212], **kwargs_175217)
    
    # Assigning a type to the variable 'call_assignment_173632' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173632', array_call_result_175218)
    
    # Assigning a Call to a Name (line 1033):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 8), 'int')
    # Processing the call keyword arguments
    kwargs_175222 = {}
    # Getting the type of 'call_assignment_173632' (line 1033)
    call_assignment_173632_175219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173632', False)
    # Obtaining the member '__getitem__' of a type (line 1033)
    getitem___175220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 8), call_assignment_173632_175219, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175223 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175220, *[int_175221], **kwargs_175222)
    
    # Assigning a type to the variable 'call_assignment_173633' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173633', getitem___call_result_175223)
    
    # Assigning a Name to a Name (line 1033):
    # Getting the type of 'call_assignment_173633' (line 1033)
    call_assignment_173633_175224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173633')
    # Assigning a type to the variable 'x' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'x', call_assignment_173633_175224)
    
    # Assigning a Call to a Name (line 1033):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 8), 'int')
    # Processing the call keyword arguments
    kwargs_175228 = {}
    # Getting the type of 'call_assignment_173632' (line 1033)
    call_assignment_173632_175225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173632', False)
    # Obtaining the member '__getitem__' of a type (line 1033)
    getitem___175226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 8), call_assignment_173632_175225, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175229 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175226, *[int_175227], **kwargs_175228)
    
    # Assigning a type to the variable 'call_assignment_173634' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173634', getitem___call_result_175229)
    
    # Assigning a Name to a Name (line 1033):
    # Getting the type of 'call_assignment_173634' (line 1033)
    call_assignment_173634_175230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'call_assignment_173634')
    # Assigning a type to the variable 'y' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 11), 'y', call_assignment_173634_175230)
    # SSA branch for the except part of a try statement (line 1032)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1032)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1035)
    # Processing the call arguments (line 1035)
    str_175232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 25), 'str', 'x, y are incompatible')
    # Processing the call keyword arguments (line 1035)
    kwargs_175233 = {}
    # Getting the type of 'ValueError' (line 1035)
    ValueError_175231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1035)
    ValueError_call_result_175234 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 14), ValueError_175231, *[str_175232], **kwargs_175233)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1035, 8), ValueError_call_result_175234, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1032)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1037):
    
    # Assigning a Call to a Name (line 1037):
    
    # Call to legval(...): (line 1037)
    # Processing the call arguments (line 1037)
    # Getting the type of 'x' (line 1037)
    x_175236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 15), 'x', False)
    # Getting the type of 'c' (line 1037)
    c_175237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 18), 'c', False)
    # Processing the call keyword arguments (line 1037)
    kwargs_175238 = {}
    # Getting the type of 'legval' (line 1037)
    legval_175235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1037)
    legval_call_result_175239 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 8), legval_175235, *[x_175236, c_175237], **kwargs_175238)
    
    # Assigning a type to the variable 'c' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'c', legval_call_result_175239)
    
    # Assigning a Call to a Name (line 1038):
    
    # Assigning a Call to a Name (line 1038):
    
    # Call to legval(...): (line 1038)
    # Processing the call arguments (line 1038)
    # Getting the type of 'y' (line 1038)
    y_175241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 15), 'y', False)
    # Getting the type of 'c' (line 1038)
    c_175242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 18), 'c', False)
    # Processing the call keyword arguments (line 1038)
    # Getting the type of 'False' (line 1038)
    False_175243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 28), 'False', False)
    keyword_175244 = False_175243
    kwargs_175245 = {'tensor': keyword_175244}
    # Getting the type of 'legval' (line 1038)
    legval_175240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1038)
    legval_call_result_175246 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 8), legval_175240, *[y_175241, c_175242], **kwargs_175245)
    
    # Assigning a type to the variable 'c' (line 1038)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 4), 'c', legval_call_result_175246)
    # Getting the type of 'c' (line 1039)
    c_175247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 4), 'stypy_return_type', c_175247)
    
    # ################# End of 'legval2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legval2d' in the type store
    # Getting the type of 'stypy_return_type' (line 986)
    stypy_return_type_175248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175248)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legval2d'
    return stypy_return_type_175248

# Assigning a type to the variable 'legval2d' (line 986)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 0), 'legval2d', legval2d)

@norecursion
def leggrid2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'leggrid2d'
    module_type_store = module_type_store.open_function_context('leggrid2d', 1042, 0, False)
    
    # Passed parameters checking function
    leggrid2d.stypy_localization = localization
    leggrid2d.stypy_type_of_self = None
    leggrid2d.stypy_type_store = module_type_store
    leggrid2d.stypy_function_name = 'leggrid2d'
    leggrid2d.stypy_param_names_list = ['x', 'y', 'c']
    leggrid2d.stypy_varargs_param_name = None
    leggrid2d.stypy_kwargs_param_name = None
    leggrid2d.stypy_call_defaults = defaults
    leggrid2d.stypy_call_varargs = varargs
    leggrid2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'leggrid2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'leggrid2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'leggrid2d(...)' code ##################

    str_175249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, (-1)), 'str', "\n    Evaluate a 2-D Legendre series on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape + y.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional Chebyshev series at points in the\n        Cartesian product of `x` and `y`.\n\n    See Also\n    --------\n    legval, legval2d, legval3d, leggrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1092):
    
    # Assigning a Call to a Name (line 1092):
    
    # Call to legval(...): (line 1092)
    # Processing the call arguments (line 1092)
    # Getting the type of 'x' (line 1092)
    x_175251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 15), 'x', False)
    # Getting the type of 'c' (line 1092)
    c_175252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 18), 'c', False)
    # Processing the call keyword arguments (line 1092)
    kwargs_175253 = {}
    # Getting the type of 'legval' (line 1092)
    legval_175250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1092)
    legval_call_result_175254 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 8), legval_175250, *[x_175251, c_175252], **kwargs_175253)
    
    # Assigning a type to the variable 'c' (line 1092)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 4), 'c', legval_call_result_175254)
    
    # Assigning a Call to a Name (line 1093):
    
    # Assigning a Call to a Name (line 1093):
    
    # Call to legval(...): (line 1093)
    # Processing the call arguments (line 1093)
    # Getting the type of 'y' (line 1093)
    y_175256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 15), 'y', False)
    # Getting the type of 'c' (line 1093)
    c_175257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 18), 'c', False)
    # Processing the call keyword arguments (line 1093)
    kwargs_175258 = {}
    # Getting the type of 'legval' (line 1093)
    legval_175255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1093)
    legval_call_result_175259 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 8), legval_175255, *[y_175256, c_175257], **kwargs_175258)
    
    # Assigning a type to the variable 'c' (line 1093)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 4), 'c', legval_call_result_175259)
    # Getting the type of 'c' (line 1094)
    c_175260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 4), 'stypy_return_type', c_175260)
    
    # ################# End of 'leggrid2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'leggrid2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1042)
    stypy_return_type_175261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'leggrid2d'
    return stypy_return_type_175261

# Assigning a type to the variable 'leggrid2d' (line 1042)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 0), 'leggrid2d', leggrid2d)

@norecursion
def legval3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legval3d'
    module_type_store = module_type_store.open_function_context('legval3d', 1097, 0, False)
    
    # Passed parameters checking function
    legval3d.stypy_localization = localization
    legval3d.stypy_type_of_self = None
    legval3d.stypy_type_store = module_type_store
    legval3d.stypy_function_name = 'legval3d'
    legval3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    legval3d.stypy_varargs_param_name = None
    legval3d.stypy_kwargs_param_name = None
    legval3d.stypy_call_defaults = defaults
    legval3d.stypy_call_varargs = varargs
    legval3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legval3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legval3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legval3d(...)' code ##################

    str_175262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, (-1)), 'str', "\n    Evaluate a 3-D Legendre series at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimensional polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    legval, legval2d, leggrid2d, leggrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1146):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1146)
    # Processing the call arguments (line 1146)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1146)
    tuple_175265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1146)
    # Adding element type (line 1146)
    # Getting the type of 'x' (line 1146)
    x_175266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 28), tuple_175265, x_175266)
    # Adding element type (line 1146)
    # Getting the type of 'y' (line 1146)
    y_175267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 28), tuple_175265, y_175267)
    # Adding element type (line 1146)
    # Getting the type of 'z' (line 1146)
    z_175268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 34), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1146, 28), tuple_175265, z_175268)
    
    # Processing the call keyword arguments (line 1146)
    int_175269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 43), 'int')
    keyword_175270 = int_175269
    kwargs_175271 = {'copy': keyword_175270}
    # Getting the type of 'np' (line 1146)
    np_175263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 1146)
    array_175264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 18), np_175263, 'array')
    # Calling array(args, kwargs) (line 1146)
    array_call_result_175272 = invoke(stypy.reporting.localization.Localization(__file__, 1146, 18), array_175264, *[tuple_175265], **kwargs_175271)
    
    # Assigning a type to the variable 'call_assignment_173635' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173635', array_call_result_175272)
    
    # Assigning a Call to a Name (line 1146):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 8), 'int')
    # Processing the call keyword arguments
    kwargs_175276 = {}
    # Getting the type of 'call_assignment_173635' (line 1146)
    call_assignment_173635_175273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173635', False)
    # Obtaining the member '__getitem__' of a type (line 1146)
    getitem___175274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 8), call_assignment_173635_175273, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175277 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175274, *[int_175275], **kwargs_175276)
    
    # Assigning a type to the variable 'call_assignment_173636' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173636', getitem___call_result_175277)
    
    # Assigning a Name to a Name (line 1146):
    # Getting the type of 'call_assignment_173636' (line 1146)
    call_assignment_173636_175278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173636')
    # Assigning a type to the variable 'x' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'x', call_assignment_173636_175278)
    
    # Assigning a Call to a Name (line 1146):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 8), 'int')
    # Processing the call keyword arguments
    kwargs_175282 = {}
    # Getting the type of 'call_assignment_173635' (line 1146)
    call_assignment_173635_175279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173635', False)
    # Obtaining the member '__getitem__' of a type (line 1146)
    getitem___175280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 8), call_assignment_173635_175279, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175283 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175280, *[int_175281], **kwargs_175282)
    
    # Assigning a type to the variable 'call_assignment_173637' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173637', getitem___call_result_175283)
    
    # Assigning a Name to a Name (line 1146):
    # Getting the type of 'call_assignment_173637' (line 1146)
    call_assignment_173637_175284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173637')
    # Assigning a type to the variable 'y' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 11), 'y', call_assignment_173637_175284)
    
    # Assigning a Call to a Name (line 1146):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 8), 'int')
    # Processing the call keyword arguments
    kwargs_175288 = {}
    # Getting the type of 'call_assignment_173635' (line 1146)
    call_assignment_173635_175285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173635', False)
    # Obtaining the member '__getitem__' of a type (line 1146)
    getitem___175286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 8), call_assignment_173635_175285, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175289 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175286, *[int_175287], **kwargs_175288)
    
    # Assigning a type to the variable 'call_assignment_173638' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173638', getitem___call_result_175289)
    
    # Assigning a Name to a Name (line 1146):
    # Getting the type of 'call_assignment_173638' (line 1146)
    call_assignment_173638_175290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'call_assignment_173638')
    # Assigning a type to the variable 'z' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 14), 'z', call_assignment_173638_175290)
    # SSA branch for the except part of a try statement (line 1145)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1145)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1148)
    # Processing the call arguments (line 1148)
    str_175292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 25), 'str', 'x, y, z are incompatible')
    # Processing the call keyword arguments (line 1148)
    kwargs_175293 = {}
    # Getting the type of 'ValueError' (line 1148)
    ValueError_175291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1148)
    ValueError_call_result_175294 = invoke(stypy.reporting.localization.Localization(__file__, 1148, 14), ValueError_175291, *[str_175292], **kwargs_175293)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1148, 8), ValueError_call_result_175294, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1150):
    
    # Assigning a Call to a Name (line 1150):
    
    # Call to legval(...): (line 1150)
    # Processing the call arguments (line 1150)
    # Getting the type of 'x' (line 1150)
    x_175296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 15), 'x', False)
    # Getting the type of 'c' (line 1150)
    c_175297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 18), 'c', False)
    # Processing the call keyword arguments (line 1150)
    kwargs_175298 = {}
    # Getting the type of 'legval' (line 1150)
    legval_175295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1150)
    legval_call_result_175299 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 8), legval_175295, *[x_175296, c_175297], **kwargs_175298)
    
    # Assigning a type to the variable 'c' (line 1150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 4), 'c', legval_call_result_175299)
    
    # Assigning a Call to a Name (line 1151):
    
    # Assigning a Call to a Name (line 1151):
    
    # Call to legval(...): (line 1151)
    # Processing the call arguments (line 1151)
    # Getting the type of 'y' (line 1151)
    y_175301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 15), 'y', False)
    # Getting the type of 'c' (line 1151)
    c_175302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 18), 'c', False)
    # Processing the call keyword arguments (line 1151)
    # Getting the type of 'False' (line 1151)
    False_175303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 28), 'False', False)
    keyword_175304 = False_175303
    kwargs_175305 = {'tensor': keyword_175304}
    # Getting the type of 'legval' (line 1151)
    legval_175300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1151)
    legval_call_result_175306 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 8), legval_175300, *[y_175301, c_175302], **kwargs_175305)
    
    # Assigning a type to the variable 'c' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 4), 'c', legval_call_result_175306)
    
    # Assigning a Call to a Name (line 1152):
    
    # Assigning a Call to a Name (line 1152):
    
    # Call to legval(...): (line 1152)
    # Processing the call arguments (line 1152)
    # Getting the type of 'z' (line 1152)
    z_175308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 15), 'z', False)
    # Getting the type of 'c' (line 1152)
    c_175309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 18), 'c', False)
    # Processing the call keyword arguments (line 1152)
    # Getting the type of 'False' (line 1152)
    False_175310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 28), 'False', False)
    keyword_175311 = False_175310
    kwargs_175312 = {'tensor': keyword_175311}
    # Getting the type of 'legval' (line 1152)
    legval_175307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1152)
    legval_call_result_175313 = invoke(stypy.reporting.localization.Localization(__file__, 1152, 8), legval_175307, *[z_175308, c_175309], **kwargs_175312)
    
    # Assigning a type to the variable 'c' (line 1152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 4), 'c', legval_call_result_175313)
    # Getting the type of 'c' (line 1153)
    c_175314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 4), 'stypy_return_type', c_175314)
    
    # ################# End of 'legval3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legval3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1097)
    stypy_return_type_175315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175315)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legval3d'
    return stypy_return_type_175315

# Assigning a type to the variable 'legval3d' (line 1097)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 0), 'legval3d', legval3d)

@norecursion
def leggrid3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'leggrid3d'
    module_type_store = module_type_store.open_function_context('leggrid3d', 1156, 0, False)
    
    # Passed parameters checking function
    leggrid3d.stypy_localization = localization
    leggrid3d.stypy_type_of_self = None
    leggrid3d.stypy_type_store = module_type_store
    leggrid3d.stypy_function_name = 'leggrid3d'
    leggrid3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    leggrid3d.stypy_varargs_param_name = None
    leggrid3d.stypy_kwargs_param_name = None
    leggrid3d.stypy_call_defaults = defaults
    leggrid3d.stypy_call_varargs = varargs
    leggrid3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'leggrid3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'leggrid3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'leggrid3d(...)' code ##################

    str_175316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1208, (-1)), 'str', "\n    Evaluate a 3-D Legendre series on the Cartesian product of x, y, and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    legval, legval2d, leggrid2d, legval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1209):
    
    # Assigning a Call to a Name (line 1209):
    
    # Call to legval(...): (line 1209)
    # Processing the call arguments (line 1209)
    # Getting the type of 'x' (line 1209)
    x_175318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 15), 'x', False)
    # Getting the type of 'c' (line 1209)
    c_175319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 18), 'c', False)
    # Processing the call keyword arguments (line 1209)
    kwargs_175320 = {}
    # Getting the type of 'legval' (line 1209)
    legval_175317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1209)
    legval_call_result_175321 = invoke(stypy.reporting.localization.Localization(__file__, 1209, 8), legval_175317, *[x_175318, c_175319], **kwargs_175320)
    
    # Assigning a type to the variable 'c' (line 1209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 4), 'c', legval_call_result_175321)
    
    # Assigning a Call to a Name (line 1210):
    
    # Assigning a Call to a Name (line 1210):
    
    # Call to legval(...): (line 1210)
    # Processing the call arguments (line 1210)
    # Getting the type of 'y' (line 1210)
    y_175323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 15), 'y', False)
    # Getting the type of 'c' (line 1210)
    c_175324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 18), 'c', False)
    # Processing the call keyword arguments (line 1210)
    kwargs_175325 = {}
    # Getting the type of 'legval' (line 1210)
    legval_175322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1210)
    legval_call_result_175326 = invoke(stypy.reporting.localization.Localization(__file__, 1210, 8), legval_175322, *[y_175323, c_175324], **kwargs_175325)
    
    # Assigning a type to the variable 'c' (line 1210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1210, 4), 'c', legval_call_result_175326)
    
    # Assigning a Call to a Name (line 1211):
    
    # Assigning a Call to a Name (line 1211):
    
    # Call to legval(...): (line 1211)
    # Processing the call arguments (line 1211)
    # Getting the type of 'z' (line 1211)
    z_175328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 15), 'z', False)
    # Getting the type of 'c' (line 1211)
    c_175329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 18), 'c', False)
    # Processing the call keyword arguments (line 1211)
    kwargs_175330 = {}
    # Getting the type of 'legval' (line 1211)
    legval_175327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 8), 'legval', False)
    # Calling legval(args, kwargs) (line 1211)
    legval_call_result_175331 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 8), legval_175327, *[z_175328, c_175329], **kwargs_175330)
    
    # Assigning a type to the variable 'c' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 4), 'c', legval_call_result_175331)
    # Getting the type of 'c' (line 1212)
    c_175332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1212, 4), 'stypy_return_type', c_175332)
    
    # ################# End of 'leggrid3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'leggrid3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1156)
    stypy_return_type_175333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175333)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'leggrid3d'
    return stypy_return_type_175333

# Assigning a type to the variable 'leggrid3d' (line 1156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 0), 'leggrid3d', leggrid3d)

@norecursion
def legvander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legvander'
    module_type_store = module_type_store.open_function_context('legvander', 1215, 0, False)
    
    # Passed parameters checking function
    legvander.stypy_localization = localization
    legvander.stypy_type_of_self = None
    legvander.stypy_type_store = module_type_store
    legvander.stypy_function_name = 'legvander'
    legvander.stypy_param_names_list = ['x', 'deg']
    legvander.stypy_varargs_param_name = None
    legvander.stypy_kwargs_param_name = None
    legvander.stypy_call_defaults = defaults
    legvander.stypy_call_varargs = varargs
    legvander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legvander', ['x', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legvander', localization, ['x', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legvander(...)' code ##################

    str_175334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1249, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degree.\n\n    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points\n    `x`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = L_i(x)\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the degree of the Legendre polynomial.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    array ``V = legvander(x, n)``, then ``np.dot(V, c)`` and\n    ``legval(x, c)`` are the same up to roundoff. This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of Legendre series of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray\n        The pseudo-Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where The last index is the degree of the\n        corresponding Legendre polynomial.  The dtype will be the same as\n        the converted `x`.\n\n    ')
    
    # Assigning a Call to a Name (line 1250):
    
    # Assigning a Call to a Name (line 1250):
    
    # Call to int(...): (line 1250)
    # Processing the call arguments (line 1250)
    # Getting the type of 'deg' (line 1250)
    deg_175336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 15), 'deg', False)
    # Processing the call keyword arguments (line 1250)
    kwargs_175337 = {}
    # Getting the type of 'int' (line 1250)
    int_175335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 11), 'int', False)
    # Calling int(args, kwargs) (line 1250)
    int_call_result_175338 = invoke(stypy.reporting.localization.Localization(__file__, 1250, 11), int_175335, *[deg_175336], **kwargs_175337)
    
    # Assigning a type to the variable 'ideg' (line 1250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1250, 4), 'ideg', int_call_result_175338)
    
    
    # Getting the type of 'ideg' (line 1251)
    ideg_175339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 7), 'ideg')
    # Getting the type of 'deg' (line 1251)
    deg_175340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 15), 'deg')
    # Applying the binary operator '!=' (line 1251)
    result_ne_175341 = python_operator(stypy.reporting.localization.Localization(__file__, 1251, 7), '!=', ideg_175339, deg_175340)
    
    # Testing the type of an if condition (line 1251)
    if_condition_175342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1251, 4), result_ne_175341)
    # Assigning a type to the variable 'if_condition_175342' (line 1251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1251, 4), 'if_condition_175342', if_condition_175342)
    # SSA begins for if statement (line 1251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1252)
    # Processing the call arguments (line 1252)
    str_175344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1252, 25), 'str', 'deg must be integer')
    # Processing the call keyword arguments (line 1252)
    kwargs_175345 = {}
    # Getting the type of 'ValueError' (line 1252)
    ValueError_175343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1252, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1252)
    ValueError_call_result_175346 = invoke(stypy.reporting.localization.Localization(__file__, 1252, 14), ValueError_175343, *[str_175344], **kwargs_175345)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1252, 8), ValueError_call_result_175346, 'raise parameter', BaseException)
    # SSA join for if statement (line 1251)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ideg' (line 1253)
    ideg_175347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 7), 'ideg')
    int_175348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1253, 14), 'int')
    # Applying the binary operator '<' (line 1253)
    result_lt_175349 = python_operator(stypy.reporting.localization.Localization(__file__, 1253, 7), '<', ideg_175347, int_175348)
    
    # Testing the type of an if condition (line 1253)
    if_condition_175350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1253, 4), result_lt_175349)
    # Assigning a type to the variable 'if_condition_175350' (line 1253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1253, 4), 'if_condition_175350', if_condition_175350)
    # SSA begins for if statement (line 1253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1254)
    # Processing the call arguments (line 1254)
    str_175352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1254, 25), 'str', 'deg must be non-negative')
    # Processing the call keyword arguments (line 1254)
    kwargs_175353 = {}
    # Getting the type of 'ValueError' (line 1254)
    ValueError_175351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1254, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1254)
    ValueError_call_result_175354 = invoke(stypy.reporting.localization.Localization(__file__, 1254, 14), ValueError_175351, *[str_175352], **kwargs_175353)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1254, 8), ValueError_call_result_175354, 'raise parameter', BaseException)
    # SSA join for if statement (line 1253)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1256):
    
    # Assigning a BinOp to a Name (line 1256):
    
    # Call to array(...): (line 1256)
    # Processing the call arguments (line 1256)
    # Getting the type of 'x' (line 1256)
    x_175357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 17), 'x', False)
    # Processing the call keyword arguments (line 1256)
    int_175358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1256, 25), 'int')
    keyword_175359 = int_175358
    int_175360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1256, 34), 'int')
    keyword_175361 = int_175360
    kwargs_175362 = {'copy': keyword_175359, 'ndmin': keyword_175361}
    # Getting the type of 'np' (line 1256)
    np_175355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1256)
    array_175356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1256, 8), np_175355, 'array')
    # Calling array(args, kwargs) (line 1256)
    array_call_result_175363 = invoke(stypy.reporting.localization.Localization(__file__, 1256, 8), array_175356, *[x_175357], **kwargs_175362)
    
    float_175364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1256, 39), 'float')
    # Applying the binary operator '+' (line 1256)
    result_add_175365 = python_operator(stypy.reporting.localization.Localization(__file__, 1256, 8), '+', array_call_result_175363, float_175364)
    
    # Assigning a type to the variable 'x' (line 1256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1256, 4), 'x', result_add_175365)
    
    # Assigning a BinOp to a Name (line 1257):
    
    # Assigning a BinOp to a Name (line 1257):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1257)
    tuple_175366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1257)
    # Adding element type (line 1257)
    # Getting the type of 'ideg' (line 1257)
    ideg_175367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 12), 'ideg')
    int_175368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 19), 'int')
    # Applying the binary operator '+' (line 1257)
    result_add_175369 = python_operator(stypy.reporting.localization.Localization(__file__, 1257, 12), '+', ideg_175367, int_175368)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1257, 12), tuple_175366, result_add_175369)
    
    # Getting the type of 'x' (line 1257)
    x_175370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 25), 'x')
    # Obtaining the member 'shape' of a type (line 1257)
    shape_175371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 25), x_175370, 'shape')
    # Applying the binary operator '+' (line 1257)
    result_add_175372 = python_operator(stypy.reporting.localization.Localization(__file__, 1257, 11), '+', tuple_175366, shape_175371)
    
    # Assigning a type to the variable 'dims' (line 1257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1257, 4), 'dims', result_add_175372)
    
    # Assigning a Attribute to a Name (line 1258):
    
    # Assigning a Attribute to a Name (line 1258):
    # Getting the type of 'x' (line 1258)
    x_175373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 1258)
    dtype_175374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 11), x_175373, 'dtype')
    # Assigning a type to the variable 'dtyp' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 4), 'dtyp', dtype_175374)
    
    # Assigning a Call to a Name (line 1259):
    
    # Assigning a Call to a Name (line 1259):
    
    # Call to empty(...): (line 1259)
    # Processing the call arguments (line 1259)
    # Getting the type of 'dims' (line 1259)
    dims_175377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 17), 'dims', False)
    # Processing the call keyword arguments (line 1259)
    # Getting the type of 'dtyp' (line 1259)
    dtyp_175378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 29), 'dtyp', False)
    keyword_175379 = dtyp_175378
    kwargs_175380 = {'dtype': keyword_175379}
    # Getting the type of 'np' (line 1259)
    np_175375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1259)
    empty_175376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1259, 8), np_175375, 'empty')
    # Calling empty(args, kwargs) (line 1259)
    empty_call_result_175381 = invoke(stypy.reporting.localization.Localization(__file__, 1259, 8), empty_175376, *[dims_175377], **kwargs_175380)
    
    # Assigning a type to the variable 'v' (line 1259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1259, 4), 'v', empty_call_result_175381)
    
    # Assigning a BinOp to a Subscript (line 1262):
    
    # Assigning a BinOp to a Subscript (line 1262):
    # Getting the type of 'x' (line 1262)
    x_175382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 11), 'x')
    int_175383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1262, 13), 'int')
    # Applying the binary operator '*' (line 1262)
    result_mul_175384 = python_operator(stypy.reporting.localization.Localization(__file__, 1262, 11), '*', x_175382, int_175383)
    
    int_175385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1262, 17), 'int')
    # Applying the binary operator '+' (line 1262)
    result_add_175386 = python_operator(stypy.reporting.localization.Localization(__file__, 1262, 11), '+', result_mul_175384, int_175385)
    
    # Getting the type of 'v' (line 1262)
    v_175387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 4), 'v')
    int_175388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1262, 6), 'int')
    # Storing an element on a container (line 1262)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1262, 4), v_175387, (int_175388, result_add_175386))
    
    
    # Getting the type of 'ideg' (line 1263)
    ideg_175389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 7), 'ideg')
    int_175390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1263, 14), 'int')
    # Applying the binary operator '>' (line 1263)
    result_gt_175391 = python_operator(stypy.reporting.localization.Localization(__file__, 1263, 7), '>', ideg_175389, int_175390)
    
    # Testing the type of an if condition (line 1263)
    if_condition_175392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1263, 4), result_gt_175391)
    # Assigning a type to the variable 'if_condition_175392' (line 1263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1263, 4), 'if_condition_175392', if_condition_175392)
    # SSA begins for if statement (line 1263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 1264):
    
    # Assigning a Name to a Subscript (line 1264):
    # Getting the type of 'x' (line 1264)
    x_175393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 15), 'x')
    # Getting the type of 'v' (line 1264)
    v_175394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 8), 'v')
    int_175395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1264, 10), 'int')
    # Storing an element on a container (line 1264)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1264, 8), v_175394, (int_175395, x_175393))
    
    
    # Call to range(...): (line 1265)
    # Processing the call arguments (line 1265)
    int_175397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 23), 'int')
    # Getting the type of 'ideg' (line 1265)
    ideg_175398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 26), 'ideg', False)
    int_175399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 33), 'int')
    # Applying the binary operator '+' (line 1265)
    result_add_175400 = python_operator(stypy.reporting.localization.Localization(__file__, 1265, 26), '+', ideg_175398, int_175399)
    
    # Processing the call keyword arguments (line 1265)
    kwargs_175401 = {}
    # Getting the type of 'range' (line 1265)
    range_175396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 17), 'range', False)
    # Calling range(args, kwargs) (line 1265)
    range_call_result_175402 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 17), range_175396, *[int_175397, result_add_175400], **kwargs_175401)
    
    # Testing the type of a for loop iterable (line 1265)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1265, 8), range_call_result_175402)
    # Getting the type of the for loop variable (line 1265)
    for_loop_var_175403 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1265, 8), range_call_result_175402)
    # Assigning a type to the variable 'i' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 8), 'i', for_loop_var_175403)
    # SSA begins for a for statement (line 1265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 1266):
    
    # Assigning a BinOp to a Subscript (line 1266):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1266)
    i_175404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 22), 'i')
    int_175405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 24), 'int')
    # Applying the binary operator '-' (line 1266)
    result_sub_175406 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 22), '-', i_175404, int_175405)
    
    # Getting the type of 'v' (line 1266)
    v_175407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 20), 'v')
    # Obtaining the member '__getitem__' of a type (line 1266)
    getitem___175408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 20), v_175407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1266)
    subscript_call_result_175409 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 20), getitem___175408, result_sub_175406)
    
    # Getting the type of 'x' (line 1266)
    x_175410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 27), 'x')
    # Applying the binary operator '*' (line 1266)
    result_mul_175411 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 20), '*', subscript_call_result_175409, x_175410)
    
    int_175412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 30), 'int')
    # Getting the type of 'i' (line 1266)
    i_175413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 32), 'i')
    # Applying the binary operator '*' (line 1266)
    result_mul_175414 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 30), '*', int_175412, i_175413)
    
    int_175415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 36), 'int')
    # Applying the binary operator '-' (line 1266)
    result_sub_175416 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 30), '-', result_mul_175414, int_175415)
    
    # Applying the binary operator '*' (line 1266)
    result_mul_175417 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 28), '*', result_mul_175411, result_sub_175416)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1266)
    i_175418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 43), 'i')
    int_175419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 45), 'int')
    # Applying the binary operator '-' (line 1266)
    result_sub_175420 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 43), '-', i_175418, int_175419)
    
    # Getting the type of 'v' (line 1266)
    v_175421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 41), 'v')
    # Obtaining the member '__getitem__' of a type (line 1266)
    getitem___175422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 41), v_175421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1266)
    subscript_call_result_175423 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 41), getitem___175422, result_sub_175420)
    
    # Getting the type of 'i' (line 1266)
    i_175424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 49), 'i')
    int_175425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 53), 'int')
    # Applying the binary operator '-' (line 1266)
    result_sub_175426 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 49), '-', i_175424, int_175425)
    
    # Applying the binary operator '*' (line 1266)
    result_mul_175427 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 41), '*', subscript_call_result_175423, result_sub_175426)
    
    # Applying the binary operator '-' (line 1266)
    result_sub_175428 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 20), '-', result_mul_175417, result_mul_175427)
    
    # Getting the type of 'i' (line 1266)
    i_175429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 57), 'i')
    # Applying the binary operator 'div' (line 1266)
    result_div_175430 = python_operator(stypy.reporting.localization.Localization(__file__, 1266, 19), 'div', result_sub_175428, i_175429)
    
    # Getting the type of 'v' (line 1266)
    v_175431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 12), 'v')
    # Getting the type of 'i' (line 1266)
    i_175432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 14), 'i')
    # Storing an element on a container (line 1266)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1266, 12), v_175431, (i_175432, result_div_175430))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rollaxis(...): (line 1267)
    # Processing the call arguments (line 1267)
    # Getting the type of 'v' (line 1267)
    v_175435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 23), 'v', False)
    int_175436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1267, 26), 'int')
    # Getting the type of 'v' (line 1267)
    v_175437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 29), 'v', False)
    # Obtaining the member 'ndim' of a type (line 1267)
    ndim_175438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 29), v_175437, 'ndim')
    # Processing the call keyword arguments (line 1267)
    kwargs_175439 = {}
    # Getting the type of 'np' (line 1267)
    np_175433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 11), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1267)
    rollaxis_175434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 11), np_175433, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1267)
    rollaxis_call_result_175440 = invoke(stypy.reporting.localization.Localization(__file__, 1267, 11), rollaxis_175434, *[v_175435, int_175436, ndim_175438], **kwargs_175439)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1267, 4), 'stypy_return_type', rollaxis_call_result_175440)
    
    # ################# End of 'legvander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legvander' in the type store
    # Getting the type of 'stypy_return_type' (line 1215)
    stypy_return_type_175441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legvander'
    return stypy_return_type_175441

# Assigning a type to the variable 'legvander' (line 1215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 0), 'legvander', legvander)

@norecursion
def legvander2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legvander2d'
    module_type_store = module_type_store.open_function_context('legvander2d', 1270, 0, False)
    
    # Passed parameters checking function
    legvander2d.stypy_localization = localization
    legvander2d.stypy_type_of_self = None
    legvander2d.stypy_type_store = module_type_store
    legvander2d.stypy_function_name = 'legvander2d'
    legvander2d.stypy_param_names_list = ['x', 'y', 'deg']
    legvander2d.stypy_varargs_param_name = None
    legvander2d.stypy_kwargs_param_name = None
    legvander2d.stypy_call_defaults = defaults
    legvander2d.stypy_call_varargs = varargs
    legvander2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legvander2d', ['x', 'y', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legvander2d', localization, ['x', 'y', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legvander2d(...)' code ##################

    str_175442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1319, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., deg[1]*i + j] = L_i(x) * L_j(y),\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the degrees of\n    the Legendre polynomials.\n\n    If ``V = legvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``legval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D Legendre\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to\n        1-D arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    legvander, legvander3d. legval2d, legval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1320):
    
    # Assigning a ListComp to a Name (line 1320):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1320)
    deg_175447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 28), 'deg')
    comprehension_175448 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 12), deg_175447)
    # Assigning a type to the variable 'd' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 12), 'd', comprehension_175448)
    
    # Call to int(...): (line 1320)
    # Processing the call arguments (line 1320)
    # Getting the type of 'd' (line 1320)
    d_175444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 16), 'd', False)
    # Processing the call keyword arguments (line 1320)
    kwargs_175445 = {}
    # Getting the type of 'int' (line 1320)
    int_175443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 12), 'int', False)
    # Calling int(args, kwargs) (line 1320)
    int_call_result_175446 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 12), int_175443, *[d_175444], **kwargs_175445)
    
    list_175449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 12), list_175449, int_call_result_175446)
    # Assigning a type to the variable 'ideg' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'ideg', list_175449)
    
    # Assigning a ListComp to a Name (line 1321):
    
    # Assigning a ListComp to a Name (line 1321):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1321)
    # Processing the call arguments (line 1321)
    # Getting the type of 'ideg' (line 1321)
    ideg_175458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1321)
    deg_175459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 59), 'deg', False)
    # Processing the call keyword arguments (line 1321)
    kwargs_175460 = {}
    # Getting the type of 'zip' (line 1321)
    zip_175457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1321)
    zip_call_result_175461 = invoke(stypy.reporting.localization.Localization(__file__, 1321, 49), zip_175457, *[ideg_175458, deg_175459], **kwargs_175460)
    
    comprehension_175462 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1321, 16), zip_call_result_175461)
    # Assigning a type to the variable 'id' (line 1321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1321, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1321, 16), comprehension_175462))
    # Assigning a type to the variable 'd' (line 1321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1321, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1321, 16), comprehension_175462))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1321)
    id_175450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 16), 'id')
    # Getting the type of 'd' (line 1321)
    d_175451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 22), 'd')
    # Applying the binary operator '==' (line 1321)
    result_eq_175452 = python_operator(stypy.reporting.localization.Localization(__file__, 1321, 16), '==', id_175450, d_175451)
    
    
    # Getting the type of 'id' (line 1321)
    id_175453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 28), 'id')
    int_175454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1321, 34), 'int')
    # Applying the binary operator '>=' (line 1321)
    result_ge_175455 = python_operator(stypy.reporting.localization.Localization(__file__, 1321, 28), '>=', id_175453, int_175454)
    
    # Applying the binary operator 'and' (line 1321)
    result_and_keyword_175456 = python_operator(stypy.reporting.localization.Localization(__file__, 1321, 16), 'and', result_eq_175452, result_ge_175455)
    
    list_175463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1321, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1321, 16), list_175463, result_and_keyword_175456)
    # Assigning a type to the variable 'is_valid' (line 1321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1321, 4), 'is_valid', list_175463)
    
    
    # Getting the type of 'is_valid' (line 1322)
    is_valid_175464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1322, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1322)
    list_175465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1322, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1322)
    # Adding element type (line 1322)
    int_175466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1322, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1322, 19), list_175465, int_175466)
    # Adding element type (line 1322)
    int_175467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1322, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1322, 19), list_175465, int_175467)
    
    # Applying the binary operator '!=' (line 1322)
    result_ne_175468 = python_operator(stypy.reporting.localization.Localization(__file__, 1322, 7), '!=', is_valid_175464, list_175465)
    
    # Testing the type of an if condition (line 1322)
    if_condition_175469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1322, 4), result_ne_175468)
    # Assigning a type to the variable 'if_condition_175469' (line 1322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1322, 4), 'if_condition_175469', if_condition_175469)
    # SSA begins for if statement (line 1322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1323)
    # Processing the call arguments (line 1323)
    str_175471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1323, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1323)
    kwargs_175472 = {}
    # Getting the type of 'ValueError' (line 1323)
    ValueError_175470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1323, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1323)
    ValueError_call_result_175473 = invoke(stypy.reporting.localization.Localization(__file__, 1323, 14), ValueError_175470, *[str_175471], **kwargs_175472)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1323, 8), ValueError_call_result_175473, 'raise parameter', BaseException)
    # SSA join for if statement (line 1322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1324):
    
    # Assigning a Subscript to a Name (line 1324):
    
    # Obtaining the type of the subscript
    int_175474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1324, 4), 'int')
    # Getting the type of 'ideg' (line 1324)
    ideg_175475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1324, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1324)
    getitem___175476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1324, 4), ideg_175475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1324)
    subscript_call_result_175477 = invoke(stypy.reporting.localization.Localization(__file__, 1324, 4), getitem___175476, int_175474)
    
    # Assigning a type to the variable 'tuple_var_assignment_173639' (line 1324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1324, 4), 'tuple_var_assignment_173639', subscript_call_result_175477)
    
    # Assigning a Subscript to a Name (line 1324):
    
    # Obtaining the type of the subscript
    int_175478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1324, 4), 'int')
    # Getting the type of 'ideg' (line 1324)
    ideg_175479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1324, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1324)
    getitem___175480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1324, 4), ideg_175479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1324)
    subscript_call_result_175481 = invoke(stypy.reporting.localization.Localization(__file__, 1324, 4), getitem___175480, int_175478)
    
    # Assigning a type to the variable 'tuple_var_assignment_173640' (line 1324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1324, 4), 'tuple_var_assignment_173640', subscript_call_result_175481)
    
    # Assigning a Name to a Name (line 1324):
    # Getting the type of 'tuple_var_assignment_173639' (line 1324)
    tuple_var_assignment_173639_175482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1324, 4), 'tuple_var_assignment_173639')
    # Assigning a type to the variable 'degx' (line 1324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1324, 4), 'degx', tuple_var_assignment_173639_175482)
    
    # Assigning a Name to a Name (line 1324):
    # Getting the type of 'tuple_var_assignment_173640' (line 1324)
    tuple_var_assignment_173640_175483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1324, 4), 'tuple_var_assignment_173640')
    # Assigning a type to the variable 'degy' (line 1324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1324, 10), 'degy', tuple_var_assignment_173640_175483)
    
    # Assigning a BinOp to a Tuple (line 1325):
    
    # Assigning a Subscript to a Name (line 1325):
    
    # Obtaining the type of the subscript
    int_175484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 4), 'int')
    
    # Call to array(...): (line 1325)
    # Processing the call arguments (line 1325)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1325)
    tuple_175487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1325)
    # Adding element type (line 1325)
    # Getting the type of 'x' (line 1325)
    x_175488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 21), tuple_175487, x_175488)
    # Adding element type (line 1325)
    # Getting the type of 'y' (line 1325)
    y_175489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 21), tuple_175487, y_175489)
    
    # Processing the call keyword arguments (line 1325)
    int_175490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 33), 'int')
    keyword_175491 = int_175490
    kwargs_175492 = {'copy': keyword_175491}
    # Getting the type of 'np' (line 1325)
    np_175485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1325)
    array_175486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 11), np_175485, 'array')
    # Calling array(args, kwargs) (line 1325)
    array_call_result_175493 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 11), array_175486, *[tuple_175487], **kwargs_175492)
    
    float_175494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 38), 'float')
    # Applying the binary operator '+' (line 1325)
    result_add_175495 = python_operator(stypy.reporting.localization.Localization(__file__, 1325, 11), '+', array_call_result_175493, float_175494)
    
    # Obtaining the member '__getitem__' of a type (line 1325)
    getitem___175496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 4), result_add_175495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1325)
    subscript_call_result_175497 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 4), getitem___175496, int_175484)
    
    # Assigning a type to the variable 'tuple_var_assignment_173641' (line 1325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1325, 4), 'tuple_var_assignment_173641', subscript_call_result_175497)
    
    # Assigning a Subscript to a Name (line 1325):
    
    # Obtaining the type of the subscript
    int_175498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 4), 'int')
    
    # Call to array(...): (line 1325)
    # Processing the call arguments (line 1325)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1325)
    tuple_175501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1325)
    # Adding element type (line 1325)
    # Getting the type of 'x' (line 1325)
    x_175502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 21), tuple_175501, x_175502)
    # Adding element type (line 1325)
    # Getting the type of 'y' (line 1325)
    y_175503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 21), tuple_175501, y_175503)
    
    # Processing the call keyword arguments (line 1325)
    int_175504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 33), 'int')
    keyword_175505 = int_175504
    kwargs_175506 = {'copy': keyword_175505}
    # Getting the type of 'np' (line 1325)
    np_175499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1325)
    array_175500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 11), np_175499, 'array')
    # Calling array(args, kwargs) (line 1325)
    array_call_result_175507 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 11), array_175500, *[tuple_175501], **kwargs_175506)
    
    float_175508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 38), 'float')
    # Applying the binary operator '+' (line 1325)
    result_add_175509 = python_operator(stypy.reporting.localization.Localization(__file__, 1325, 11), '+', array_call_result_175507, float_175508)
    
    # Obtaining the member '__getitem__' of a type (line 1325)
    getitem___175510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 4), result_add_175509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1325)
    subscript_call_result_175511 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 4), getitem___175510, int_175498)
    
    # Assigning a type to the variable 'tuple_var_assignment_173642' (line 1325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1325, 4), 'tuple_var_assignment_173642', subscript_call_result_175511)
    
    # Assigning a Name to a Name (line 1325):
    # Getting the type of 'tuple_var_assignment_173641' (line 1325)
    tuple_var_assignment_173641_175512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 4), 'tuple_var_assignment_173641')
    # Assigning a type to the variable 'x' (line 1325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1325, 4), 'x', tuple_var_assignment_173641_175512)
    
    # Assigning a Name to a Name (line 1325):
    # Getting the type of 'tuple_var_assignment_173642' (line 1325)
    tuple_var_assignment_173642_175513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 4), 'tuple_var_assignment_173642')
    # Assigning a type to the variable 'y' (line 1325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1325, 7), 'y', tuple_var_assignment_173642_175513)
    
    # Assigning a Call to a Name (line 1327):
    
    # Assigning a Call to a Name (line 1327):
    
    # Call to legvander(...): (line 1327)
    # Processing the call arguments (line 1327)
    # Getting the type of 'x' (line 1327)
    x_175515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 19), 'x', False)
    # Getting the type of 'degx' (line 1327)
    degx_175516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 22), 'degx', False)
    # Processing the call keyword arguments (line 1327)
    kwargs_175517 = {}
    # Getting the type of 'legvander' (line 1327)
    legvander_175514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 9), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1327)
    legvander_call_result_175518 = invoke(stypy.reporting.localization.Localization(__file__, 1327, 9), legvander_175514, *[x_175515, degx_175516], **kwargs_175517)
    
    # Assigning a type to the variable 'vx' (line 1327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1327, 4), 'vx', legvander_call_result_175518)
    
    # Assigning a Call to a Name (line 1328):
    
    # Assigning a Call to a Name (line 1328):
    
    # Call to legvander(...): (line 1328)
    # Processing the call arguments (line 1328)
    # Getting the type of 'y' (line 1328)
    y_175520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 19), 'y', False)
    # Getting the type of 'degy' (line 1328)
    degy_175521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 22), 'degy', False)
    # Processing the call keyword arguments (line 1328)
    kwargs_175522 = {}
    # Getting the type of 'legvander' (line 1328)
    legvander_175519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 9), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1328)
    legvander_call_result_175523 = invoke(stypy.reporting.localization.Localization(__file__, 1328, 9), legvander_175519, *[y_175520, degy_175521], **kwargs_175522)
    
    # Assigning a type to the variable 'vy' (line 1328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'vy', legvander_call_result_175523)
    
    # Assigning a BinOp to a Name (line 1329):
    
    # Assigning a BinOp to a Name (line 1329):
    
    # Obtaining the type of the subscript
    Ellipsis_175524 = Ellipsis
    # Getting the type of 'None' (line 1329)
    None_175525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 16), 'None')
    # Getting the type of 'vx' (line 1329)
    vx_175526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1329)
    getitem___175527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1329, 8), vx_175526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1329)
    subscript_call_result_175528 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 8), getitem___175527, (Ellipsis_175524, None_175525))
    
    
    # Obtaining the type of the subscript
    Ellipsis_175529 = Ellipsis
    # Getting the type of 'None' (line 1329)
    None_175530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 30), 'None')
    slice_175531 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1329, 22), None, None, None)
    # Getting the type of 'vy' (line 1329)
    vy_175532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 22), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1329)
    getitem___175533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1329, 22), vy_175532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1329)
    subscript_call_result_175534 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 22), getitem___175533, (Ellipsis_175529, None_175530, slice_175531))
    
    # Applying the binary operator '*' (line 1329)
    result_mul_175535 = python_operator(stypy.reporting.localization.Localization(__file__, 1329, 8), '*', subscript_call_result_175528, subscript_call_result_175534)
    
    # Assigning a type to the variable 'v' (line 1329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'v', result_mul_175535)
    
    # Call to reshape(...): (line 1330)
    # Processing the call arguments (line 1330)
    
    # Obtaining the type of the subscript
    int_175538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 30), 'int')
    slice_175539 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1330, 21), None, int_175538, None)
    # Getting the type of 'v' (line 1330)
    v_175540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1330)
    shape_175541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1330, 21), v_175540, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1330)
    getitem___175542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1330, 21), shape_175541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1330)
    subscript_call_result_175543 = invoke(stypy.reporting.localization.Localization(__file__, 1330, 21), getitem___175542, slice_175539)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1330)
    tuple_175544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1330)
    # Adding element type (line 1330)
    int_175545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1330, 37), tuple_175544, int_175545)
    
    # Applying the binary operator '+' (line 1330)
    result_add_175546 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 21), '+', subscript_call_result_175543, tuple_175544)
    
    # Processing the call keyword arguments (line 1330)
    kwargs_175547 = {}
    # Getting the type of 'v' (line 1330)
    v_175536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1330)
    reshape_175537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1330, 11), v_175536, 'reshape')
    # Calling reshape(args, kwargs) (line 1330)
    reshape_call_result_175548 = invoke(stypy.reporting.localization.Localization(__file__, 1330, 11), reshape_175537, *[result_add_175546], **kwargs_175547)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1330, 4), 'stypy_return_type', reshape_call_result_175548)
    
    # ################# End of 'legvander2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legvander2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1270)
    stypy_return_type_175549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175549)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legvander2d'
    return stypy_return_type_175549

# Assigning a type to the variable 'legvander2d' (line 1270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1270, 0), 'legvander2d', legvander2d)

@norecursion
def legvander3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legvander3d'
    module_type_store = module_type_store.open_function_context('legvander3d', 1333, 0, False)
    
    # Passed parameters checking function
    legvander3d.stypy_localization = localization
    legvander3d.stypy_type_of_self = None
    legvander3d.stypy_type_store = module_type_store
    legvander3d.stypy_function_name = 'legvander3d'
    legvander3d.stypy_param_names_list = ['x', 'y', 'z', 'deg']
    legvander3d.stypy_varargs_param_name = None
    legvander3d.stypy_kwargs_param_name = None
    legvander3d.stypy_call_defaults = defaults
    legvander3d.stypy_call_varargs = varargs
    legvander3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legvander3d', ['x', 'y', 'z', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legvander3d', localization, ['x', 'y', 'z', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legvander3d(...)' code ##################

    str_175550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the degrees of the Legendre polynomials.\n\n    If ``V = legvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and ``np.dot(V, c.flat)`` and ``legval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D Legendre\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    legvander, legvander3d. legval2d, legval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1384):
    
    # Assigning a ListComp to a Name (line 1384):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1384)
    deg_175555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 28), 'deg')
    comprehension_175556 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1384, 12), deg_175555)
    # Assigning a type to the variable 'd' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 12), 'd', comprehension_175556)
    
    # Call to int(...): (line 1384)
    # Processing the call arguments (line 1384)
    # Getting the type of 'd' (line 1384)
    d_175552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 16), 'd', False)
    # Processing the call keyword arguments (line 1384)
    kwargs_175553 = {}
    # Getting the type of 'int' (line 1384)
    int_175551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 12), 'int', False)
    # Calling int(args, kwargs) (line 1384)
    int_call_result_175554 = invoke(stypy.reporting.localization.Localization(__file__, 1384, 12), int_175551, *[d_175552], **kwargs_175553)
    
    list_175557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1384, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1384, 12), list_175557, int_call_result_175554)
    # Assigning a type to the variable 'ideg' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 4), 'ideg', list_175557)
    
    # Assigning a ListComp to a Name (line 1385):
    
    # Assigning a ListComp to a Name (line 1385):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1385)
    # Processing the call arguments (line 1385)
    # Getting the type of 'ideg' (line 1385)
    ideg_175566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1385)
    deg_175567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 59), 'deg', False)
    # Processing the call keyword arguments (line 1385)
    kwargs_175568 = {}
    # Getting the type of 'zip' (line 1385)
    zip_175565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1385)
    zip_call_result_175569 = invoke(stypy.reporting.localization.Localization(__file__, 1385, 49), zip_175565, *[ideg_175566, deg_175567], **kwargs_175568)
    
    comprehension_175570 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1385, 16), zip_call_result_175569)
    # Assigning a type to the variable 'id' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1385, 16), comprehension_175570))
    # Assigning a type to the variable 'd' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1385, 16), comprehension_175570))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1385)
    id_175558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 16), 'id')
    # Getting the type of 'd' (line 1385)
    d_175559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 22), 'd')
    # Applying the binary operator '==' (line 1385)
    result_eq_175560 = python_operator(stypy.reporting.localization.Localization(__file__, 1385, 16), '==', id_175558, d_175559)
    
    
    # Getting the type of 'id' (line 1385)
    id_175561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 28), 'id')
    int_175562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1385, 34), 'int')
    # Applying the binary operator '>=' (line 1385)
    result_ge_175563 = python_operator(stypy.reporting.localization.Localization(__file__, 1385, 28), '>=', id_175561, int_175562)
    
    # Applying the binary operator 'and' (line 1385)
    result_and_keyword_175564 = python_operator(stypy.reporting.localization.Localization(__file__, 1385, 16), 'and', result_eq_175560, result_ge_175563)
    
    list_175571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1385, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1385, 16), list_175571, result_and_keyword_175564)
    # Assigning a type to the variable 'is_valid' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 4), 'is_valid', list_175571)
    
    
    # Getting the type of 'is_valid' (line 1386)
    is_valid_175572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1386)
    list_175573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1386)
    # Adding element type (line 1386)
    int_175574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1386, 19), list_175573, int_175574)
    # Adding element type (line 1386)
    int_175575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1386, 19), list_175573, int_175575)
    # Adding element type (line 1386)
    int_175576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1386, 19), list_175573, int_175576)
    
    # Applying the binary operator '!=' (line 1386)
    result_ne_175577 = python_operator(stypy.reporting.localization.Localization(__file__, 1386, 7), '!=', is_valid_175572, list_175573)
    
    # Testing the type of an if condition (line 1386)
    if_condition_175578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1386, 4), result_ne_175577)
    # Assigning a type to the variable 'if_condition_175578' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 4), 'if_condition_175578', if_condition_175578)
    # SSA begins for if statement (line 1386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1387)
    # Processing the call arguments (line 1387)
    str_175580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1387)
    kwargs_175581 = {}
    # Getting the type of 'ValueError' (line 1387)
    ValueError_175579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1387)
    ValueError_call_result_175582 = invoke(stypy.reporting.localization.Localization(__file__, 1387, 14), ValueError_175579, *[str_175580], **kwargs_175581)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1387, 8), ValueError_call_result_175582, 'raise parameter', BaseException)
    # SSA join for if statement (line 1386)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1388):
    
    # Assigning a Subscript to a Name (line 1388):
    
    # Obtaining the type of the subscript
    int_175583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, 4), 'int')
    # Getting the type of 'ideg' (line 1388)
    ideg_175584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1388)
    getitem___175585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 4), ideg_175584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1388)
    subscript_call_result_175586 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 4), getitem___175585, int_175583)
    
    # Assigning a type to the variable 'tuple_var_assignment_173643' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'tuple_var_assignment_173643', subscript_call_result_175586)
    
    # Assigning a Subscript to a Name (line 1388):
    
    # Obtaining the type of the subscript
    int_175587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, 4), 'int')
    # Getting the type of 'ideg' (line 1388)
    ideg_175588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1388)
    getitem___175589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 4), ideg_175588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1388)
    subscript_call_result_175590 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 4), getitem___175589, int_175587)
    
    # Assigning a type to the variable 'tuple_var_assignment_173644' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'tuple_var_assignment_173644', subscript_call_result_175590)
    
    # Assigning a Subscript to a Name (line 1388):
    
    # Obtaining the type of the subscript
    int_175591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, 4), 'int')
    # Getting the type of 'ideg' (line 1388)
    ideg_175592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1388)
    getitem___175593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 4), ideg_175592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1388)
    subscript_call_result_175594 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 4), getitem___175593, int_175591)
    
    # Assigning a type to the variable 'tuple_var_assignment_173645' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'tuple_var_assignment_173645', subscript_call_result_175594)
    
    # Assigning a Name to a Name (line 1388):
    # Getting the type of 'tuple_var_assignment_173643' (line 1388)
    tuple_var_assignment_173643_175595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'tuple_var_assignment_173643')
    # Assigning a type to the variable 'degx' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'degx', tuple_var_assignment_173643_175595)
    
    # Assigning a Name to a Name (line 1388):
    # Getting the type of 'tuple_var_assignment_173644' (line 1388)
    tuple_var_assignment_173644_175596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'tuple_var_assignment_173644')
    # Assigning a type to the variable 'degy' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 10), 'degy', tuple_var_assignment_173644_175596)
    
    # Assigning a Name to a Name (line 1388):
    # Getting the type of 'tuple_var_assignment_173645' (line 1388)
    tuple_var_assignment_173645_175597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'tuple_var_assignment_173645')
    # Assigning a type to the variable 'degz' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 16), 'degz', tuple_var_assignment_173645_175597)
    
    # Assigning a BinOp to a Tuple (line 1389):
    
    # Assigning a Subscript to a Name (line 1389):
    
    # Obtaining the type of the subscript
    int_175598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 4), 'int')
    
    # Call to array(...): (line 1389)
    # Processing the call arguments (line 1389)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1389)
    tuple_175601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1389)
    # Adding element type (line 1389)
    # Getting the type of 'x' (line 1389)
    x_175602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175601, x_175602)
    # Adding element type (line 1389)
    # Getting the type of 'y' (line 1389)
    y_175603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175601, y_175603)
    # Adding element type (line 1389)
    # Getting the type of 'z' (line 1389)
    z_175604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175601, z_175604)
    
    # Processing the call keyword arguments (line 1389)
    int_175605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 39), 'int')
    keyword_175606 = int_175605
    kwargs_175607 = {'copy': keyword_175606}
    # Getting the type of 'np' (line 1389)
    np_175599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1389)
    array_175600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 14), np_175599, 'array')
    # Calling array(args, kwargs) (line 1389)
    array_call_result_175608 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 14), array_175600, *[tuple_175601], **kwargs_175607)
    
    float_175609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 44), 'float')
    # Applying the binary operator '+' (line 1389)
    result_add_175610 = python_operator(stypy.reporting.localization.Localization(__file__, 1389, 14), '+', array_call_result_175608, float_175609)
    
    # Obtaining the member '__getitem__' of a type (line 1389)
    getitem___175611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 4), result_add_175610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1389)
    subscript_call_result_175612 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 4), getitem___175611, int_175598)
    
    # Assigning a type to the variable 'tuple_var_assignment_173646' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tuple_var_assignment_173646', subscript_call_result_175612)
    
    # Assigning a Subscript to a Name (line 1389):
    
    # Obtaining the type of the subscript
    int_175613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 4), 'int')
    
    # Call to array(...): (line 1389)
    # Processing the call arguments (line 1389)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1389)
    tuple_175616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1389)
    # Adding element type (line 1389)
    # Getting the type of 'x' (line 1389)
    x_175617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175616, x_175617)
    # Adding element type (line 1389)
    # Getting the type of 'y' (line 1389)
    y_175618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175616, y_175618)
    # Adding element type (line 1389)
    # Getting the type of 'z' (line 1389)
    z_175619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175616, z_175619)
    
    # Processing the call keyword arguments (line 1389)
    int_175620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 39), 'int')
    keyword_175621 = int_175620
    kwargs_175622 = {'copy': keyword_175621}
    # Getting the type of 'np' (line 1389)
    np_175614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1389)
    array_175615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 14), np_175614, 'array')
    # Calling array(args, kwargs) (line 1389)
    array_call_result_175623 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 14), array_175615, *[tuple_175616], **kwargs_175622)
    
    float_175624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 44), 'float')
    # Applying the binary operator '+' (line 1389)
    result_add_175625 = python_operator(stypy.reporting.localization.Localization(__file__, 1389, 14), '+', array_call_result_175623, float_175624)
    
    # Obtaining the member '__getitem__' of a type (line 1389)
    getitem___175626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 4), result_add_175625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1389)
    subscript_call_result_175627 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 4), getitem___175626, int_175613)
    
    # Assigning a type to the variable 'tuple_var_assignment_173647' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tuple_var_assignment_173647', subscript_call_result_175627)
    
    # Assigning a Subscript to a Name (line 1389):
    
    # Obtaining the type of the subscript
    int_175628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 4), 'int')
    
    # Call to array(...): (line 1389)
    # Processing the call arguments (line 1389)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1389)
    tuple_175631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1389)
    # Adding element type (line 1389)
    # Getting the type of 'x' (line 1389)
    x_175632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175631, x_175632)
    # Adding element type (line 1389)
    # Getting the type of 'y' (line 1389)
    y_175633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175631, y_175633)
    # Adding element type (line 1389)
    # Getting the type of 'z' (line 1389)
    z_175634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 24), tuple_175631, z_175634)
    
    # Processing the call keyword arguments (line 1389)
    int_175635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 39), 'int')
    keyword_175636 = int_175635
    kwargs_175637 = {'copy': keyword_175636}
    # Getting the type of 'np' (line 1389)
    np_175629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1389)
    array_175630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 14), np_175629, 'array')
    # Calling array(args, kwargs) (line 1389)
    array_call_result_175638 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 14), array_175630, *[tuple_175631], **kwargs_175637)
    
    float_175639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 44), 'float')
    # Applying the binary operator '+' (line 1389)
    result_add_175640 = python_operator(stypy.reporting.localization.Localization(__file__, 1389, 14), '+', array_call_result_175638, float_175639)
    
    # Obtaining the member '__getitem__' of a type (line 1389)
    getitem___175641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 4), result_add_175640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1389)
    subscript_call_result_175642 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 4), getitem___175641, int_175628)
    
    # Assigning a type to the variable 'tuple_var_assignment_173648' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tuple_var_assignment_173648', subscript_call_result_175642)
    
    # Assigning a Name to a Name (line 1389):
    # Getting the type of 'tuple_var_assignment_173646' (line 1389)
    tuple_var_assignment_173646_175643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tuple_var_assignment_173646')
    # Assigning a type to the variable 'x' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'x', tuple_var_assignment_173646_175643)
    
    # Assigning a Name to a Name (line 1389):
    # Getting the type of 'tuple_var_assignment_173647' (line 1389)
    tuple_var_assignment_173647_175644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tuple_var_assignment_173647')
    # Assigning a type to the variable 'y' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 7), 'y', tuple_var_assignment_173647_175644)
    
    # Assigning a Name to a Name (line 1389):
    # Getting the type of 'tuple_var_assignment_173648' (line 1389)
    tuple_var_assignment_173648_175645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tuple_var_assignment_173648')
    # Assigning a type to the variable 'z' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 10), 'z', tuple_var_assignment_173648_175645)
    
    # Assigning a Call to a Name (line 1391):
    
    # Assigning a Call to a Name (line 1391):
    
    # Call to legvander(...): (line 1391)
    # Processing the call arguments (line 1391)
    # Getting the type of 'x' (line 1391)
    x_175647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 19), 'x', False)
    # Getting the type of 'degx' (line 1391)
    degx_175648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 22), 'degx', False)
    # Processing the call keyword arguments (line 1391)
    kwargs_175649 = {}
    # Getting the type of 'legvander' (line 1391)
    legvander_175646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 9), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1391)
    legvander_call_result_175650 = invoke(stypy.reporting.localization.Localization(__file__, 1391, 9), legvander_175646, *[x_175647, degx_175648], **kwargs_175649)
    
    # Assigning a type to the variable 'vx' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 4), 'vx', legvander_call_result_175650)
    
    # Assigning a Call to a Name (line 1392):
    
    # Assigning a Call to a Name (line 1392):
    
    # Call to legvander(...): (line 1392)
    # Processing the call arguments (line 1392)
    # Getting the type of 'y' (line 1392)
    y_175652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 19), 'y', False)
    # Getting the type of 'degy' (line 1392)
    degy_175653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 22), 'degy', False)
    # Processing the call keyword arguments (line 1392)
    kwargs_175654 = {}
    # Getting the type of 'legvander' (line 1392)
    legvander_175651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 9), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1392)
    legvander_call_result_175655 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 9), legvander_175651, *[y_175652, degy_175653], **kwargs_175654)
    
    # Assigning a type to the variable 'vy' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'vy', legvander_call_result_175655)
    
    # Assigning a Call to a Name (line 1393):
    
    # Assigning a Call to a Name (line 1393):
    
    # Call to legvander(...): (line 1393)
    # Processing the call arguments (line 1393)
    # Getting the type of 'z' (line 1393)
    z_175657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 19), 'z', False)
    # Getting the type of 'degz' (line 1393)
    degz_175658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 22), 'degz', False)
    # Processing the call keyword arguments (line 1393)
    kwargs_175659 = {}
    # Getting the type of 'legvander' (line 1393)
    legvander_175656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 9), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1393)
    legvander_call_result_175660 = invoke(stypy.reporting.localization.Localization(__file__, 1393, 9), legvander_175656, *[z_175657, degz_175658], **kwargs_175659)
    
    # Assigning a type to the variable 'vz' (line 1393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1393, 4), 'vz', legvander_call_result_175660)
    
    # Assigning a BinOp to a Name (line 1394):
    
    # Assigning a BinOp to a Name (line 1394):
    
    # Obtaining the type of the subscript
    Ellipsis_175661 = Ellipsis
    # Getting the type of 'None' (line 1394)
    None_175662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 16), 'None')
    # Getting the type of 'None' (line 1394)
    None_175663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 22), 'None')
    # Getting the type of 'vx' (line 1394)
    vx_175664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1394)
    getitem___175665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1394, 8), vx_175664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1394)
    subscript_call_result_175666 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 8), getitem___175665, (Ellipsis_175661, None_175662, None_175663))
    
    
    # Obtaining the type of the subscript
    Ellipsis_175667 = Ellipsis
    # Getting the type of 'None' (line 1394)
    None_175668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 36), 'None')
    slice_175669 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1394, 28), None, None, None)
    # Getting the type of 'None' (line 1394)
    None_175670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 44), 'None')
    # Getting the type of 'vy' (line 1394)
    vy_175671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 28), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1394)
    getitem___175672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1394, 28), vy_175671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1394)
    subscript_call_result_175673 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 28), getitem___175672, (Ellipsis_175667, None_175668, slice_175669, None_175670))
    
    # Applying the binary operator '*' (line 1394)
    result_mul_175674 = python_operator(stypy.reporting.localization.Localization(__file__, 1394, 8), '*', subscript_call_result_175666, subscript_call_result_175673)
    
    
    # Obtaining the type of the subscript
    Ellipsis_175675 = Ellipsis
    # Getting the type of 'None' (line 1394)
    None_175676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 58), 'None')
    # Getting the type of 'None' (line 1394)
    None_175677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 64), 'None')
    slice_175678 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1394, 50), None, None, None)
    # Getting the type of 'vz' (line 1394)
    vz_175679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 50), 'vz')
    # Obtaining the member '__getitem__' of a type (line 1394)
    getitem___175680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1394, 50), vz_175679, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1394)
    subscript_call_result_175681 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 50), getitem___175680, (Ellipsis_175675, None_175676, None_175677, slice_175678))
    
    # Applying the binary operator '*' (line 1394)
    result_mul_175682 = python_operator(stypy.reporting.localization.Localization(__file__, 1394, 49), '*', result_mul_175674, subscript_call_result_175681)
    
    # Assigning a type to the variable 'v' (line 1394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1394, 4), 'v', result_mul_175682)
    
    # Call to reshape(...): (line 1395)
    # Processing the call arguments (line 1395)
    
    # Obtaining the type of the subscript
    int_175685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1395, 30), 'int')
    slice_175686 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1395, 21), None, int_175685, None)
    # Getting the type of 'v' (line 1395)
    v_175687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1395, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1395)
    shape_175688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1395, 21), v_175687, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1395)
    getitem___175689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1395, 21), shape_175688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1395)
    subscript_call_result_175690 = invoke(stypy.reporting.localization.Localization(__file__, 1395, 21), getitem___175689, slice_175686)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1395)
    tuple_175691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1395, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1395)
    # Adding element type (line 1395)
    int_175692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1395, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1395, 37), tuple_175691, int_175692)
    
    # Applying the binary operator '+' (line 1395)
    result_add_175693 = python_operator(stypy.reporting.localization.Localization(__file__, 1395, 21), '+', subscript_call_result_175690, tuple_175691)
    
    # Processing the call keyword arguments (line 1395)
    kwargs_175694 = {}
    # Getting the type of 'v' (line 1395)
    v_175683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1395, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1395)
    reshape_175684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1395, 11), v_175683, 'reshape')
    # Calling reshape(args, kwargs) (line 1395)
    reshape_call_result_175695 = invoke(stypy.reporting.localization.Localization(__file__, 1395, 11), reshape_175684, *[result_add_175693], **kwargs_175694)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1395, 4), 'stypy_return_type', reshape_call_result_175695)
    
    # ################# End of 'legvander3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legvander3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1333)
    stypy_return_type_175696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1333, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_175696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legvander3d'
    return stypy_return_type_175696

# Assigning a type to the variable 'legvander3d' (line 1333)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1333, 0), 'legvander3d', legvander3d)

@norecursion
def legfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1398)
    None_175697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 28), 'None')
    # Getting the type of 'False' (line 1398)
    False_175698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 39), 'False')
    # Getting the type of 'None' (line 1398)
    None_175699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 48), 'None')
    defaults = [None_175697, False_175698, None_175699]
    # Create a new context for function 'legfit'
    module_type_store = module_type_store.open_function_context('legfit', 1398, 0, False)
    
    # Passed parameters checking function
    legfit.stypy_localization = localization
    legfit.stypy_type_of_self = None
    legfit.stypy_type_store = module_type_store
    legfit.stypy_function_name = 'legfit'
    legfit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w']
    legfit.stypy_varargs_param_name = None
    legfit.stypy_kwargs_param_name = None
    legfit.stypy_call_defaults = defaults
    legfit.stypy_call_varargs = varargs
    legfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legfit', ['x', 'y', 'deg', 'rcond', 'full', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legfit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legfit(...)' code ##################

    str_175700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1518, (-1)), 'str', '\n    Least squares fit of Legendre series to data.\n\n    Return the coefficients of a Legendre series of degree `deg` that is the\n    least squares fit to the data values `y` given at points `x`. If `y` is\n    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple\n    fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),\n\n    where `n` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For Numpy versions >= 1.11 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the contribution of each point\n        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n        weights are chosen so that the errors of the products ``w[i]*y[i]``\n        all have the same variance.  The default value is None.\n\n        .. versionadded:: 1.5.0\n\n    Returns\n    -------\n    coef : ndarray, shape (M,) or (M, K)\n        Legendre coefficients ordered from low to high. If `y` was\n        2-D, the coefficients for the data in column k of `y` are in\n        column `k`. If `deg` is specified as a list, coefficients for\n        terms not included in the fit are set equal to zero in the\n        returned `coef`.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if `full` = True\n\n        resid -- sum of squared residuals of the least squares fit\n        rank -- the numerical rank of the scaled Vandermonde matrix\n        sv -- singular values of the scaled Vandermonde matrix\n        rcond -- value of `rcond`.\n\n        For more details, see `linalg.lstsq`.\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if `full` = False.  The\n        warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', RankWarning)\n\n    See Also\n    --------\n    chebfit, polyfit, lagfit, hermfit, hermefit\n    legval : Evaluates a Legendre series.\n    legvander : Vandermonde matrix of Legendre series.\n    legweight : Legendre weight function (= 1).\n    linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the Legendre series `p` that\n    minimizes the sum of the weighted squared errors\n\n    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where :math:`w_j` are the weights. This problem is solved by setting up\n    as the (typically) overdetermined matrix equation\n\n    .. math:: V(x) * c = w * y,\n\n    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the\n    coefficients to be solved for, `w` are the weights, and `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of `V`.\n\n    If some of the singular values of `V` are so small that they are\n    neglected, then a `RankWarning` will be issued. This means that the\n    coefficient values may be poorly determined. Using a lower order fit\n    will usually get rid of the warning.  The `rcond` parameter can also be\n    set to a value smaller than its default, but the resulting fit may be\n    spurious and have large contributions from roundoff error.\n\n    Fits using Legendre series are usually better conditioned than fits\n    using power series, but much can depend on the distribution of the\n    sample points and the smoothness of the data. If the quality of the fit\n    is inadequate splines may be a good alternative.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           http://en.wikipedia.org/wiki/Curve_fitting\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a BinOp to a Name (line 1519):
    
    # Assigning a BinOp to a Name (line 1519):
    
    # Call to asarray(...): (line 1519)
    # Processing the call arguments (line 1519)
    # Getting the type of 'x' (line 1519)
    x_175703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 19), 'x', False)
    # Processing the call keyword arguments (line 1519)
    kwargs_175704 = {}
    # Getting the type of 'np' (line 1519)
    np_175701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1519)
    asarray_175702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1519, 8), np_175701, 'asarray')
    # Calling asarray(args, kwargs) (line 1519)
    asarray_call_result_175705 = invoke(stypy.reporting.localization.Localization(__file__, 1519, 8), asarray_175702, *[x_175703], **kwargs_175704)
    
    float_175706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1519, 24), 'float')
    # Applying the binary operator '+' (line 1519)
    result_add_175707 = python_operator(stypy.reporting.localization.Localization(__file__, 1519, 8), '+', asarray_call_result_175705, float_175706)
    
    # Assigning a type to the variable 'x' (line 1519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1519, 4), 'x', result_add_175707)
    
    # Assigning a BinOp to a Name (line 1520):
    
    # Assigning a BinOp to a Name (line 1520):
    
    # Call to asarray(...): (line 1520)
    # Processing the call arguments (line 1520)
    # Getting the type of 'y' (line 1520)
    y_175710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 19), 'y', False)
    # Processing the call keyword arguments (line 1520)
    kwargs_175711 = {}
    # Getting the type of 'np' (line 1520)
    np_175708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1520)
    asarray_175709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1520, 8), np_175708, 'asarray')
    # Calling asarray(args, kwargs) (line 1520)
    asarray_call_result_175712 = invoke(stypy.reporting.localization.Localization(__file__, 1520, 8), asarray_175709, *[y_175710], **kwargs_175711)
    
    float_175713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1520, 24), 'float')
    # Applying the binary operator '+' (line 1520)
    result_add_175714 = python_operator(stypy.reporting.localization.Localization(__file__, 1520, 8), '+', asarray_call_result_175712, float_175713)
    
    # Assigning a type to the variable 'y' (line 1520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1520, 4), 'y', result_add_175714)
    
    # Assigning a Call to a Name (line 1521):
    
    # Assigning a Call to a Name (line 1521):
    
    # Call to asarray(...): (line 1521)
    # Processing the call arguments (line 1521)
    # Getting the type of 'deg' (line 1521)
    deg_175717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 21), 'deg', False)
    # Processing the call keyword arguments (line 1521)
    kwargs_175718 = {}
    # Getting the type of 'np' (line 1521)
    np_175715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1521)
    asarray_175716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1521, 10), np_175715, 'asarray')
    # Calling asarray(args, kwargs) (line 1521)
    asarray_call_result_175719 = invoke(stypy.reporting.localization.Localization(__file__, 1521, 10), asarray_175716, *[deg_175717], **kwargs_175718)
    
    # Assigning a type to the variable 'deg' (line 1521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1521, 4), 'deg', asarray_call_result_175719)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deg' (line 1524)
    deg_175720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1524)
    ndim_175721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 7), deg_175720, 'ndim')
    int_175722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 18), 'int')
    # Applying the binary operator '>' (line 1524)
    result_gt_175723 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 7), '>', ndim_175721, int_175722)
    
    
    # Getting the type of 'deg' (line 1524)
    deg_175724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 23), 'deg')
    # Obtaining the member 'dtype' of a type (line 1524)
    dtype_175725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 23), deg_175724, 'dtype')
    # Obtaining the member 'kind' of a type (line 1524)
    kind_175726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 23), dtype_175725, 'kind')
    str_175727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 45), 'str', 'iu')
    # Applying the binary operator 'notin' (line 1524)
    result_contains_175728 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 23), 'notin', kind_175726, str_175727)
    
    # Applying the binary operator 'or' (line 1524)
    result_or_keyword_175729 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 7), 'or', result_gt_175723, result_contains_175728)
    
    # Getting the type of 'deg' (line 1524)
    deg_175730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 53), 'deg')
    # Obtaining the member 'size' of a type (line 1524)
    size_175731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 53), deg_175730, 'size')
    int_175732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 65), 'int')
    # Applying the binary operator '==' (line 1524)
    result_eq_175733 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 53), '==', size_175731, int_175732)
    
    # Applying the binary operator 'or' (line 1524)
    result_or_keyword_175734 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 7), 'or', result_or_keyword_175729, result_eq_175733)
    
    # Testing the type of an if condition (line 1524)
    if_condition_175735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1524, 4), result_or_keyword_175734)
    # Assigning a type to the variable 'if_condition_175735' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'if_condition_175735', if_condition_175735)
    # SSA begins for if statement (line 1524)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1525)
    # Processing the call arguments (line 1525)
    str_175737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1525, 24), 'str', 'deg must be an int or non-empty 1-D array of int')
    # Processing the call keyword arguments (line 1525)
    kwargs_175738 = {}
    # Getting the type of 'TypeError' (line 1525)
    TypeError_175736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1525, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1525)
    TypeError_call_result_175739 = invoke(stypy.reporting.localization.Localization(__file__, 1525, 14), TypeError_175736, *[str_175737], **kwargs_175738)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1525, 8), TypeError_call_result_175739, 'raise parameter', BaseException)
    # SSA join for if statement (line 1524)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to min(...): (line 1526)
    # Processing the call keyword arguments (line 1526)
    kwargs_175742 = {}
    # Getting the type of 'deg' (line 1526)
    deg_175740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 7), 'deg', False)
    # Obtaining the member 'min' of a type (line 1526)
    min_175741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1526, 7), deg_175740, 'min')
    # Calling min(args, kwargs) (line 1526)
    min_call_result_175743 = invoke(stypy.reporting.localization.Localization(__file__, 1526, 7), min_175741, *[], **kwargs_175742)
    
    int_175744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1526, 19), 'int')
    # Applying the binary operator '<' (line 1526)
    result_lt_175745 = python_operator(stypy.reporting.localization.Localization(__file__, 1526, 7), '<', min_call_result_175743, int_175744)
    
    # Testing the type of an if condition (line 1526)
    if_condition_175746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1526, 4), result_lt_175745)
    # Assigning a type to the variable 'if_condition_175746' (line 1526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1526, 4), 'if_condition_175746', if_condition_175746)
    # SSA begins for if statement (line 1526)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1527)
    # Processing the call arguments (line 1527)
    str_175748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1527, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 1527)
    kwargs_175749 = {}
    # Getting the type of 'ValueError' (line 1527)
    ValueError_175747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1527)
    ValueError_call_result_175750 = invoke(stypy.reporting.localization.Localization(__file__, 1527, 14), ValueError_175747, *[str_175748], **kwargs_175749)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1527, 8), ValueError_call_result_175750, 'raise parameter', BaseException)
    # SSA join for if statement (line 1526)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1528)
    x_175751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1528)
    ndim_175752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 7), x_175751, 'ndim')
    int_175753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1528, 17), 'int')
    # Applying the binary operator '!=' (line 1528)
    result_ne_175754 = python_operator(stypy.reporting.localization.Localization(__file__, 1528, 7), '!=', ndim_175752, int_175753)
    
    # Testing the type of an if condition (line 1528)
    if_condition_175755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1528, 4), result_ne_175754)
    # Assigning a type to the variable 'if_condition_175755' (line 1528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1528, 4), 'if_condition_175755', if_condition_175755)
    # SSA begins for if statement (line 1528)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1529)
    # Processing the call arguments (line 1529)
    str_175757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1529, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 1529)
    kwargs_175758 = {}
    # Getting the type of 'TypeError' (line 1529)
    TypeError_175756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1529)
    TypeError_call_result_175759 = invoke(stypy.reporting.localization.Localization(__file__, 1529, 14), TypeError_175756, *[str_175757], **kwargs_175758)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1529, 8), TypeError_call_result_175759, 'raise parameter', BaseException)
    # SSA join for if statement (line 1528)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1530)
    x_175760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1530, 7), 'x')
    # Obtaining the member 'size' of a type (line 1530)
    size_175761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1530, 7), x_175760, 'size')
    int_175762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1530, 17), 'int')
    # Applying the binary operator '==' (line 1530)
    result_eq_175763 = python_operator(stypy.reporting.localization.Localization(__file__, 1530, 7), '==', size_175761, int_175762)
    
    # Testing the type of an if condition (line 1530)
    if_condition_175764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1530, 4), result_eq_175763)
    # Assigning a type to the variable 'if_condition_175764' (line 1530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1530, 4), 'if_condition_175764', if_condition_175764)
    # SSA begins for if statement (line 1530)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1531)
    # Processing the call arguments (line 1531)
    str_175766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1531, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 1531)
    kwargs_175767 = {}
    # Getting the type of 'TypeError' (line 1531)
    TypeError_175765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1531)
    TypeError_call_result_175768 = invoke(stypy.reporting.localization.Localization(__file__, 1531, 14), TypeError_175765, *[str_175766], **kwargs_175767)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1531, 8), TypeError_call_result_175768, 'raise parameter', BaseException)
    # SSA join for if statement (line 1530)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 1532)
    y_175769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1532)
    ndim_175770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1532, 7), y_175769, 'ndim')
    int_175771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1532, 16), 'int')
    # Applying the binary operator '<' (line 1532)
    result_lt_175772 = python_operator(stypy.reporting.localization.Localization(__file__, 1532, 7), '<', ndim_175770, int_175771)
    
    
    # Getting the type of 'y' (line 1532)
    y_175773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 1532)
    ndim_175774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1532, 21), y_175773, 'ndim')
    int_175775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1532, 30), 'int')
    # Applying the binary operator '>' (line 1532)
    result_gt_175776 = python_operator(stypy.reporting.localization.Localization(__file__, 1532, 21), '>', ndim_175774, int_175775)
    
    # Applying the binary operator 'or' (line 1532)
    result_or_keyword_175777 = python_operator(stypy.reporting.localization.Localization(__file__, 1532, 7), 'or', result_lt_175772, result_gt_175776)
    
    # Testing the type of an if condition (line 1532)
    if_condition_175778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1532, 4), result_or_keyword_175777)
    # Assigning a type to the variable 'if_condition_175778' (line 1532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1532, 4), 'if_condition_175778', if_condition_175778)
    # SSA begins for if statement (line 1532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1533)
    # Processing the call arguments (line 1533)
    str_175780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1533, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 1533)
    kwargs_175781 = {}
    # Getting the type of 'TypeError' (line 1533)
    TypeError_175779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1533)
    TypeError_call_result_175782 = invoke(stypy.reporting.localization.Localization(__file__, 1533, 14), TypeError_175779, *[str_175780], **kwargs_175781)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1533, 8), TypeError_call_result_175782, 'raise parameter', BaseException)
    # SSA join for if statement (line 1532)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1534)
    # Processing the call arguments (line 1534)
    # Getting the type of 'x' (line 1534)
    x_175784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 11), 'x', False)
    # Processing the call keyword arguments (line 1534)
    kwargs_175785 = {}
    # Getting the type of 'len' (line 1534)
    len_175783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 7), 'len', False)
    # Calling len(args, kwargs) (line 1534)
    len_call_result_175786 = invoke(stypy.reporting.localization.Localization(__file__, 1534, 7), len_175783, *[x_175784], **kwargs_175785)
    
    
    # Call to len(...): (line 1534)
    # Processing the call arguments (line 1534)
    # Getting the type of 'y' (line 1534)
    y_175788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 21), 'y', False)
    # Processing the call keyword arguments (line 1534)
    kwargs_175789 = {}
    # Getting the type of 'len' (line 1534)
    len_175787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 17), 'len', False)
    # Calling len(args, kwargs) (line 1534)
    len_call_result_175790 = invoke(stypy.reporting.localization.Localization(__file__, 1534, 17), len_175787, *[y_175788], **kwargs_175789)
    
    # Applying the binary operator '!=' (line 1534)
    result_ne_175791 = python_operator(stypy.reporting.localization.Localization(__file__, 1534, 7), '!=', len_call_result_175786, len_call_result_175790)
    
    # Testing the type of an if condition (line 1534)
    if_condition_175792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1534, 4), result_ne_175791)
    # Assigning a type to the variable 'if_condition_175792' (line 1534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1534, 4), 'if_condition_175792', if_condition_175792)
    # SSA begins for if statement (line 1534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1535)
    # Processing the call arguments (line 1535)
    str_175794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1535, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 1535)
    kwargs_175795 = {}
    # Getting the type of 'TypeError' (line 1535)
    TypeError_175793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1535)
    TypeError_call_result_175796 = invoke(stypy.reporting.localization.Localization(__file__, 1535, 14), TypeError_175793, *[str_175794], **kwargs_175795)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1535, 8), TypeError_call_result_175796, 'raise parameter', BaseException)
    # SSA join for if statement (line 1534)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'deg' (line 1537)
    deg_175797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1537)
    ndim_175798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 7), deg_175797, 'ndim')
    int_175799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1537, 19), 'int')
    # Applying the binary operator '==' (line 1537)
    result_eq_175800 = python_operator(stypy.reporting.localization.Localization(__file__, 1537, 7), '==', ndim_175798, int_175799)
    
    # Testing the type of an if condition (line 1537)
    if_condition_175801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1537, 4), result_eq_175800)
    # Assigning a type to the variable 'if_condition_175801' (line 1537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1537, 4), 'if_condition_175801', if_condition_175801)
    # SSA begins for if statement (line 1537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1538):
    
    # Assigning a Name to a Name (line 1538):
    # Getting the type of 'deg' (line 1538)
    deg_175802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 15), 'deg')
    # Assigning a type to the variable 'lmax' (line 1538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1538, 8), 'lmax', deg_175802)
    
    # Assigning a BinOp to a Name (line 1539):
    
    # Assigning a BinOp to a Name (line 1539):
    # Getting the type of 'lmax' (line 1539)
    lmax_175803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 16), 'lmax')
    int_175804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1539, 23), 'int')
    # Applying the binary operator '+' (line 1539)
    result_add_175805 = python_operator(stypy.reporting.localization.Localization(__file__, 1539, 16), '+', lmax_175803, int_175804)
    
    # Assigning a type to the variable 'order' (line 1539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1539, 8), 'order', result_add_175805)
    
    # Assigning a Call to a Name (line 1540):
    
    # Assigning a Call to a Name (line 1540):
    
    # Call to legvander(...): (line 1540)
    # Processing the call arguments (line 1540)
    # Getting the type of 'x' (line 1540)
    x_175807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 24), 'x', False)
    # Getting the type of 'lmax' (line 1540)
    lmax_175808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 27), 'lmax', False)
    # Processing the call keyword arguments (line 1540)
    kwargs_175809 = {}
    # Getting the type of 'legvander' (line 1540)
    legvander_175806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 14), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1540)
    legvander_call_result_175810 = invoke(stypy.reporting.localization.Localization(__file__, 1540, 14), legvander_175806, *[x_175807, lmax_175808], **kwargs_175809)
    
    # Assigning a type to the variable 'van' (line 1540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1540, 8), 'van', legvander_call_result_175810)
    # SSA branch for the else part of an if statement (line 1537)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1542):
    
    # Assigning a Call to a Name (line 1542):
    
    # Call to sort(...): (line 1542)
    # Processing the call arguments (line 1542)
    # Getting the type of 'deg' (line 1542)
    deg_175813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 22), 'deg', False)
    # Processing the call keyword arguments (line 1542)
    kwargs_175814 = {}
    # Getting the type of 'np' (line 1542)
    np_175811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 14), 'np', False)
    # Obtaining the member 'sort' of a type (line 1542)
    sort_175812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1542, 14), np_175811, 'sort')
    # Calling sort(args, kwargs) (line 1542)
    sort_call_result_175815 = invoke(stypy.reporting.localization.Localization(__file__, 1542, 14), sort_175812, *[deg_175813], **kwargs_175814)
    
    # Assigning a type to the variable 'deg' (line 1542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1542, 8), 'deg', sort_call_result_175815)
    
    # Assigning a Subscript to a Name (line 1543):
    
    # Assigning a Subscript to a Name (line 1543):
    
    # Obtaining the type of the subscript
    int_175816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 19), 'int')
    # Getting the type of 'deg' (line 1543)
    deg_175817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 15), 'deg')
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___175818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 15), deg_175817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1543)
    subscript_call_result_175819 = invoke(stypy.reporting.localization.Localization(__file__, 1543, 15), getitem___175818, int_175816)
    
    # Assigning a type to the variable 'lmax' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 8), 'lmax', subscript_call_result_175819)
    
    # Assigning a Call to a Name (line 1544):
    
    # Assigning a Call to a Name (line 1544):
    
    # Call to len(...): (line 1544)
    # Processing the call arguments (line 1544)
    # Getting the type of 'deg' (line 1544)
    deg_175821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 20), 'deg', False)
    # Processing the call keyword arguments (line 1544)
    kwargs_175822 = {}
    # Getting the type of 'len' (line 1544)
    len_175820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 16), 'len', False)
    # Calling len(args, kwargs) (line 1544)
    len_call_result_175823 = invoke(stypy.reporting.localization.Localization(__file__, 1544, 16), len_175820, *[deg_175821], **kwargs_175822)
    
    # Assigning a type to the variable 'order' (line 1544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1544, 8), 'order', len_call_result_175823)
    
    # Assigning a Subscript to a Name (line 1545):
    
    # Assigning a Subscript to a Name (line 1545):
    
    # Obtaining the type of the subscript
    slice_175824 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1545, 14), None, None, None)
    # Getting the type of 'deg' (line 1545)
    deg_175825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 36), 'deg')
    
    # Call to legvander(...): (line 1545)
    # Processing the call arguments (line 1545)
    # Getting the type of 'x' (line 1545)
    x_175827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 24), 'x', False)
    # Getting the type of 'lmax' (line 1545)
    lmax_175828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 27), 'lmax', False)
    # Processing the call keyword arguments (line 1545)
    kwargs_175829 = {}
    # Getting the type of 'legvander' (line 1545)
    legvander_175826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 14), 'legvander', False)
    # Calling legvander(args, kwargs) (line 1545)
    legvander_call_result_175830 = invoke(stypy.reporting.localization.Localization(__file__, 1545, 14), legvander_175826, *[x_175827, lmax_175828], **kwargs_175829)
    
    # Obtaining the member '__getitem__' of a type (line 1545)
    getitem___175831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 14), legvander_call_result_175830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1545)
    subscript_call_result_175832 = invoke(stypy.reporting.localization.Localization(__file__, 1545, 14), getitem___175831, (slice_175824, deg_175825))
    
    # Assigning a type to the variable 'van' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 8), 'van', subscript_call_result_175832)
    # SSA join for if statement (line 1537)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1548):
    
    # Assigning a Attribute to a Name (line 1548):
    # Getting the type of 'van' (line 1548)
    van_175833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1548, 10), 'van')
    # Obtaining the member 'T' of a type (line 1548)
    T_175834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1548, 10), van_175833, 'T')
    # Assigning a type to the variable 'lhs' (line 1548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1548, 4), 'lhs', T_175834)
    
    # Assigning a Attribute to a Name (line 1549):
    
    # Assigning a Attribute to a Name (line 1549):
    # Getting the type of 'y' (line 1549)
    y_175835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 10), 'y')
    # Obtaining the member 'T' of a type (line 1549)
    T_175836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 10), y_175835, 'T')
    # Assigning a type to the variable 'rhs' (line 1549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1549, 4), 'rhs', T_175836)
    
    # Type idiom detected: calculating its left and rigth part (line 1550)
    # Getting the type of 'w' (line 1550)
    w_175837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 4), 'w')
    # Getting the type of 'None' (line 1550)
    None_175838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 16), 'None')
    
    (may_be_175839, more_types_in_union_175840) = may_not_be_none(w_175837, None_175838)

    if may_be_175839:

        if more_types_in_union_175840:
            # Runtime conditional SSA (line 1550)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1551):
        
        # Assigning a BinOp to a Name (line 1551):
        
        # Call to asarray(...): (line 1551)
        # Processing the call arguments (line 1551)
        # Getting the type of 'w' (line 1551)
        w_175843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 23), 'w', False)
        # Processing the call keyword arguments (line 1551)
        kwargs_175844 = {}
        # Getting the type of 'np' (line 1551)
        np_175841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1551)
        asarray_175842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 12), np_175841, 'asarray')
        # Calling asarray(args, kwargs) (line 1551)
        asarray_call_result_175845 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 12), asarray_175842, *[w_175843], **kwargs_175844)
        
        float_175846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 28), 'float')
        # Applying the binary operator '+' (line 1551)
        result_add_175847 = python_operator(stypy.reporting.localization.Localization(__file__, 1551, 12), '+', asarray_call_result_175845, float_175846)
        
        # Assigning a type to the variable 'w' (line 1551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1551, 8), 'w', result_add_175847)
        
        
        # Getting the type of 'w' (line 1552)
        w_175848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 1552)
        ndim_175849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1552, 11), w_175848, 'ndim')
        int_175850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1552, 21), 'int')
        # Applying the binary operator '!=' (line 1552)
        result_ne_175851 = python_operator(stypy.reporting.localization.Localization(__file__, 1552, 11), '!=', ndim_175849, int_175850)
        
        # Testing the type of an if condition (line 1552)
        if_condition_175852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1552, 8), result_ne_175851)
        # Assigning a type to the variable 'if_condition_175852' (line 1552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1552, 8), 'if_condition_175852', if_condition_175852)
        # SSA begins for if statement (line 1552)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1553)
        # Processing the call arguments (line 1553)
        str_175854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1553, 28), 'str', 'expected 1D vector for w')
        # Processing the call keyword arguments (line 1553)
        kwargs_175855 = {}
        # Getting the type of 'TypeError' (line 1553)
        TypeError_175853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1553)
        TypeError_call_result_175856 = invoke(stypy.reporting.localization.Localization(__file__, 1553, 18), TypeError_175853, *[str_175854], **kwargs_175855)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1553, 12), TypeError_call_result_175856, 'raise parameter', BaseException)
        # SSA join for if statement (line 1552)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 1554)
        # Processing the call arguments (line 1554)
        # Getting the type of 'x' (line 1554)
        x_175858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 15), 'x', False)
        # Processing the call keyword arguments (line 1554)
        kwargs_175859 = {}
        # Getting the type of 'len' (line 1554)
        len_175857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 11), 'len', False)
        # Calling len(args, kwargs) (line 1554)
        len_call_result_175860 = invoke(stypy.reporting.localization.Localization(__file__, 1554, 11), len_175857, *[x_175858], **kwargs_175859)
        
        
        # Call to len(...): (line 1554)
        # Processing the call arguments (line 1554)
        # Getting the type of 'w' (line 1554)
        w_175862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 25), 'w', False)
        # Processing the call keyword arguments (line 1554)
        kwargs_175863 = {}
        # Getting the type of 'len' (line 1554)
        len_175861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 21), 'len', False)
        # Calling len(args, kwargs) (line 1554)
        len_call_result_175864 = invoke(stypy.reporting.localization.Localization(__file__, 1554, 21), len_175861, *[w_175862], **kwargs_175863)
        
        # Applying the binary operator '!=' (line 1554)
        result_ne_175865 = python_operator(stypy.reporting.localization.Localization(__file__, 1554, 11), '!=', len_call_result_175860, len_call_result_175864)
        
        # Testing the type of an if condition (line 1554)
        if_condition_175866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1554, 8), result_ne_175865)
        # Assigning a type to the variable 'if_condition_175866' (line 1554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1554, 8), 'if_condition_175866', if_condition_175866)
        # SSA begins for if statement (line 1554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1555)
        # Processing the call arguments (line 1555)
        str_175868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1555, 28), 'str', 'expected x and w to have same length')
        # Processing the call keyword arguments (line 1555)
        kwargs_175869 = {}
        # Getting the type of 'TypeError' (line 1555)
        TypeError_175867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1555)
        TypeError_call_result_175870 = invoke(stypy.reporting.localization.Localization(__file__, 1555, 18), TypeError_175867, *[str_175868], **kwargs_175869)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1555, 12), TypeError_call_result_175870, 'raise parameter', BaseException)
        # SSA join for if statement (line 1554)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1558):
        
        # Assigning a BinOp to a Name (line 1558):
        # Getting the type of 'lhs' (line 1558)
        lhs_175871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 14), 'lhs')
        # Getting the type of 'w' (line 1558)
        w_175872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 20), 'w')
        # Applying the binary operator '*' (line 1558)
        result_mul_175873 = python_operator(stypy.reporting.localization.Localization(__file__, 1558, 14), '*', lhs_175871, w_175872)
        
        # Assigning a type to the variable 'lhs' (line 1558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1558, 8), 'lhs', result_mul_175873)
        
        # Assigning a BinOp to a Name (line 1559):
        
        # Assigning a BinOp to a Name (line 1559):
        # Getting the type of 'rhs' (line 1559)
        rhs_175874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 14), 'rhs')
        # Getting the type of 'w' (line 1559)
        w_175875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 20), 'w')
        # Applying the binary operator '*' (line 1559)
        result_mul_175876 = python_operator(stypy.reporting.localization.Localization(__file__, 1559, 14), '*', rhs_175874, w_175875)
        
        # Assigning a type to the variable 'rhs' (line 1559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 8), 'rhs', result_mul_175876)

        if more_types_in_union_175840:
            # SSA join for if statement (line 1550)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1562)
    # Getting the type of 'rcond' (line 1562)
    rcond_175877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1562, 7), 'rcond')
    # Getting the type of 'None' (line 1562)
    None_175878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1562, 16), 'None')
    
    (may_be_175879, more_types_in_union_175880) = may_be_none(rcond_175877, None_175878)

    if may_be_175879:

        if more_types_in_union_175880:
            # Runtime conditional SSA (line 1562)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1563):
        
        # Assigning a BinOp to a Name (line 1563):
        
        # Call to len(...): (line 1563)
        # Processing the call arguments (line 1563)
        # Getting the type of 'x' (line 1563)
        x_175882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 20), 'x', False)
        # Processing the call keyword arguments (line 1563)
        kwargs_175883 = {}
        # Getting the type of 'len' (line 1563)
        len_175881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 16), 'len', False)
        # Calling len(args, kwargs) (line 1563)
        len_call_result_175884 = invoke(stypy.reporting.localization.Localization(__file__, 1563, 16), len_175881, *[x_175882], **kwargs_175883)
        
        
        # Call to finfo(...): (line 1563)
        # Processing the call arguments (line 1563)
        # Getting the type of 'x' (line 1563)
        x_175887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 32), 'x', False)
        # Obtaining the member 'dtype' of a type (line 1563)
        dtype_175888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1563, 32), x_175887, 'dtype')
        # Processing the call keyword arguments (line 1563)
        kwargs_175889 = {}
        # Getting the type of 'np' (line 1563)
        np_175885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 23), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1563)
        finfo_175886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1563, 23), np_175885, 'finfo')
        # Calling finfo(args, kwargs) (line 1563)
        finfo_call_result_175890 = invoke(stypy.reporting.localization.Localization(__file__, 1563, 23), finfo_175886, *[dtype_175888], **kwargs_175889)
        
        # Obtaining the member 'eps' of a type (line 1563)
        eps_175891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1563, 23), finfo_call_result_175890, 'eps')
        # Applying the binary operator '*' (line 1563)
        result_mul_175892 = python_operator(stypy.reporting.localization.Localization(__file__, 1563, 16), '*', len_call_result_175884, eps_175891)
        
        # Assigning a type to the variable 'rcond' (line 1563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1563, 8), 'rcond', result_mul_175892)

        if more_types_in_union_175880:
            # SSA join for if statement (line 1562)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issubclass(...): (line 1566)
    # Processing the call arguments (line 1566)
    # Getting the type of 'lhs' (line 1566)
    lhs_175894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 18), 'lhs', False)
    # Obtaining the member 'dtype' of a type (line 1566)
    dtype_175895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1566, 18), lhs_175894, 'dtype')
    # Obtaining the member 'type' of a type (line 1566)
    type_175896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1566, 18), dtype_175895, 'type')
    # Getting the type of 'np' (line 1566)
    np_175897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 34), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1566)
    complexfloating_175898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1566, 34), np_175897, 'complexfloating')
    # Processing the call keyword arguments (line 1566)
    kwargs_175899 = {}
    # Getting the type of 'issubclass' (line 1566)
    issubclass_175893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1566)
    issubclass_call_result_175900 = invoke(stypy.reporting.localization.Localization(__file__, 1566, 7), issubclass_175893, *[type_175896, complexfloating_175898], **kwargs_175899)
    
    # Testing the type of an if condition (line 1566)
    if_condition_175901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1566, 4), issubclass_call_result_175900)
    # Assigning a type to the variable 'if_condition_175901' (line 1566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1566, 4), 'if_condition_175901', if_condition_175901)
    # SSA begins for if statement (line 1566)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1567):
    
    # Assigning a Call to a Name (line 1567):
    
    # Call to sqrt(...): (line 1567)
    # Processing the call arguments (line 1567)
    
    # Call to sum(...): (line 1567)
    # Processing the call arguments (line 1567)
    int_175918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1567, 70), 'int')
    # Processing the call keyword arguments (line 1567)
    kwargs_175919 = {}
    
    # Call to square(...): (line 1567)
    # Processing the call arguments (line 1567)
    # Getting the type of 'lhs' (line 1567)
    lhs_175906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1567, 33), 'lhs', False)
    # Obtaining the member 'real' of a type (line 1567)
    real_175907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1567, 33), lhs_175906, 'real')
    # Processing the call keyword arguments (line 1567)
    kwargs_175908 = {}
    # Getting the type of 'np' (line 1567)
    np_175904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1567, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 1567)
    square_175905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1567, 23), np_175904, 'square')
    # Calling square(args, kwargs) (line 1567)
    square_call_result_175909 = invoke(stypy.reporting.localization.Localization(__file__, 1567, 23), square_175905, *[real_175907], **kwargs_175908)
    
    
    # Call to square(...): (line 1567)
    # Processing the call arguments (line 1567)
    # Getting the type of 'lhs' (line 1567)
    lhs_175912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1567, 55), 'lhs', False)
    # Obtaining the member 'imag' of a type (line 1567)
    imag_175913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1567, 55), lhs_175912, 'imag')
    # Processing the call keyword arguments (line 1567)
    kwargs_175914 = {}
    # Getting the type of 'np' (line 1567)
    np_175910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1567, 45), 'np', False)
    # Obtaining the member 'square' of a type (line 1567)
    square_175911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1567, 45), np_175910, 'square')
    # Calling square(args, kwargs) (line 1567)
    square_call_result_175915 = invoke(stypy.reporting.localization.Localization(__file__, 1567, 45), square_175911, *[imag_175913], **kwargs_175914)
    
    # Applying the binary operator '+' (line 1567)
    result_add_175916 = python_operator(stypy.reporting.localization.Localization(__file__, 1567, 23), '+', square_call_result_175909, square_call_result_175915)
    
    # Obtaining the member 'sum' of a type (line 1567)
    sum_175917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1567, 23), result_add_175916, 'sum')
    # Calling sum(args, kwargs) (line 1567)
    sum_call_result_175920 = invoke(stypy.reporting.localization.Localization(__file__, 1567, 23), sum_175917, *[int_175918], **kwargs_175919)
    
    # Processing the call keyword arguments (line 1567)
    kwargs_175921 = {}
    # Getting the type of 'np' (line 1567)
    np_175902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1567, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1567)
    sqrt_175903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1567, 14), np_175902, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1567)
    sqrt_call_result_175922 = invoke(stypy.reporting.localization.Localization(__file__, 1567, 14), sqrt_175903, *[sum_call_result_175920], **kwargs_175921)
    
    # Assigning a type to the variable 'scl' (line 1567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1567, 8), 'scl', sqrt_call_result_175922)
    # SSA branch for the else part of an if statement (line 1566)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1569):
    
    # Assigning a Call to a Name (line 1569):
    
    # Call to sqrt(...): (line 1569)
    # Processing the call arguments (line 1569)
    
    # Call to sum(...): (line 1569)
    # Processing the call arguments (line 1569)
    int_175931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1569, 41), 'int')
    # Processing the call keyword arguments (line 1569)
    kwargs_175932 = {}
    
    # Call to square(...): (line 1569)
    # Processing the call arguments (line 1569)
    # Getting the type of 'lhs' (line 1569)
    lhs_175927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1569, 32), 'lhs', False)
    # Processing the call keyword arguments (line 1569)
    kwargs_175928 = {}
    # Getting the type of 'np' (line 1569)
    np_175925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1569, 22), 'np', False)
    # Obtaining the member 'square' of a type (line 1569)
    square_175926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1569, 22), np_175925, 'square')
    # Calling square(args, kwargs) (line 1569)
    square_call_result_175929 = invoke(stypy.reporting.localization.Localization(__file__, 1569, 22), square_175926, *[lhs_175927], **kwargs_175928)
    
    # Obtaining the member 'sum' of a type (line 1569)
    sum_175930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1569, 22), square_call_result_175929, 'sum')
    # Calling sum(args, kwargs) (line 1569)
    sum_call_result_175933 = invoke(stypy.reporting.localization.Localization(__file__, 1569, 22), sum_175930, *[int_175931], **kwargs_175932)
    
    # Processing the call keyword arguments (line 1569)
    kwargs_175934 = {}
    # Getting the type of 'np' (line 1569)
    np_175923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1569, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1569)
    sqrt_175924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1569, 14), np_175923, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1569)
    sqrt_call_result_175935 = invoke(stypy.reporting.localization.Localization(__file__, 1569, 14), sqrt_175924, *[sum_call_result_175933], **kwargs_175934)
    
    # Assigning a type to the variable 'scl' (line 1569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1569, 8), 'scl', sqrt_call_result_175935)
    # SSA join for if statement (line 1566)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 1570):
    
    # Assigning a Num to a Subscript (line 1570):
    int_175936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1570, 20), 'int')
    # Getting the type of 'scl' (line 1570)
    scl_175937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1570, 4), 'scl')
    
    # Getting the type of 'scl' (line 1570)
    scl_175938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1570, 8), 'scl')
    int_175939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1570, 15), 'int')
    # Applying the binary operator '==' (line 1570)
    result_eq_175940 = python_operator(stypy.reporting.localization.Localization(__file__, 1570, 8), '==', scl_175938, int_175939)
    
    # Storing an element on a container (line 1570)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1570, 4), scl_175937, (result_eq_175940, int_175936))
    
    # Assigning a Call to a Tuple (line 1573):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 1573)
    # Processing the call arguments (line 1573)
    # Getting the type of 'lhs' (line 1573)
    lhs_175943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 34), 'lhs', False)
    # Obtaining the member 'T' of a type (line 1573)
    T_175944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 34), lhs_175943, 'T')
    # Getting the type of 'scl' (line 1573)
    scl_175945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 40), 'scl', False)
    # Applying the binary operator 'div' (line 1573)
    result_div_175946 = python_operator(stypy.reporting.localization.Localization(__file__, 1573, 34), 'div', T_175944, scl_175945)
    
    # Getting the type of 'rhs' (line 1573)
    rhs_175947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 45), 'rhs', False)
    # Obtaining the member 'T' of a type (line 1573)
    T_175948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 45), rhs_175947, 'T')
    # Getting the type of 'rcond' (line 1573)
    rcond_175949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 52), 'rcond', False)
    # Processing the call keyword arguments (line 1573)
    kwargs_175950 = {}
    # Getting the type of 'la' (line 1573)
    la_175941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 25), 'la', False)
    # Obtaining the member 'lstsq' of a type (line 1573)
    lstsq_175942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 25), la_175941, 'lstsq')
    # Calling lstsq(args, kwargs) (line 1573)
    lstsq_call_result_175951 = invoke(stypy.reporting.localization.Localization(__file__, 1573, 25), lstsq_175942, *[result_div_175946, T_175948, rcond_175949], **kwargs_175950)
    
    # Assigning a type to the variable 'call_assignment_173649' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173649', lstsq_call_result_175951)
    
    # Assigning a Call to a Name (line 1573):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 4), 'int')
    # Processing the call keyword arguments
    kwargs_175955 = {}
    # Getting the type of 'call_assignment_173649' (line 1573)
    call_assignment_173649_175952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173649', False)
    # Obtaining the member '__getitem__' of a type (line 1573)
    getitem___175953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 4), call_assignment_173649_175952, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175956 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175953, *[int_175954], **kwargs_175955)
    
    # Assigning a type to the variable 'call_assignment_173650' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173650', getitem___call_result_175956)
    
    # Assigning a Name to a Name (line 1573):
    # Getting the type of 'call_assignment_173650' (line 1573)
    call_assignment_173650_175957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173650')
    # Assigning a type to the variable 'c' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'c', call_assignment_173650_175957)
    
    # Assigning a Call to a Name (line 1573):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 4), 'int')
    # Processing the call keyword arguments
    kwargs_175961 = {}
    # Getting the type of 'call_assignment_173649' (line 1573)
    call_assignment_173649_175958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173649', False)
    # Obtaining the member '__getitem__' of a type (line 1573)
    getitem___175959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 4), call_assignment_173649_175958, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175962 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175959, *[int_175960], **kwargs_175961)
    
    # Assigning a type to the variable 'call_assignment_173651' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173651', getitem___call_result_175962)
    
    # Assigning a Name to a Name (line 1573):
    # Getting the type of 'call_assignment_173651' (line 1573)
    call_assignment_173651_175963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173651')
    # Assigning a type to the variable 'resids' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 7), 'resids', call_assignment_173651_175963)
    
    # Assigning a Call to a Name (line 1573):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 4), 'int')
    # Processing the call keyword arguments
    kwargs_175967 = {}
    # Getting the type of 'call_assignment_173649' (line 1573)
    call_assignment_173649_175964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173649', False)
    # Obtaining the member '__getitem__' of a type (line 1573)
    getitem___175965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 4), call_assignment_173649_175964, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175968 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175965, *[int_175966], **kwargs_175967)
    
    # Assigning a type to the variable 'call_assignment_173652' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173652', getitem___call_result_175968)
    
    # Assigning a Name to a Name (line 1573):
    # Getting the type of 'call_assignment_173652' (line 1573)
    call_assignment_173652_175969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173652')
    # Assigning a type to the variable 'rank' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 15), 'rank', call_assignment_173652_175969)
    
    # Assigning a Call to a Name (line 1573):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_175972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 4), 'int')
    # Processing the call keyword arguments
    kwargs_175973 = {}
    # Getting the type of 'call_assignment_173649' (line 1573)
    call_assignment_173649_175970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173649', False)
    # Obtaining the member '__getitem__' of a type (line 1573)
    getitem___175971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 4), call_assignment_173649_175970, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_175974 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___175971, *[int_175972], **kwargs_175973)
    
    # Assigning a type to the variable 'call_assignment_173653' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173653', getitem___call_result_175974)
    
    # Assigning a Name to a Name (line 1573):
    # Getting the type of 'call_assignment_173653' (line 1573)
    call_assignment_173653_175975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'call_assignment_173653')
    # Assigning a type to the variable 's' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 21), 's', call_assignment_173653_175975)
    
    # Assigning a Attribute to a Name (line 1574):
    
    # Assigning a Attribute to a Name (line 1574):
    # Getting the type of 'c' (line 1574)
    c_175976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 9), 'c')
    # Obtaining the member 'T' of a type (line 1574)
    T_175977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1574, 9), c_175976, 'T')
    # Getting the type of 'scl' (line 1574)
    scl_175978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 13), 'scl')
    # Applying the binary operator 'div' (line 1574)
    result_div_175979 = python_operator(stypy.reporting.localization.Localization(__file__, 1574, 9), 'div', T_175977, scl_175978)
    
    # Obtaining the member 'T' of a type (line 1574)
    T_175980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1574, 9), result_div_175979, 'T')
    # Assigning a type to the variable 'c' (line 1574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1574, 4), 'c', T_175980)
    
    
    # Getting the type of 'deg' (line 1577)
    deg_175981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1577, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1577)
    ndim_175982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1577, 7), deg_175981, 'ndim')
    int_175983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1577, 18), 'int')
    # Applying the binary operator '>' (line 1577)
    result_gt_175984 = python_operator(stypy.reporting.localization.Localization(__file__, 1577, 7), '>', ndim_175982, int_175983)
    
    # Testing the type of an if condition (line 1577)
    if_condition_175985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1577, 4), result_gt_175984)
    # Assigning a type to the variable 'if_condition_175985' (line 1577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1577, 4), 'if_condition_175985', if_condition_175985)
    # SSA begins for if statement (line 1577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'c' (line 1578)
    c_175986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1578, 11), 'c')
    # Obtaining the member 'ndim' of a type (line 1578)
    ndim_175987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1578, 11), c_175986, 'ndim')
    int_175988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1578, 21), 'int')
    # Applying the binary operator '==' (line 1578)
    result_eq_175989 = python_operator(stypy.reporting.localization.Localization(__file__, 1578, 11), '==', ndim_175987, int_175988)
    
    # Testing the type of an if condition (line 1578)
    if_condition_175990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1578, 8), result_eq_175989)
    # Assigning a type to the variable 'if_condition_175990' (line 1578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1578, 8), 'if_condition_175990', if_condition_175990)
    # SSA begins for if statement (line 1578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1579):
    
    # Assigning a Call to a Name (line 1579):
    
    # Call to zeros(...): (line 1579)
    # Processing the call arguments (line 1579)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1579)
    tuple_175993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1579, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1579)
    # Adding element type (line 1579)
    # Getting the type of 'lmax' (line 1579)
    lmax_175994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 27), 'lmax', False)
    int_175995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1579, 32), 'int')
    # Applying the binary operator '+' (line 1579)
    result_add_175996 = python_operator(stypy.reporting.localization.Localization(__file__, 1579, 27), '+', lmax_175994, int_175995)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1579, 27), tuple_175993, result_add_175996)
    # Adding element type (line 1579)
    
    # Obtaining the type of the subscript
    int_175997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1579, 43), 'int')
    # Getting the type of 'c' (line 1579)
    c_175998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 35), 'c', False)
    # Obtaining the member 'shape' of a type (line 1579)
    shape_175999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 35), c_175998, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1579)
    getitem___176000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 35), shape_175999, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1579)
    subscript_call_result_176001 = invoke(stypy.reporting.localization.Localization(__file__, 1579, 35), getitem___176000, int_175997)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1579, 27), tuple_175993, subscript_call_result_176001)
    
    # Processing the call keyword arguments (line 1579)
    # Getting the type of 'c' (line 1579)
    c_176002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 54), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1579)
    dtype_176003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 54), c_176002, 'dtype')
    keyword_176004 = dtype_176003
    kwargs_176005 = {'dtype': keyword_176004}
    # Getting the type of 'np' (line 1579)
    np_175991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1579)
    zeros_175992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 17), np_175991, 'zeros')
    # Calling zeros(args, kwargs) (line 1579)
    zeros_call_result_176006 = invoke(stypy.reporting.localization.Localization(__file__, 1579, 17), zeros_175992, *[tuple_175993], **kwargs_176005)
    
    # Assigning a type to the variable 'cc' (line 1579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1579, 12), 'cc', zeros_call_result_176006)
    # SSA branch for the else part of an if statement (line 1578)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1581):
    
    # Assigning a Call to a Name (line 1581):
    
    # Call to zeros(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'lmax' (line 1581)
    lmax_176009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'lmax', False)
    int_176010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 31), 'int')
    # Applying the binary operator '+' (line 1581)
    result_add_176011 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 26), '+', lmax_176009, int_176010)
    
    # Processing the call keyword arguments (line 1581)
    # Getting the type of 'c' (line 1581)
    c_176012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 40), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1581)
    dtype_176013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 40), c_176012, 'dtype')
    keyword_176014 = dtype_176013
    kwargs_176015 = {'dtype': keyword_176014}
    # Getting the type of 'np' (line 1581)
    np_176007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1581)
    zeros_176008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 17), np_176007, 'zeros')
    # Calling zeros(args, kwargs) (line 1581)
    zeros_call_result_176016 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 17), zeros_176008, *[result_add_176011], **kwargs_176015)
    
    # Assigning a type to the variable 'cc' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 12), 'cc', zeros_call_result_176016)
    # SSA join for if statement (line 1578)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1582):
    
    # Assigning a Name to a Subscript (line 1582):
    # Getting the type of 'c' (line 1582)
    c_176017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 18), 'c')
    # Getting the type of 'cc' (line 1582)
    cc_176018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 8), 'cc')
    # Getting the type of 'deg' (line 1582)
    deg_176019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 11), 'deg')
    # Storing an element on a container (line 1582)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1582, 8), cc_176018, (deg_176019, c_176017))
    
    # Assigning a Name to a Name (line 1583):
    
    # Assigning a Name to a Name (line 1583):
    # Getting the type of 'cc' (line 1583)
    cc_176020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 12), 'cc')
    # Assigning a type to the variable 'c' (line 1583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1583, 8), 'c', cc_176020)
    # SSA join for if statement (line 1577)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1586)
    rank_176021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 7), 'rank')
    # Getting the type of 'order' (line 1586)
    order_176022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 15), 'order')
    # Applying the binary operator '!=' (line 1586)
    result_ne_176023 = python_operator(stypy.reporting.localization.Localization(__file__, 1586, 7), '!=', rank_176021, order_176022)
    
    
    # Getting the type of 'full' (line 1586)
    full_176024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 29), 'full')
    # Applying the 'not' unary operator (line 1586)
    result_not__176025 = python_operator(stypy.reporting.localization.Localization(__file__, 1586, 25), 'not', full_176024)
    
    # Applying the binary operator 'and' (line 1586)
    result_and_keyword_176026 = python_operator(stypy.reporting.localization.Localization(__file__, 1586, 7), 'and', result_ne_176023, result_not__176025)
    
    # Testing the type of an if condition (line 1586)
    if_condition_176027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1586, 4), result_and_keyword_176026)
    # Assigning a type to the variable 'if_condition_176027' (line 1586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1586, 4), 'if_condition_176027', if_condition_176027)
    # SSA begins for if statement (line 1586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1587):
    
    # Assigning a Str to a Name (line 1587):
    str_176028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1587, 14), 'str', 'The fit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 8), 'msg', str_176028)
    
    # Call to warn(...): (line 1588)
    # Processing the call arguments (line 1588)
    # Getting the type of 'msg' (line 1588)
    msg_176031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 22), 'msg', False)
    # Getting the type of 'pu' (line 1588)
    pu_176032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 27), 'pu', False)
    # Obtaining the member 'RankWarning' of a type (line 1588)
    RankWarning_176033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 27), pu_176032, 'RankWarning')
    # Processing the call keyword arguments (line 1588)
    kwargs_176034 = {}
    # Getting the type of 'warnings' (line 1588)
    warnings_176029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1588)
    warn_176030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 8), warnings_176029, 'warn')
    # Calling warn(args, kwargs) (line 1588)
    warn_call_result_176035 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 8), warn_176030, *[msg_176031, RankWarning_176033], **kwargs_176034)
    
    # SSA join for if statement (line 1586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 1590)
    full_176036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 7), 'full')
    # Testing the type of an if condition (line 1590)
    if_condition_176037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1590, 4), full_176036)
    # Assigning a type to the variable 'if_condition_176037' (line 1590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1590, 4), 'if_condition_176037', if_condition_176037)
    # SSA begins for if statement (line 1590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1591)
    tuple_176038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1591, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1591)
    # Adding element type (line 1591)
    # Getting the type of 'c' (line 1591)
    c_176039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1591, 15), tuple_176038, c_176039)
    # Adding element type (line 1591)
    
    # Obtaining an instance of the builtin type 'list' (line 1591)
    list_176040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1591, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1591)
    # Adding element type (line 1591)
    # Getting the type of 'resids' (line 1591)
    resids_176041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 19), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1591, 18), list_176040, resids_176041)
    # Adding element type (line 1591)
    # Getting the type of 'rank' (line 1591)
    rank_176042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 27), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1591, 18), list_176040, rank_176042)
    # Adding element type (line 1591)
    # Getting the type of 's' (line 1591)
    s_176043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 33), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1591, 18), list_176040, s_176043)
    # Adding element type (line 1591)
    # Getting the type of 'rcond' (line 1591)
    rcond_176044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 36), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1591, 18), list_176040, rcond_176044)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1591, 15), tuple_176038, list_176040)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1591, 8), 'stypy_return_type', tuple_176038)
    # SSA branch for the else part of an if statement (line 1590)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 1593)
    c_176045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1593, 8), 'stypy_return_type', c_176045)
    # SSA join for if statement (line 1590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'legfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legfit' in the type store
    # Getting the type of 'stypy_return_type' (line 1398)
    stypy_return_type_176046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176046)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legfit'
    return stypy_return_type_176046

# Assigning a type to the variable 'legfit' (line 1398)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1398, 0), 'legfit', legfit)

@norecursion
def legcompanion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legcompanion'
    module_type_store = module_type_store.open_function_context('legcompanion', 1596, 0, False)
    
    # Passed parameters checking function
    legcompanion.stypy_localization = localization
    legcompanion.stypy_type_of_self = None
    legcompanion.stypy_type_store = module_type_store
    legcompanion.stypy_function_name = 'legcompanion'
    legcompanion.stypy_param_names_list = ['c']
    legcompanion.stypy_varargs_param_name = None
    legcompanion.stypy_kwargs_param_name = None
    legcompanion.stypy_call_defaults = defaults
    legcompanion.stypy_call_varargs = varargs
    legcompanion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legcompanion', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legcompanion', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legcompanion(...)' code ##################

    str_176047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1621, (-1)), 'str', 'Return the scaled companion matrix of c.\n\n    The basis polynomials are scaled so that the companion matrix is\n    symmetric when `c` is an Legendre basis polynomial. This provides\n    better eigenvalue estimates than the unscaled case and for basis\n    polynomials the eigenvalues are guaranteed to be real if\n    `numpy.linalg.eigvalsh` is used to obtain them.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Legendre series coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Scaled companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a List (line 1623):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1623)
    # Processing the call arguments (line 1623)
    
    # Obtaining an instance of the builtin type 'list' (line 1623)
    list_176050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1623, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1623)
    # Adding element type (line 1623)
    # Getting the type of 'c' (line 1623)
    c_176051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1623, 23), list_176050, c_176051)
    
    # Processing the call keyword arguments (line 1623)
    kwargs_176052 = {}
    # Getting the type of 'pu' (line 1623)
    pu_176048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1623)
    as_series_176049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1623, 10), pu_176048, 'as_series')
    # Calling as_series(args, kwargs) (line 1623)
    as_series_call_result_176053 = invoke(stypy.reporting.localization.Localization(__file__, 1623, 10), as_series_176049, *[list_176050], **kwargs_176052)
    
    # Assigning a type to the variable 'call_assignment_173654' (line 1623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1623, 4), 'call_assignment_173654', as_series_call_result_176053)
    
    # Assigning a Call to a Name (line 1623):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1623, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176057 = {}
    # Getting the type of 'call_assignment_173654' (line 1623)
    call_assignment_173654_176054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 4), 'call_assignment_173654', False)
    # Obtaining the member '__getitem__' of a type (line 1623)
    getitem___176055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1623, 4), call_assignment_173654_176054, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176058 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176055, *[int_176056], **kwargs_176057)
    
    # Assigning a type to the variable 'call_assignment_173655' (line 1623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1623, 4), 'call_assignment_173655', getitem___call_result_176058)
    
    # Assigning a Name to a Name (line 1623):
    # Getting the type of 'call_assignment_173655' (line 1623)
    call_assignment_173655_176059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 4), 'call_assignment_173655')
    # Assigning a type to the variable 'c' (line 1623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1623, 5), 'c', call_assignment_173655_176059)
    
    
    
    # Call to len(...): (line 1624)
    # Processing the call arguments (line 1624)
    # Getting the type of 'c' (line 1624)
    c_176061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 11), 'c', False)
    # Processing the call keyword arguments (line 1624)
    kwargs_176062 = {}
    # Getting the type of 'len' (line 1624)
    len_176060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 7), 'len', False)
    # Calling len(args, kwargs) (line 1624)
    len_call_result_176063 = invoke(stypy.reporting.localization.Localization(__file__, 1624, 7), len_176060, *[c_176061], **kwargs_176062)
    
    int_176064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1624, 16), 'int')
    # Applying the binary operator '<' (line 1624)
    result_lt_176065 = python_operator(stypy.reporting.localization.Localization(__file__, 1624, 7), '<', len_call_result_176063, int_176064)
    
    # Testing the type of an if condition (line 1624)
    if_condition_176066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1624, 4), result_lt_176065)
    # Assigning a type to the variable 'if_condition_176066' (line 1624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1624, 4), 'if_condition_176066', if_condition_176066)
    # SSA begins for if statement (line 1624)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1625)
    # Processing the call arguments (line 1625)
    str_176068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1625, 25), 'str', 'Series must have maximum degree of at least 1.')
    # Processing the call keyword arguments (line 1625)
    kwargs_176069 = {}
    # Getting the type of 'ValueError' (line 1625)
    ValueError_176067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1625, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1625)
    ValueError_call_result_176070 = invoke(stypy.reporting.localization.Localization(__file__, 1625, 14), ValueError_176067, *[str_176068], **kwargs_176069)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1625, 8), ValueError_call_result_176070, 'raise parameter', BaseException)
    # SSA join for if statement (line 1624)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1626)
    # Processing the call arguments (line 1626)
    # Getting the type of 'c' (line 1626)
    c_176072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 11), 'c', False)
    # Processing the call keyword arguments (line 1626)
    kwargs_176073 = {}
    # Getting the type of 'len' (line 1626)
    len_176071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 7), 'len', False)
    # Calling len(args, kwargs) (line 1626)
    len_call_result_176074 = invoke(stypy.reporting.localization.Localization(__file__, 1626, 7), len_176071, *[c_176072], **kwargs_176073)
    
    int_176075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1626, 17), 'int')
    # Applying the binary operator '==' (line 1626)
    result_eq_176076 = python_operator(stypy.reporting.localization.Localization(__file__, 1626, 7), '==', len_call_result_176074, int_176075)
    
    # Testing the type of an if condition (line 1626)
    if_condition_176077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1626, 4), result_eq_176076)
    # Assigning a type to the variable 'if_condition_176077' (line 1626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1626, 4), 'if_condition_176077', if_condition_176077)
    # SSA begins for if statement (line 1626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1627)
    # Processing the call arguments (line 1627)
    
    # Obtaining an instance of the builtin type 'list' (line 1627)
    list_176080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1627, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1627)
    # Adding element type (line 1627)
    
    # Obtaining an instance of the builtin type 'list' (line 1627)
    list_176081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1627, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1627)
    # Adding element type (line 1627)
    
    
    # Obtaining the type of the subscript
    int_176082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1627, 29), 'int')
    # Getting the type of 'c' (line 1627)
    c_176083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 27), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1627)
    getitem___176084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1627, 27), c_176083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1627)
    subscript_call_result_176085 = invoke(stypy.reporting.localization.Localization(__file__, 1627, 27), getitem___176084, int_176082)
    
    # Applying the 'usub' unary operator (line 1627)
    result___neg___176086 = python_operator(stypy.reporting.localization.Localization(__file__, 1627, 26), 'usub', subscript_call_result_176085)
    
    
    # Obtaining the type of the subscript
    int_176087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1627, 34), 'int')
    # Getting the type of 'c' (line 1627)
    c_176088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 32), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1627)
    getitem___176089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1627, 32), c_176088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1627)
    subscript_call_result_176090 = invoke(stypy.reporting.localization.Localization(__file__, 1627, 32), getitem___176089, int_176087)
    
    # Applying the binary operator 'div' (line 1627)
    result_div_176091 = python_operator(stypy.reporting.localization.Localization(__file__, 1627, 26), 'div', result___neg___176086, subscript_call_result_176090)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1627, 25), list_176081, result_div_176091)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1627, 24), list_176080, list_176081)
    
    # Processing the call keyword arguments (line 1627)
    kwargs_176092 = {}
    # Getting the type of 'np' (line 1627)
    np_176078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1627)
    array_176079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1627, 15), np_176078, 'array')
    # Calling array(args, kwargs) (line 1627)
    array_call_result_176093 = invoke(stypy.reporting.localization.Localization(__file__, 1627, 15), array_176079, *[list_176080], **kwargs_176092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1627, 8), 'stypy_return_type', array_call_result_176093)
    # SSA join for if statement (line 1626)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1629):
    
    # Assigning a BinOp to a Name (line 1629):
    
    # Call to len(...): (line 1629)
    # Processing the call arguments (line 1629)
    # Getting the type of 'c' (line 1629)
    c_176095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 12), 'c', False)
    # Processing the call keyword arguments (line 1629)
    kwargs_176096 = {}
    # Getting the type of 'len' (line 1629)
    len_176094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 8), 'len', False)
    # Calling len(args, kwargs) (line 1629)
    len_call_result_176097 = invoke(stypy.reporting.localization.Localization(__file__, 1629, 8), len_176094, *[c_176095], **kwargs_176096)
    
    int_176098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1629, 17), 'int')
    # Applying the binary operator '-' (line 1629)
    result_sub_176099 = python_operator(stypy.reporting.localization.Localization(__file__, 1629, 8), '-', len_call_result_176097, int_176098)
    
    # Assigning a type to the variable 'n' (line 1629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1629, 4), 'n', result_sub_176099)
    
    # Assigning a Call to a Name (line 1630):
    
    # Assigning a Call to a Name (line 1630):
    
    # Call to zeros(...): (line 1630)
    # Processing the call arguments (line 1630)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1630)
    tuple_176102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1630, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1630)
    # Adding element type (line 1630)
    # Getting the type of 'n' (line 1630)
    n_176103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1630, 20), tuple_176102, n_176103)
    # Adding element type (line 1630)
    # Getting the type of 'n' (line 1630)
    n_176104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1630, 20), tuple_176102, n_176104)
    
    # Processing the call keyword arguments (line 1630)
    # Getting the type of 'c' (line 1630)
    c_176105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 33), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1630)
    dtype_176106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1630, 33), c_176105, 'dtype')
    keyword_176107 = dtype_176106
    kwargs_176108 = {'dtype': keyword_176107}
    # Getting the type of 'np' (line 1630)
    np_176100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1630)
    zeros_176101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1630, 10), np_176100, 'zeros')
    # Calling zeros(args, kwargs) (line 1630)
    zeros_call_result_176109 = invoke(stypy.reporting.localization.Localization(__file__, 1630, 10), zeros_176101, *[tuple_176102], **kwargs_176108)
    
    # Assigning a type to the variable 'mat' (line 1630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1630, 4), 'mat', zeros_call_result_176109)
    
    # Assigning a BinOp to a Name (line 1631):
    
    # Assigning a BinOp to a Name (line 1631):
    float_176110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1631, 10), 'float')
    
    # Call to sqrt(...): (line 1631)
    # Processing the call arguments (line 1631)
    int_176113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1631, 21), 'int')
    
    # Call to arange(...): (line 1631)
    # Processing the call arguments (line 1631)
    # Getting the type of 'n' (line 1631)
    n_176116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 33), 'n', False)
    # Processing the call keyword arguments (line 1631)
    kwargs_176117 = {}
    # Getting the type of 'np' (line 1631)
    np_176114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 1631)
    arange_176115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1631, 23), np_176114, 'arange')
    # Calling arange(args, kwargs) (line 1631)
    arange_call_result_176118 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 23), arange_176115, *[n_176116], **kwargs_176117)
    
    # Applying the binary operator '*' (line 1631)
    result_mul_176119 = python_operator(stypy.reporting.localization.Localization(__file__, 1631, 21), '*', int_176113, arange_call_result_176118)
    
    int_176120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1631, 38), 'int')
    # Applying the binary operator '+' (line 1631)
    result_add_176121 = python_operator(stypy.reporting.localization.Localization(__file__, 1631, 21), '+', result_mul_176119, int_176120)
    
    # Processing the call keyword arguments (line 1631)
    kwargs_176122 = {}
    # Getting the type of 'np' (line 1631)
    np_176111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 13), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1631)
    sqrt_176112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1631, 13), np_176111, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1631)
    sqrt_call_result_176123 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 13), sqrt_176112, *[result_add_176121], **kwargs_176122)
    
    # Applying the binary operator 'div' (line 1631)
    result_div_176124 = python_operator(stypy.reporting.localization.Localization(__file__, 1631, 10), 'div', float_176110, sqrt_call_result_176123)
    
    # Assigning a type to the variable 'scl' (line 1631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 4), 'scl', result_div_176124)
    
    # Assigning a Subscript to a Name (line 1632):
    
    # Assigning a Subscript to a Name (line 1632):
    
    # Obtaining the type of the subscript
    int_176125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1632, 26), 'int')
    # Getting the type of 'n' (line 1632)
    n_176126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1632, 29), 'n')
    int_176127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1632, 31), 'int')
    # Applying the binary operator '+' (line 1632)
    result_add_176128 = python_operator(stypy.reporting.localization.Localization(__file__, 1632, 29), '+', n_176126, int_176127)
    
    slice_176129 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1632, 10), int_176125, None, result_add_176128)
    
    # Call to reshape(...): (line 1632)
    # Processing the call arguments (line 1632)
    int_176132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1632, 22), 'int')
    # Processing the call keyword arguments (line 1632)
    kwargs_176133 = {}
    # Getting the type of 'mat' (line 1632)
    mat_176130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1632, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1632)
    reshape_176131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1632, 10), mat_176130, 'reshape')
    # Calling reshape(args, kwargs) (line 1632)
    reshape_call_result_176134 = invoke(stypy.reporting.localization.Localization(__file__, 1632, 10), reshape_176131, *[int_176132], **kwargs_176133)
    
    # Obtaining the member '__getitem__' of a type (line 1632)
    getitem___176135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1632, 10), reshape_call_result_176134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1632)
    subscript_call_result_176136 = invoke(stypy.reporting.localization.Localization(__file__, 1632, 10), getitem___176135, slice_176129)
    
    # Assigning a type to the variable 'top' (line 1632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1632, 4), 'top', subscript_call_result_176136)
    
    # Assigning a Subscript to a Name (line 1633):
    
    # Assigning a Subscript to a Name (line 1633):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1633)
    n_176137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 26), 'n')
    # Getting the type of 'n' (line 1633)
    n_176138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 29), 'n')
    int_176139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1633, 31), 'int')
    # Applying the binary operator '+' (line 1633)
    result_add_176140 = python_operator(stypy.reporting.localization.Localization(__file__, 1633, 29), '+', n_176138, int_176139)
    
    slice_176141 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1633, 10), n_176137, None, result_add_176140)
    
    # Call to reshape(...): (line 1633)
    # Processing the call arguments (line 1633)
    int_176144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1633, 22), 'int')
    # Processing the call keyword arguments (line 1633)
    kwargs_176145 = {}
    # Getting the type of 'mat' (line 1633)
    mat_176142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1633)
    reshape_176143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1633, 10), mat_176142, 'reshape')
    # Calling reshape(args, kwargs) (line 1633)
    reshape_call_result_176146 = invoke(stypy.reporting.localization.Localization(__file__, 1633, 10), reshape_176143, *[int_176144], **kwargs_176145)
    
    # Obtaining the member '__getitem__' of a type (line 1633)
    getitem___176147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1633, 10), reshape_call_result_176146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1633)
    subscript_call_result_176148 = invoke(stypy.reporting.localization.Localization(__file__, 1633, 10), getitem___176147, slice_176141)
    
    # Assigning a type to the variable 'bot' (line 1633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1633, 4), 'bot', subscript_call_result_176148)
    
    # Assigning a BinOp to a Subscript (line 1634):
    
    # Assigning a BinOp to a Subscript (line 1634):
    
    # Call to arange(...): (line 1634)
    # Processing the call arguments (line 1634)
    int_176151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 25), 'int')
    # Getting the type of 'n' (line 1634)
    n_176152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 28), 'n', False)
    # Processing the call keyword arguments (line 1634)
    kwargs_176153 = {}
    # Getting the type of 'np' (line 1634)
    np_176149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 15), 'np', False)
    # Obtaining the member 'arange' of a type (line 1634)
    arange_176150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1634, 15), np_176149, 'arange')
    # Calling arange(args, kwargs) (line 1634)
    arange_call_result_176154 = invoke(stypy.reporting.localization.Localization(__file__, 1634, 15), arange_176150, *[int_176151, n_176152], **kwargs_176153)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1634)
    n_176155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 36), 'n')
    int_176156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 38), 'int')
    # Applying the binary operator '-' (line 1634)
    result_sub_176157 = python_operator(stypy.reporting.localization.Localization(__file__, 1634, 36), '-', n_176155, int_176156)
    
    slice_176158 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1634, 31), None, result_sub_176157, None)
    # Getting the type of 'scl' (line 1634)
    scl_176159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 31), 'scl')
    # Obtaining the member '__getitem__' of a type (line 1634)
    getitem___176160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1634, 31), scl_176159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1634)
    subscript_call_result_176161 = invoke(stypy.reporting.localization.Localization(__file__, 1634, 31), getitem___176160, slice_176158)
    
    # Applying the binary operator '*' (line 1634)
    result_mul_176162 = python_operator(stypy.reporting.localization.Localization(__file__, 1634, 15), '*', arange_call_result_176154, subscript_call_result_176161)
    
    
    # Obtaining the type of the subscript
    int_176163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 45), 'int')
    # Getting the type of 'n' (line 1634)
    n_176164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 47), 'n')
    slice_176165 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1634, 41), int_176163, n_176164, None)
    # Getting the type of 'scl' (line 1634)
    scl_176166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 41), 'scl')
    # Obtaining the member '__getitem__' of a type (line 1634)
    getitem___176167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1634, 41), scl_176166, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1634)
    subscript_call_result_176168 = invoke(stypy.reporting.localization.Localization(__file__, 1634, 41), getitem___176167, slice_176165)
    
    # Applying the binary operator '*' (line 1634)
    result_mul_176169 = python_operator(stypy.reporting.localization.Localization(__file__, 1634, 40), '*', result_mul_176162, subscript_call_result_176168)
    
    # Getting the type of 'top' (line 1634)
    top_176170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 4), 'top')
    Ellipsis_176171 = Ellipsis
    # Storing an element on a container (line 1634)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1634, 4), top_176170, (Ellipsis_176171, result_mul_176169))
    
    # Assigning a Name to a Subscript (line 1635):
    
    # Assigning a Name to a Subscript (line 1635):
    # Getting the type of 'top' (line 1635)
    top_176172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 15), 'top')
    # Getting the type of 'bot' (line 1635)
    bot_176173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 4), 'bot')
    Ellipsis_176174 = Ellipsis
    # Storing an element on a container (line 1635)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1635, 4), bot_176173, (Ellipsis_176174, top_176172))
    
    # Getting the type of 'mat' (line 1636)
    mat_176175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 4), 'mat')
    
    # Obtaining the type of the subscript
    slice_176176 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1636, 4), None, None, None)
    int_176177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 11), 'int')
    # Getting the type of 'mat' (line 1636)
    mat_176178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 4), 'mat')
    # Obtaining the member '__getitem__' of a type (line 1636)
    getitem___176179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 4), mat_176178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1636)
    subscript_call_result_176180 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 4), getitem___176179, (slice_176176, int_176177))
    
    
    # Obtaining the type of the subscript
    int_176181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 22), 'int')
    slice_176182 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1636, 19), None, int_176181, None)
    # Getting the type of 'c' (line 1636)
    c_176183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 1636)
    getitem___176184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 19), c_176183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1636)
    subscript_call_result_176185 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 19), getitem___176184, slice_176182)
    
    
    # Obtaining the type of the subscript
    int_176186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 28), 'int')
    # Getting the type of 'c' (line 1636)
    c_176187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 26), 'c')
    # Obtaining the member '__getitem__' of a type (line 1636)
    getitem___176188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 26), c_176187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1636)
    subscript_call_result_176189 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 26), getitem___176188, int_176186)
    
    # Applying the binary operator 'div' (line 1636)
    result_div_176190 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 19), 'div', subscript_call_result_176185, subscript_call_result_176189)
    
    # Getting the type of 'scl' (line 1636)
    scl_176191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 34), 'scl')
    
    # Obtaining the type of the subscript
    int_176192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 42), 'int')
    # Getting the type of 'scl' (line 1636)
    scl_176193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 38), 'scl')
    # Obtaining the member '__getitem__' of a type (line 1636)
    getitem___176194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 38), scl_176193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1636)
    subscript_call_result_176195 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 38), getitem___176194, int_176192)
    
    # Applying the binary operator 'div' (line 1636)
    result_div_176196 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 34), 'div', scl_176191, subscript_call_result_176195)
    
    # Applying the binary operator '*' (line 1636)
    result_mul_176197 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 18), '*', result_div_176190, result_div_176196)
    
    # Getting the type of 'n' (line 1636)
    n_176198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 48), 'n')
    int_176199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 51), 'int')
    # Getting the type of 'n' (line 1636)
    n_176200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 53), 'n')
    # Applying the binary operator '*' (line 1636)
    result_mul_176201 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 51), '*', int_176199, n_176200)
    
    int_176202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 57), 'int')
    # Applying the binary operator '-' (line 1636)
    result_sub_176203 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 51), '-', result_mul_176201, int_176202)
    
    # Applying the binary operator 'div' (line 1636)
    result_div_176204 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 48), 'div', n_176198, result_sub_176203)
    
    # Applying the binary operator '*' (line 1636)
    result_mul_176205 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 46), '*', result_mul_176197, result_div_176204)
    
    # Applying the binary operator '-=' (line 1636)
    result_isub_176206 = python_operator(stypy.reporting.localization.Localization(__file__, 1636, 4), '-=', subscript_call_result_176180, result_mul_176205)
    # Getting the type of 'mat' (line 1636)
    mat_176207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 4), 'mat')
    slice_176208 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1636, 4), None, None, None)
    int_176209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 11), 'int')
    # Storing an element on a container (line 1636)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1636, 4), mat_176207, ((slice_176208, int_176209), result_isub_176206))
    
    # Getting the type of 'mat' (line 1637)
    mat_176210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 11), 'mat')
    # Assigning a type to the variable 'stypy_return_type' (line 1637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1637, 4), 'stypy_return_type', mat_176210)
    
    # ################# End of 'legcompanion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legcompanion' in the type store
    # Getting the type of 'stypy_return_type' (line 1596)
    stypy_return_type_176211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176211)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legcompanion'
    return stypy_return_type_176211

# Assigning a type to the variable 'legcompanion' (line 1596)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1596, 0), 'legcompanion', legcompanion)

@norecursion
def legroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legroots'
    module_type_store = module_type_store.open_function_context('legroots', 1640, 0, False)
    
    # Passed parameters checking function
    legroots.stypy_localization = localization
    legroots.stypy_type_of_self = None
    legroots.stypy_type_store = module_type_store
    legroots.stypy_function_name = 'legroots'
    legroots.stypy_param_names_list = ['c']
    legroots.stypy_varargs_param_name = None
    legroots.stypy_kwargs_param_name = None
    legroots.stypy_call_defaults = defaults
    legroots.stypy_call_varargs = varargs
    legroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legroots', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legroots', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legroots(...)' code ##################

    str_176212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1682, (-1)), 'str', '\n    Compute the roots of a Legendre series.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * L_i(x).\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the series. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    polyroots, chebroots, lagroots, hermroots, hermeroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the series for such values.\n    Roots with multiplicity greater than 1 will also show larger errors as\n    the value of the series near such points is relatively insensitive to\n    errors in the roots. Isolated roots near the origin can be improved by\n    a few iterations of Newton\'s method.\n\n    The Legendre series basis polynomials aren\'t powers of ``x`` so the\n    results of this function may seem unintuitive.\n\n    Examples\n    --------\n    >>> import numpy.polynomial.legendre as leg\n    >>> leg.legroots((1, 2, 3, 4)) # 4L_3 + 3L_2 + 2L_1 + 1L_0, all real roots\n    array([-0.85099543, -0.11407192,  0.51506735])\n\n    ')
    
    # Assigning a Call to a List (line 1684):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1684)
    # Processing the call arguments (line 1684)
    
    # Obtaining an instance of the builtin type 'list' (line 1684)
    list_176215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1684, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1684)
    # Adding element type (line 1684)
    # Getting the type of 'c' (line 1684)
    c_176216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1684, 23), list_176215, c_176216)
    
    # Processing the call keyword arguments (line 1684)
    kwargs_176217 = {}
    # Getting the type of 'pu' (line 1684)
    pu_176213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1684)
    as_series_176214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1684, 10), pu_176213, 'as_series')
    # Calling as_series(args, kwargs) (line 1684)
    as_series_call_result_176218 = invoke(stypy.reporting.localization.Localization(__file__, 1684, 10), as_series_176214, *[list_176215], **kwargs_176217)
    
    # Assigning a type to the variable 'call_assignment_173656' (line 1684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1684, 4), 'call_assignment_173656', as_series_call_result_176218)
    
    # Assigning a Call to a Name (line 1684):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1684, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176222 = {}
    # Getting the type of 'call_assignment_173656' (line 1684)
    call_assignment_173656_176219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 4), 'call_assignment_173656', False)
    # Obtaining the member '__getitem__' of a type (line 1684)
    getitem___176220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1684, 4), call_assignment_173656_176219, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176223 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176220, *[int_176221], **kwargs_176222)
    
    # Assigning a type to the variable 'call_assignment_173657' (line 1684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1684, 4), 'call_assignment_173657', getitem___call_result_176223)
    
    # Assigning a Name to a Name (line 1684):
    # Getting the type of 'call_assignment_173657' (line 1684)
    call_assignment_173657_176224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 4), 'call_assignment_173657')
    # Assigning a type to the variable 'c' (line 1684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1684, 5), 'c', call_assignment_173657_176224)
    
    
    
    # Call to len(...): (line 1685)
    # Processing the call arguments (line 1685)
    # Getting the type of 'c' (line 1685)
    c_176226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 11), 'c', False)
    # Processing the call keyword arguments (line 1685)
    kwargs_176227 = {}
    # Getting the type of 'len' (line 1685)
    len_176225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 7), 'len', False)
    # Calling len(args, kwargs) (line 1685)
    len_call_result_176228 = invoke(stypy.reporting.localization.Localization(__file__, 1685, 7), len_176225, *[c_176226], **kwargs_176227)
    
    int_176229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1685, 16), 'int')
    # Applying the binary operator '<' (line 1685)
    result_lt_176230 = python_operator(stypy.reporting.localization.Localization(__file__, 1685, 7), '<', len_call_result_176228, int_176229)
    
    # Testing the type of an if condition (line 1685)
    if_condition_176231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1685, 4), result_lt_176230)
    # Assigning a type to the variable 'if_condition_176231' (line 1685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1685, 4), 'if_condition_176231', if_condition_176231)
    # SSA begins for if statement (line 1685)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1686)
    # Processing the call arguments (line 1686)
    
    # Obtaining an instance of the builtin type 'list' (line 1686)
    list_176234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1686, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1686)
    
    # Processing the call keyword arguments (line 1686)
    # Getting the type of 'c' (line 1686)
    c_176235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 34), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1686)
    dtype_176236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1686, 34), c_176235, 'dtype')
    keyword_176237 = dtype_176236
    kwargs_176238 = {'dtype': keyword_176237}
    # Getting the type of 'np' (line 1686)
    np_176232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1686)
    array_176233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1686, 15), np_176232, 'array')
    # Calling array(args, kwargs) (line 1686)
    array_call_result_176239 = invoke(stypy.reporting.localization.Localization(__file__, 1686, 15), array_176233, *[list_176234], **kwargs_176238)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1686, 8), 'stypy_return_type', array_call_result_176239)
    # SSA join for if statement (line 1685)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1687)
    # Processing the call arguments (line 1687)
    # Getting the type of 'c' (line 1687)
    c_176241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1687, 11), 'c', False)
    # Processing the call keyword arguments (line 1687)
    kwargs_176242 = {}
    # Getting the type of 'len' (line 1687)
    len_176240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1687, 7), 'len', False)
    # Calling len(args, kwargs) (line 1687)
    len_call_result_176243 = invoke(stypy.reporting.localization.Localization(__file__, 1687, 7), len_176240, *[c_176241], **kwargs_176242)
    
    int_176244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1687, 17), 'int')
    # Applying the binary operator '==' (line 1687)
    result_eq_176245 = python_operator(stypy.reporting.localization.Localization(__file__, 1687, 7), '==', len_call_result_176243, int_176244)
    
    # Testing the type of an if condition (line 1687)
    if_condition_176246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1687, 4), result_eq_176245)
    # Assigning a type to the variable 'if_condition_176246' (line 1687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1687, 4), 'if_condition_176246', if_condition_176246)
    # SSA begins for if statement (line 1687)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1688)
    # Processing the call arguments (line 1688)
    
    # Obtaining an instance of the builtin type 'list' (line 1688)
    list_176249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1688, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1688)
    # Adding element type (line 1688)
    
    
    # Obtaining the type of the subscript
    int_176250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1688, 28), 'int')
    # Getting the type of 'c' (line 1688)
    c_176251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1688, 26), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1688)
    getitem___176252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1688, 26), c_176251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1688)
    subscript_call_result_176253 = invoke(stypy.reporting.localization.Localization(__file__, 1688, 26), getitem___176252, int_176250)
    
    # Applying the 'usub' unary operator (line 1688)
    result___neg___176254 = python_operator(stypy.reporting.localization.Localization(__file__, 1688, 25), 'usub', subscript_call_result_176253)
    
    
    # Obtaining the type of the subscript
    int_176255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1688, 33), 'int')
    # Getting the type of 'c' (line 1688)
    c_176256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1688, 31), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1688)
    getitem___176257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1688, 31), c_176256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1688)
    subscript_call_result_176258 = invoke(stypy.reporting.localization.Localization(__file__, 1688, 31), getitem___176257, int_176255)
    
    # Applying the binary operator 'div' (line 1688)
    result_div_176259 = python_operator(stypy.reporting.localization.Localization(__file__, 1688, 25), 'div', result___neg___176254, subscript_call_result_176258)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1688, 24), list_176249, result_div_176259)
    
    # Processing the call keyword arguments (line 1688)
    kwargs_176260 = {}
    # Getting the type of 'np' (line 1688)
    np_176247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1688, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1688)
    array_176248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1688, 15), np_176247, 'array')
    # Calling array(args, kwargs) (line 1688)
    array_call_result_176261 = invoke(stypy.reporting.localization.Localization(__file__, 1688, 15), array_176248, *[list_176249], **kwargs_176260)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1688, 8), 'stypy_return_type', array_call_result_176261)
    # SSA join for if statement (line 1687)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1690):
    
    # Assigning a Call to a Name (line 1690):
    
    # Call to legcompanion(...): (line 1690)
    # Processing the call arguments (line 1690)
    # Getting the type of 'c' (line 1690)
    c_176263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1690, 21), 'c', False)
    # Processing the call keyword arguments (line 1690)
    kwargs_176264 = {}
    # Getting the type of 'legcompanion' (line 1690)
    legcompanion_176262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1690, 8), 'legcompanion', False)
    # Calling legcompanion(args, kwargs) (line 1690)
    legcompanion_call_result_176265 = invoke(stypy.reporting.localization.Localization(__file__, 1690, 8), legcompanion_176262, *[c_176263], **kwargs_176264)
    
    # Assigning a type to the variable 'm' (line 1690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1690, 4), 'm', legcompanion_call_result_176265)
    
    # Assigning a Call to a Name (line 1691):
    
    # Assigning a Call to a Name (line 1691):
    
    # Call to eigvals(...): (line 1691)
    # Processing the call arguments (line 1691)
    # Getting the type of 'm' (line 1691)
    m_176268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1691, 19), 'm', False)
    # Processing the call keyword arguments (line 1691)
    kwargs_176269 = {}
    # Getting the type of 'la' (line 1691)
    la_176266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1691, 8), 'la', False)
    # Obtaining the member 'eigvals' of a type (line 1691)
    eigvals_176267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1691, 8), la_176266, 'eigvals')
    # Calling eigvals(args, kwargs) (line 1691)
    eigvals_call_result_176270 = invoke(stypy.reporting.localization.Localization(__file__, 1691, 8), eigvals_176267, *[m_176268], **kwargs_176269)
    
    # Assigning a type to the variable 'r' (line 1691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1691, 4), 'r', eigvals_call_result_176270)
    
    # Call to sort(...): (line 1692)
    # Processing the call keyword arguments (line 1692)
    kwargs_176273 = {}
    # Getting the type of 'r' (line 1692)
    r_176271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1692, 4), 'r', False)
    # Obtaining the member 'sort' of a type (line 1692)
    sort_176272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1692, 4), r_176271, 'sort')
    # Calling sort(args, kwargs) (line 1692)
    sort_call_result_176274 = invoke(stypy.reporting.localization.Localization(__file__, 1692, 4), sort_176272, *[], **kwargs_176273)
    
    # Getting the type of 'r' (line 1693)
    r_176275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1693, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 1693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1693, 4), 'stypy_return_type', r_176275)
    
    # ################# End of 'legroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legroots' in the type store
    # Getting the type of 'stypy_return_type' (line 1640)
    stypy_return_type_176276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1640, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176276)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legroots'
    return stypy_return_type_176276

# Assigning a type to the variable 'legroots' (line 1640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1640, 0), 'legroots', legroots)

@norecursion
def leggauss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'leggauss'
    module_type_store = module_type_store.open_function_context('leggauss', 1696, 0, False)
    
    # Passed parameters checking function
    leggauss.stypy_localization = localization
    leggauss.stypy_type_of_self = None
    leggauss.stypy_type_store = module_type_store
    leggauss.stypy_function_name = 'leggauss'
    leggauss.stypy_param_names_list = ['deg']
    leggauss.stypy_varargs_param_name = None
    leggauss.stypy_kwargs_param_name = None
    leggauss.stypy_call_defaults = defaults
    leggauss.stypy_call_varargs = varargs
    leggauss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'leggauss', ['deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'leggauss', localization, ['deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'leggauss(...)' code ##################

    str_176277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1731, (-1)), 'str', "\n    Gauss-Legendre quadrature.\n\n    Computes the sample points and weights for Gauss-Legendre quadrature.\n    These sample points and weights will correctly integrate polynomials of\n    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with\n    the weight function :math:`f(x) = 1`.\n\n    Parameters\n    ----------\n    deg : int\n        Number of sample points and weights. It must be >= 1.\n\n    Returns\n    -------\n    x : ndarray\n        1-D ndarray containing the sample points.\n    y : ndarray\n        1-D ndarray containing the weights.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    The results have only been tested up to degree 100, higher degrees may\n    be problematic. The weights are determined by using the fact that\n\n    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))\n\n    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`\n    is the k'th root of :math:`L_n`, and then scaling the results to get\n    the right value when integrating 1.\n\n    ")
    
    # Assigning a Call to a Name (line 1732):
    
    # Assigning a Call to a Name (line 1732):
    
    # Call to int(...): (line 1732)
    # Processing the call arguments (line 1732)
    # Getting the type of 'deg' (line 1732)
    deg_176279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1732, 15), 'deg', False)
    # Processing the call keyword arguments (line 1732)
    kwargs_176280 = {}
    # Getting the type of 'int' (line 1732)
    int_176278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1732, 11), 'int', False)
    # Calling int(args, kwargs) (line 1732)
    int_call_result_176281 = invoke(stypy.reporting.localization.Localization(__file__, 1732, 11), int_176278, *[deg_176279], **kwargs_176280)
    
    # Assigning a type to the variable 'ideg' (line 1732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1732, 4), 'ideg', int_call_result_176281)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ideg' (line 1733)
    ideg_176282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1733, 7), 'ideg')
    # Getting the type of 'deg' (line 1733)
    deg_176283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1733, 15), 'deg')
    # Applying the binary operator '!=' (line 1733)
    result_ne_176284 = python_operator(stypy.reporting.localization.Localization(__file__, 1733, 7), '!=', ideg_176282, deg_176283)
    
    
    # Getting the type of 'ideg' (line 1733)
    ideg_176285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1733, 22), 'ideg')
    int_176286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1733, 29), 'int')
    # Applying the binary operator '<' (line 1733)
    result_lt_176287 = python_operator(stypy.reporting.localization.Localization(__file__, 1733, 22), '<', ideg_176285, int_176286)
    
    # Applying the binary operator 'or' (line 1733)
    result_or_keyword_176288 = python_operator(stypy.reporting.localization.Localization(__file__, 1733, 7), 'or', result_ne_176284, result_lt_176287)
    
    # Testing the type of an if condition (line 1733)
    if_condition_176289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1733, 4), result_or_keyword_176288)
    # Assigning a type to the variable 'if_condition_176289' (line 1733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1733, 4), 'if_condition_176289', if_condition_176289)
    # SSA begins for if statement (line 1733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1734)
    # Processing the call arguments (line 1734)
    str_176291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1734, 25), 'str', 'deg must be a non-negative integer')
    # Processing the call keyword arguments (line 1734)
    kwargs_176292 = {}
    # Getting the type of 'ValueError' (line 1734)
    ValueError_176290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1734, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1734)
    ValueError_call_result_176293 = invoke(stypy.reporting.localization.Localization(__file__, 1734, 14), ValueError_176290, *[str_176291], **kwargs_176292)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1734, 8), ValueError_call_result_176293, 'raise parameter', BaseException)
    # SSA join for if statement (line 1733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1738):
    
    # Assigning a Call to a Name (line 1738):
    
    # Call to array(...): (line 1738)
    # Processing the call arguments (line 1738)
    
    # Obtaining an instance of the builtin type 'list' (line 1738)
    list_176296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1738, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1738)
    # Adding element type (line 1738)
    int_176297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1738, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1738, 17), list_176296, int_176297)
    
    # Getting the type of 'deg' (line 1738)
    deg_176298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 21), 'deg', False)
    # Applying the binary operator '*' (line 1738)
    result_mul_176299 = python_operator(stypy.reporting.localization.Localization(__file__, 1738, 17), '*', list_176296, deg_176298)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1738)
    list_176300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1738, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1738)
    # Adding element type (line 1738)
    int_176301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1738, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1738, 27), list_176300, int_176301)
    
    # Applying the binary operator '+' (line 1738)
    result_add_176302 = python_operator(stypy.reporting.localization.Localization(__file__, 1738, 17), '+', result_mul_176299, list_176300)
    
    # Processing the call keyword arguments (line 1738)
    kwargs_176303 = {}
    # Getting the type of 'np' (line 1738)
    np_176294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1738)
    array_176295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1738, 8), np_176294, 'array')
    # Calling array(args, kwargs) (line 1738)
    array_call_result_176304 = invoke(stypy.reporting.localization.Localization(__file__, 1738, 8), array_176295, *[result_add_176302], **kwargs_176303)
    
    # Assigning a type to the variable 'c' (line 1738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1738, 4), 'c', array_call_result_176304)
    
    # Assigning a Call to a Name (line 1739):
    
    # Assigning a Call to a Name (line 1739):
    
    # Call to legcompanion(...): (line 1739)
    # Processing the call arguments (line 1739)
    # Getting the type of 'c' (line 1739)
    c_176306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 21), 'c', False)
    # Processing the call keyword arguments (line 1739)
    kwargs_176307 = {}
    # Getting the type of 'legcompanion' (line 1739)
    legcompanion_176305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 8), 'legcompanion', False)
    # Calling legcompanion(args, kwargs) (line 1739)
    legcompanion_call_result_176308 = invoke(stypy.reporting.localization.Localization(__file__, 1739, 8), legcompanion_176305, *[c_176306], **kwargs_176307)
    
    # Assigning a type to the variable 'm' (line 1739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1739, 4), 'm', legcompanion_call_result_176308)
    
    # Assigning a Call to a Name (line 1740):
    
    # Assigning a Call to a Name (line 1740):
    
    # Call to eigvalsh(...): (line 1740)
    # Processing the call arguments (line 1740)
    # Getting the type of 'm' (line 1740)
    m_176311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 20), 'm', False)
    # Processing the call keyword arguments (line 1740)
    kwargs_176312 = {}
    # Getting the type of 'la' (line 1740)
    la_176309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 8), 'la', False)
    # Obtaining the member 'eigvalsh' of a type (line 1740)
    eigvalsh_176310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1740, 8), la_176309, 'eigvalsh')
    # Calling eigvalsh(args, kwargs) (line 1740)
    eigvalsh_call_result_176313 = invoke(stypy.reporting.localization.Localization(__file__, 1740, 8), eigvalsh_176310, *[m_176311], **kwargs_176312)
    
    # Assigning a type to the variable 'x' (line 1740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1740, 4), 'x', eigvalsh_call_result_176313)
    
    # Assigning a Call to a Name (line 1743):
    
    # Assigning a Call to a Name (line 1743):
    
    # Call to legval(...): (line 1743)
    # Processing the call arguments (line 1743)
    # Getting the type of 'x' (line 1743)
    x_176315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 16), 'x', False)
    # Getting the type of 'c' (line 1743)
    c_176316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 19), 'c', False)
    # Processing the call keyword arguments (line 1743)
    kwargs_176317 = {}
    # Getting the type of 'legval' (line 1743)
    legval_176314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 9), 'legval', False)
    # Calling legval(args, kwargs) (line 1743)
    legval_call_result_176318 = invoke(stypy.reporting.localization.Localization(__file__, 1743, 9), legval_176314, *[x_176315, c_176316], **kwargs_176317)
    
    # Assigning a type to the variable 'dy' (line 1743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1743, 4), 'dy', legval_call_result_176318)
    
    # Assigning a Call to a Name (line 1744):
    
    # Assigning a Call to a Name (line 1744):
    
    # Call to legval(...): (line 1744)
    # Processing the call arguments (line 1744)
    # Getting the type of 'x' (line 1744)
    x_176320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 16), 'x', False)
    
    # Call to legder(...): (line 1744)
    # Processing the call arguments (line 1744)
    # Getting the type of 'c' (line 1744)
    c_176322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 26), 'c', False)
    # Processing the call keyword arguments (line 1744)
    kwargs_176323 = {}
    # Getting the type of 'legder' (line 1744)
    legder_176321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 19), 'legder', False)
    # Calling legder(args, kwargs) (line 1744)
    legder_call_result_176324 = invoke(stypy.reporting.localization.Localization(__file__, 1744, 19), legder_176321, *[c_176322], **kwargs_176323)
    
    # Processing the call keyword arguments (line 1744)
    kwargs_176325 = {}
    # Getting the type of 'legval' (line 1744)
    legval_176319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 9), 'legval', False)
    # Calling legval(args, kwargs) (line 1744)
    legval_call_result_176326 = invoke(stypy.reporting.localization.Localization(__file__, 1744, 9), legval_176319, *[x_176320, legder_call_result_176324], **kwargs_176325)
    
    # Assigning a type to the variable 'df' (line 1744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1744, 4), 'df', legval_call_result_176326)
    
    # Getting the type of 'x' (line 1745)
    x_176327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 4), 'x')
    # Getting the type of 'dy' (line 1745)
    dy_176328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 9), 'dy')
    # Getting the type of 'df' (line 1745)
    df_176329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 12), 'df')
    # Applying the binary operator 'div' (line 1745)
    result_div_176330 = python_operator(stypy.reporting.localization.Localization(__file__, 1745, 9), 'div', dy_176328, df_176329)
    
    # Applying the binary operator '-=' (line 1745)
    result_isub_176331 = python_operator(stypy.reporting.localization.Localization(__file__, 1745, 4), '-=', x_176327, result_div_176330)
    # Assigning a type to the variable 'x' (line 1745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1745, 4), 'x', result_isub_176331)
    
    
    # Assigning a Call to a Name (line 1749):
    
    # Assigning a Call to a Name (line 1749):
    
    # Call to legval(...): (line 1749)
    # Processing the call arguments (line 1749)
    # Getting the type of 'x' (line 1749)
    x_176333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 16), 'x', False)
    
    # Obtaining the type of the subscript
    int_176334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1749, 21), 'int')
    slice_176335 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1749, 19), int_176334, None, None)
    # Getting the type of 'c' (line 1749)
    c_176336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 19), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1749)
    getitem___176337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1749, 19), c_176336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1749)
    subscript_call_result_176338 = invoke(stypy.reporting.localization.Localization(__file__, 1749, 19), getitem___176337, slice_176335)
    
    # Processing the call keyword arguments (line 1749)
    kwargs_176339 = {}
    # Getting the type of 'legval' (line 1749)
    legval_176332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 9), 'legval', False)
    # Calling legval(args, kwargs) (line 1749)
    legval_call_result_176340 = invoke(stypy.reporting.localization.Localization(__file__, 1749, 9), legval_176332, *[x_176333, subscript_call_result_176338], **kwargs_176339)
    
    # Assigning a type to the variable 'fm' (line 1749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1749, 4), 'fm', legval_call_result_176340)
    
    # Getting the type of 'fm' (line 1750)
    fm_176341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 4), 'fm')
    
    # Call to max(...): (line 1750)
    # Processing the call keyword arguments (line 1750)
    kwargs_176348 = {}
    
    # Call to abs(...): (line 1750)
    # Processing the call arguments (line 1750)
    # Getting the type of 'fm' (line 1750)
    fm_176344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 17), 'fm', False)
    # Processing the call keyword arguments (line 1750)
    kwargs_176345 = {}
    # Getting the type of 'np' (line 1750)
    np_176342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1750)
    abs_176343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1750, 10), np_176342, 'abs')
    # Calling abs(args, kwargs) (line 1750)
    abs_call_result_176346 = invoke(stypy.reporting.localization.Localization(__file__, 1750, 10), abs_176343, *[fm_176344], **kwargs_176345)
    
    # Obtaining the member 'max' of a type (line 1750)
    max_176347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1750, 10), abs_call_result_176346, 'max')
    # Calling max(args, kwargs) (line 1750)
    max_call_result_176349 = invoke(stypy.reporting.localization.Localization(__file__, 1750, 10), max_176347, *[], **kwargs_176348)
    
    # Applying the binary operator 'div=' (line 1750)
    result_div_176350 = python_operator(stypy.reporting.localization.Localization(__file__, 1750, 4), 'div=', fm_176341, max_call_result_176349)
    # Assigning a type to the variable 'fm' (line 1750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1750, 4), 'fm', result_div_176350)
    
    
    # Getting the type of 'df' (line 1751)
    df_176351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 4), 'df')
    
    # Call to max(...): (line 1751)
    # Processing the call keyword arguments (line 1751)
    kwargs_176358 = {}
    
    # Call to abs(...): (line 1751)
    # Processing the call arguments (line 1751)
    # Getting the type of 'df' (line 1751)
    df_176354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 17), 'df', False)
    # Processing the call keyword arguments (line 1751)
    kwargs_176355 = {}
    # Getting the type of 'np' (line 1751)
    np_176352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1751)
    abs_176353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1751, 10), np_176352, 'abs')
    # Calling abs(args, kwargs) (line 1751)
    abs_call_result_176356 = invoke(stypy.reporting.localization.Localization(__file__, 1751, 10), abs_176353, *[df_176354], **kwargs_176355)
    
    # Obtaining the member 'max' of a type (line 1751)
    max_176357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1751, 10), abs_call_result_176356, 'max')
    # Calling max(args, kwargs) (line 1751)
    max_call_result_176359 = invoke(stypy.reporting.localization.Localization(__file__, 1751, 10), max_176357, *[], **kwargs_176358)
    
    # Applying the binary operator 'div=' (line 1751)
    result_div_176360 = python_operator(stypy.reporting.localization.Localization(__file__, 1751, 4), 'div=', df_176351, max_call_result_176359)
    # Assigning a type to the variable 'df' (line 1751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1751, 4), 'df', result_div_176360)
    
    
    # Assigning a BinOp to a Name (line 1752):
    
    # Assigning a BinOp to a Name (line 1752):
    int_176361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1752, 8), 'int')
    # Getting the type of 'fm' (line 1752)
    fm_176362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 11), 'fm')
    # Getting the type of 'df' (line 1752)
    df_176363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 16), 'df')
    # Applying the binary operator '*' (line 1752)
    result_mul_176364 = python_operator(stypy.reporting.localization.Localization(__file__, 1752, 11), '*', fm_176362, df_176363)
    
    # Applying the binary operator 'div' (line 1752)
    result_div_176365 = python_operator(stypy.reporting.localization.Localization(__file__, 1752, 8), 'div', int_176361, result_mul_176364)
    
    # Assigning a type to the variable 'w' (line 1752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1752, 4), 'w', result_div_176365)
    
    # Assigning a BinOp to a Name (line 1755):
    
    # Assigning a BinOp to a Name (line 1755):
    # Getting the type of 'w' (line 1755)
    w_176366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 9), 'w')
    
    # Obtaining the type of the subscript
    int_176367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1755, 17), 'int')
    slice_176368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1755, 13), None, None, int_176367)
    # Getting the type of 'w' (line 1755)
    w_176369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 13), 'w')
    # Obtaining the member '__getitem__' of a type (line 1755)
    getitem___176370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1755, 13), w_176369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1755)
    subscript_call_result_176371 = invoke(stypy.reporting.localization.Localization(__file__, 1755, 13), getitem___176370, slice_176368)
    
    # Applying the binary operator '+' (line 1755)
    result_add_176372 = python_operator(stypy.reporting.localization.Localization(__file__, 1755, 9), '+', w_176366, subscript_call_result_176371)
    
    int_176373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1755, 22), 'int')
    # Applying the binary operator 'div' (line 1755)
    result_div_176374 = python_operator(stypy.reporting.localization.Localization(__file__, 1755, 8), 'div', result_add_176372, int_176373)
    
    # Assigning a type to the variable 'w' (line 1755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1755, 4), 'w', result_div_176374)
    
    # Assigning a BinOp to a Name (line 1756):
    
    # Assigning a BinOp to a Name (line 1756):
    # Getting the type of 'x' (line 1756)
    x_176375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1756, 9), 'x')
    
    # Obtaining the type of the subscript
    int_176376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1756, 17), 'int')
    slice_176377 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1756, 13), None, None, int_176376)
    # Getting the type of 'x' (line 1756)
    x_176378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1756, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 1756)
    getitem___176379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1756, 13), x_176378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1756)
    subscript_call_result_176380 = invoke(stypy.reporting.localization.Localization(__file__, 1756, 13), getitem___176379, slice_176377)
    
    # Applying the binary operator '-' (line 1756)
    result_sub_176381 = python_operator(stypy.reporting.localization.Localization(__file__, 1756, 9), '-', x_176375, subscript_call_result_176380)
    
    int_176382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1756, 22), 'int')
    # Applying the binary operator 'div' (line 1756)
    result_div_176383 = python_operator(stypy.reporting.localization.Localization(__file__, 1756, 8), 'div', result_sub_176381, int_176382)
    
    # Assigning a type to the variable 'x' (line 1756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1756, 4), 'x', result_div_176383)
    
    # Getting the type of 'w' (line 1759)
    w_176384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 4), 'w')
    float_176385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1759, 9), 'float')
    
    # Call to sum(...): (line 1759)
    # Processing the call keyword arguments (line 1759)
    kwargs_176388 = {}
    # Getting the type of 'w' (line 1759)
    w_176386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 14), 'w', False)
    # Obtaining the member 'sum' of a type (line 1759)
    sum_176387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1759, 14), w_176386, 'sum')
    # Calling sum(args, kwargs) (line 1759)
    sum_call_result_176389 = invoke(stypy.reporting.localization.Localization(__file__, 1759, 14), sum_176387, *[], **kwargs_176388)
    
    # Applying the binary operator 'div' (line 1759)
    result_div_176390 = python_operator(stypy.reporting.localization.Localization(__file__, 1759, 9), 'div', float_176385, sum_call_result_176389)
    
    # Applying the binary operator '*=' (line 1759)
    result_imul_176391 = python_operator(stypy.reporting.localization.Localization(__file__, 1759, 4), '*=', w_176384, result_div_176390)
    # Assigning a type to the variable 'w' (line 1759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1759, 4), 'w', result_imul_176391)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1761)
    tuple_176392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1761, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1761)
    # Adding element type (line 1761)
    # Getting the type of 'x' (line 1761)
    x_176393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1761, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1761, 11), tuple_176392, x_176393)
    # Adding element type (line 1761)
    # Getting the type of 'w' (line 1761)
    w_176394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1761, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1761, 11), tuple_176392, w_176394)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1761, 4), 'stypy_return_type', tuple_176392)
    
    # ################# End of 'leggauss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'leggauss' in the type store
    # Getting the type of 'stypy_return_type' (line 1696)
    stypy_return_type_176395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1696, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176395)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'leggauss'
    return stypy_return_type_176395

# Assigning a type to the variable 'leggauss' (line 1696)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1696, 0), 'leggauss', leggauss)

@norecursion
def legweight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legweight'
    module_type_store = module_type_store.open_function_context('legweight', 1764, 0, False)
    
    # Passed parameters checking function
    legweight.stypy_localization = localization
    legweight.stypy_type_of_self = None
    legweight.stypy_type_store = module_type_store
    legweight.stypy_function_name = 'legweight'
    legweight.stypy_param_names_list = ['x']
    legweight.stypy_varargs_param_name = None
    legweight.stypy_kwargs_param_name = None
    legweight.stypy_call_defaults = defaults
    legweight.stypy_call_varargs = varargs
    legweight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legweight', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legweight', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legweight(...)' code ##################

    str_176396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1787, (-1)), 'str', '\n    Weight function of the Legendre polynomials.\n\n    The weight function is :math:`1` and the interval of integration is\n    :math:`[-1, 1]`. The Legendre polynomials are orthogonal, but not\n    normalized, with respect to this weight function.\n\n    Parameters\n    ----------\n    x : array_like\n       Values at which the weight function will be computed.\n\n    Returns\n    -------\n    w : ndarray\n       The weight function at `x`.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a BinOp to a Name (line 1788):
    
    # Assigning a BinOp to a Name (line 1788):
    # Getting the type of 'x' (line 1788)
    x_176397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 8), 'x')
    float_176398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1788, 10), 'float')
    # Applying the binary operator '*' (line 1788)
    result_mul_176399 = python_operator(stypy.reporting.localization.Localization(__file__, 1788, 8), '*', x_176397, float_176398)
    
    float_176400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1788, 16), 'float')
    # Applying the binary operator '+' (line 1788)
    result_add_176401 = python_operator(stypy.reporting.localization.Localization(__file__, 1788, 8), '+', result_mul_176399, float_176400)
    
    # Assigning a type to the variable 'w' (line 1788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1788, 4), 'w', result_add_176401)
    # Getting the type of 'w' (line 1789)
    w_176402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1789, 11), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 1789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1789, 4), 'stypy_return_type', w_176402)
    
    # ################# End of 'legweight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legweight' in the type store
    # Getting the type of 'stypy_return_type' (line 1764)
    stypy_return_type_176403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176403)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legweight'
    return stypy_return_type_176403

# Assigning a type to the variable 'legweight' (line 1764)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1764, 0), 'legweight', legweight)
# Declaration of the 'Legendre' class
# Getting the type of 'ABCPolyBase' (line 1795)
ABCPolyBase_176404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1795, 15), 'ABCPolyBase')

class Legendre(ABCPolyBase_176404, ):
    str_176405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1816, (-1)), 'str', "A Legendre series class.\n\n    The Legendre class provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the\n    attributes and methods listed in the `ABCPolyBase` documentation.\n\n    Parameters\n    ----------\n    coef : array_like\n        Legendre coefficients in order of increasing degree, i.e.,\n        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is [-1, 1].\n    window : (2,) array_like, optional\n        Window, see `domain` for its use. The default value is [-1, 1].\n\n        .. versionadded:: 1.6.0\n\n    ")
    
    # Assigning a Call to a Name (line 1818):
    
    # Assigning a Call to a Name (line 1819):
    
    # Assigning a Call to a Name (line 1820):
    
    # Assigning a Call to a Name (line 1821):
    
    # Assigning a Call to a Name (line 1822):
    
    # Assigning a Call to a Name (line 1823):
    
    # Assigning a Call to a Name (line 1824):
    
    # Assigning a Call to a Name (line 1825):
    
    # Assigning a Call to a Name (line 1826):
    
    # Assigning a Call to a Name (line 1827):
    
    # Assigning a Call to a Name (line 1828):
    
    # Assigning a Call to a Name (line 1829):
    
    # Assigning a Str to a Name (line 1832):
    
    # Assigning a Call to a Name (line 1833):
    
    # Assigning a Call to a Name (line 1834):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1795, 0, False)
        # Assigning a type to the variable 'self' (line 1796)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1796, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legendre.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Legendre' (line 1795)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1795, 0), 'Legendre', Legendre)

# Assigning a Call to a Name (line 1818):

# Call to staticmethod(...): (line 1818)
# Processing the call arguments (line 1818)
# Getting the type of 'legadd' (line 1818)
legadd_176407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1818, 24), 'legadd', False)
# Processing the call keyword arguments (line 1818)
kwargs_176408 = {}
# Getting the type of 'staticmethod' (line 1818)
staticmethod_176406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1818, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1818)
staticmethod_call_result_176409 = invoke(stypy.reporting.localization.Localization(__file__, 1818, 11), staticmethod_176406, *[legadd_176407], **kwargs_176408)

# Getting the type of 'Legendre'
Legendre_176410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176410, '_add', staticmethod_call_result_176409)

# Assigning a Call to a Name (line 1819):

# Call to staticmethod(...): (line 1819)
# Processing the call arguments (line 1819)
# Getting the type of 'legsub' (line 1819)
legsub_176412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1819, 24), 'legsub', False)
# Processing the call keyword arguments (line 1819)
kwargs_176413 = {}
# Getting the type of 'staticmethod' (line 1819)
staticmethod_176411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1819, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1819)
staticmethod_call_result_176414 = invoke(stypy.reporting.localization.Localization(__file__, 1819, 11), staticmethod_176411, *[legsub_176412], **kwargs_176413)

# Getting the type of 'Legendre'
Legendre_176415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_sub' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176415, '_sub', staticmethod_call_result_176414)

# Assigning a Call to a Name (line 1820):

# Call to staticmethod(...): (line 1820)
# Processing the call arguments (line 1820)
# Getting the type of 'legmul' (line 1820)
legmul_176417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1820, 24), 'legmul', False)
# Processing the call keyword arguments (line 1820)
kwargs_176418 = {}
# Getting the type of 'staticmethod' (line 1820)
staticmethod_176416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1820, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1820)
staticmethod_call_result_176419 = invoke(stypy.reporting.localization.Localization(__file__, 1820, 11), staticmethod_176416, *[legmul_176417], **kwargs_176418)

# Getting the type of 'Legendre'
Legendre_176420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_mul' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176420, '_mul', staticmethod_call_result_176419)

# Assigning a Call to a Name (line 1821):

# Call to staticmethod(...): (line 1821)
# Processing the call arguments (line 1821)
# Getting the type of 'legdiv' (line 1821)
legdiv_176422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1821, 24), 'legdiv', False)
# Processing the call keyword arguments (line 1821)
kwargs_176423 = {}
# Getting the type of 'staticmethod' (line 1821)
staticmethod_176421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1821, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1821)
staticmethod_call_result_176424 = invoke(stypy.reporting.localization.Localization(__file__, 1821, 11), staticmethod_176421, *[legdiv_176422], **kwargs_176423)

# Getting the type of 'Legendre'
Legendre_176425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_div' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176425, '_div', staticmethod_call_result_176424)

# Assigning a Call to a Name (line 1822):

# Call to staticmethod(...): (line 1822)
# Processing the call arguments (line 1822)
# Getting the type of 'legpow' (line 1822)
legpow_176427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1822, 24), 'legpow', False)
# Processing the call keyword arguments (line 1822)
kwargs_176428 = {}
# Getting the type of 'staticmethod' (line 1822)
staticmethod_176426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1822, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1822)
staticmethod_call_result_176429 = invoke(stypy.reporting.localization.Localization(__file__, 1822, 11), staticmethod_176426, *[legpow_176427], **kwargs_176428)

# Getting the type of 'Legendre'
Legendre_176430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_pow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176430, '_pow', staticmethod_call_result_176429)

# Assigning a Call to a Name (line 1823):

# Call to staticmethod(...): (line 1823)
# Processing the call arguments (line 1823)
# Getting the type of 'legval' (line 1823)
legval_176432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1823, 24), 'legval', False)
# Processing the call keyword arguments (line 1823)
kwargs_176433 = {}
# Getting the type of 'staticmethod' (line 1823)
staticmethod_176431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1823, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1823)
staticmethod_call_result_176434 = invoke(stypy.reporting.localization.Localization(__file__, 1823, 11), staticmethod_176431, *[legval_176432], **kwargs_176433)

# Getting the type of 'Legendre'
Legendre_176435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_val' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176435, '_val', staticmethod_call_result_176434)

# Assigning a Call to a Name (line 1824):

# Call to staticmethod(...): (line 1824)
# Processing the call arguments (line 1824)
# Getting the type of 'legint' (line 1824)
legint_176437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 24), 'legint', False)
# Processing the call keyword arguments (line 1824)
kwargs_176438 = {}
# Getting the type of 'staticmethod' (line 1824)
staticmethod_176436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1824)
staticmethod_call_result_176439 = invoke(stypy.reporting.localization.Localization(__file__, 1824, 11), staticmethod_176436, *[legint_176437], **kwargs_176438)

# Getting the type of 'Legendre'
Legendre_176440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_int' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176440, '_int', staticmethod_call_result_176439)

# Assigning a Call to a Name (line 1825):

# Call to staticmethod(...): (line 1825)
# Processing the call arguments (line 1825)
# Getting the type of 'legder' (line 1825)
legder_176442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 24), 'legder', False)
# Processing the call keyword arguments (line 1825)
kwargs_176443 = {}
# Getting the type of 'staticmethod' (line 1825)
staticmethod_176441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1825)
staticmethod_call_result_176444 = invoke(stypy.reporting.localization.Localization(__file__, 1825, 11), staticmethod_176441, *[legder_176442], **kwargs_176443)

# Getting the type of 'Legendre'
Legendre_176445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_der' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176445, '_der', staticmethod_call_result_176444)

# Assigning a Call to a Name (line 1826):

# Call to staticmethod(...): (line 1826)
# Processing the call arguments (line 1826)
# Getting the type of 'legfit' (line 1826)
legfit_176447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 24), 'legfit', False)
# Processing the call keyword arguments (line 1826)
kwargs_176448 = {}
# Getting the type of 'staticmethod' (line 1826)
staticmethod_176446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1826)
staticmethod_call_result_176449 = invoke(stypy.reporting.localization.Localization(__file__, 1826, 11), staticmethod_176446, *[legfit_176447], **kwargs_176448)

# Getting the type of 'Legendre'
Legendre_176450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_fit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176450, '_fit', staticmethod_call_result_176449)

# Assigning a Call to a Name (line 1827):

# Call to staticmethod(...): (line 1827)
# Processing the call arguments (line 1827)
# Getting the type of 'legline' (line 1827)
legline_176452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 25), 'legline', False)
# Processing the call keyword arguments (line 1827)
kwargs_176453 = {}
# Getting the type of 'staticmethod' (line 1827)
staticmethod_176451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 12), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1827)
staticmethod_call_result_176454 = invoke(stypy.reporting.localization.Localization(__file__, 1827, 12), staticmethod_176451, *[legline_176452], **kwargs_176453)

# Getting the type of 'Legendre'
Legendre_176455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_line' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176455, '_line', staticmethod_call_result_176454)

# Assigning a Call to a Name (line 1828):

# Call to staticmethod(...): (line 1828)
# Processing the call arguments (line 1828)
# Getting the type of 'legroots' (line 1828)
legroots_176457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 26), 'legroots', False)
# Processing the call keyword arguments (line 1828)
kwargs_176458 = {}
# Getting the type of 'staticmethod' (line 1828)
staticmethod_176456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 13), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1828)
staticmethod_call_result_176459 = invoke(stypy.reporting.localization.Localization(__file__, 1828, 13), staticmethod_176456, *[legroots_176457], **kwargs_176458)

# Getting the type of 'Legendre'
Legendre_176460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_roots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176460, '_roots', staticmethod_call_result_176459)

# Assigning a Call to a Name (line 1829):

# Call to staticmethod(...): (line 1829)
# Processing the call arguments (line 1829)
# Getting the type of 'legfromroots' (line 1829)
legfromroots_176462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1829, 30), 'legfromroots', False)
# Processing the call keyword arguments (line 1829)
kwargs_176463 = {}
# Getting the type of 'staticmethod' (line 1829)
staticmethod_176461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1829, 17), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1829)
staticmethod_call_result_176464 = invoke(stypy.reporting.localization.Localization(__file__, 1829, 17), staticmethod_176461, *[legfromroots_176462], **kwargs_176463)

# Getting the type of 'Legendre'
Legendre_176465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member '_fromroots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176465, '_fromroots', staticmethod_call_result_176464)

# Assigning a Str to a Name (line 1832):
str_176466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1832, 15), 'str', 'leg')
# Getting the type of 'Legendre'
Legendre_176467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member 'nickname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176467, 'nickname', str_176466)

# Assigning a Call to a Name (line 1833):

# Call to array(...): (line 1833)
# Processing the call arguments (line 1833)
# Getting the type of 'legdomain' (line 1833)
legdomain_176470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 22), 'legdomain', False)
# Processing the call keyword arguments (line 1833)
kwargs_176471 = {}
# Getting the type of 'np' (line 1833)
np_176468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1833)
array_176469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1833, 13), np_176468, 'array')
# Calling array(args, kwargs) (line 1833)
array_call_result_176472 = invoke(stypy.reporting.localization.Localization(__file__, 1833, 13), array_176469, *[legdomain_176470], **kwargs_176471)

# Getting the type of 'Legendre'
Legendre_176473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member 'domain' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176473, 'domain', array_call_result_176472)

# Assigning a Call to a Name (line 1834):

# Call to array(...): (line 1834)
# Processing the call arguments (line 1834)
# Getting the type of 'legdomain' (line 1834)
legdomain_176476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 22), 'legdomain', False)
# Processing the call keyword arguments (line 1834)
kwargs_176477 = {}
# Getting the type of 'np' (line 1834)
np_176474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1834)
array_176475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 13), np_176474, 'array')
# Calling array(args, kwargs) (line 1834)
array_call_result_176478 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 13), array_176475, *[legdomain_176476], **kwargs_176477)

# Getting the type of 'Legendre'
Legendre_176479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legendre')
# Setting the type of the member 'window' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legendre_176479, 'window', array_call_result_176478)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
