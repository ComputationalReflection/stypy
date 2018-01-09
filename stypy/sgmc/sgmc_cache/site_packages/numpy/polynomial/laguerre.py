
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Objects for dealing with Laguerre series.
3: 
4: This module provides a number of objects (mostly functions) useful for
5: dealing with Laguerre series, including a `Laguerre` class that
6: encapsulates the usual arithmetic operations.  (General information
7: on how this module represents and works with such polynomials is in the
8: docstring for its "parent" sub-package, `numpy.polynomial`).
9: 
10: Constants
11: ---------
12: - `lagdomain` -- Laguerre series default domain, [-1,1].
13: - `lagzero` -- Laguerre series that evaluates identically to 0.
14: - `lagone` -- Laguerre series that evaluates identically to 1.
15: - `lagx` -- Laguerre series for the identity map, ``f(x) = x``.
16: 
17: Arithmetic
18: ----------
19: - `lagmulx` -- multiply a Laguerre series in ``P_i(x)`` by ``x``.
20: - `lagadd` -- add two Laguerre series.
21: - `lagsub` -- subtract one Laguerre series from another.
22: - `lagmul` -- multiply two Laguerre series.
23: - `lagdiv` -- divide one Laguerre series by another.
24: - `lagval` -- evaluate a Laguerre series at given points.
25: - `lagval2d` -- evaluate a 2D Laguerre series at given points.
26: - `lagval3d` -- evaluate a 3D Laguerre series at given points.
27: - `laggrid2d` -- evaluate a 2D Laguerre series on a Cartesian product.
28: - `laggrid3d` -- evaluate a 3D Laguerre series on a Cartesian product.
29: 
30: Calculus
31: --------
32: - `lagder` -- differentiate a Laguerre series.
33: - `lagint` -- integrate a Laguerre series.
34: 
35: Misc Functions
36: --------------
37: - `lagfromroots` -- create a Laguerre series with specified roots.
38: - `lagroots` -- find the roots of a Laguerre series.
39: - `lagvander` -- Vandermonde-like matrix for Laguerre polynomials.
40: - `lagvander2d` -- Vandermonde-like matrix for 2D power series.
41: - `lagvander3d` -- Vandermonde-like matrix for 3D power series.
42: - `laggauss` -- Gauss-Laguerre quadrature, points and weights.
43: - `lagweight` -- Laguerre weight function.
44: - `lagcompanion` -- symmetrized companion matrix in Laguerre form.
45: - `lagfit` -- least-squares fit returning a Laguerre series.
46: - `lagtrim` -- trim leading coefficients from a Laguerre series.
47: - `lagline` -- Laguerre series of given straight line.
48: - `lag2poly` -- convert a Laguerre series to a polynomial.
49: - `poly2lag` -- convert a polynomial to a Laguerre series.
50: 
51: Classes
52: -------
53: - `Laguerre` -- A Laguerre series class.
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
70:     'lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd',
71:     'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder',
72:     'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander',
73:     'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d',
74:     'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion',
75:     'laggauss', 'lagweight']
76: 
77: lagtrim = pu.trimcoef
78: 
79: 
80: def poly2lag(pol):
81:     '''
82:     poly2lag(pol)
83: 
84:     Convert a polynomial to a Laguerre series.
85: 
86:     Convert an array representing the coefficients of a polynomial (relative
87:     to the "standard" basis) ordered from lowest degree to highest, to an
88:     array of the coefficients of the equivalent Laguerre series, ordered
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
99:         1-D array containing the coefficients of the equivalent Laguerre
100:         series.
101: 
102:     See Also
103:     --------
104:     lag2poly
105: 
106:     Notes
107:     -----
108:     The easy way to do conversions between polynomial basis sets
109:     is to use the convert method of a class instance.
110: 
111:     Examples
112:     --------
113:     >>> from numpy.polynomial.laguerre import poly2lag
114:     >>> poly2lag(np.arange(4))
115:     array([ 23., -63.,  58., -18.])
116: 
117:     '''
118:     [pol] = pu.as_series([pol])
119:     deg = len(pol) - 1
120:     res = 0
121:     for i in range(deg, -1, -1):
122:         res = lagadd(lagmulx(res), pol[i])
123:     return res
124: 
125: 
126: def lag2poly(c):
127:     '''
128:     Convert a Laguerre series to a polynomial.
129: 
130:     Convert an array representing the coefficients of a Laguerre series,
131:     ordered from lowest degree to highest, to an array of the coefficients
132:     of the equivalent polynomial (relative to the "standard" basis) ordered
133:     from lowest to highest degree.
134: 
135:     Parameters
136:     ----------
137:     c : array_like
138:         1-D array containing the Laguerre series coefficients, ordered
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
150:     poly2lag
151: 
152:     Notes
153:     -----
154:     The easy way to do conversions between polynomial basis sets
155:     is to use the convert method of a class instance.
156: 
157:     Examples
158:     --------
159:     >>> from numpy.polynomial.laguerre import lag2poly
160:     >>> lag2poly([ 23., -63.,  58., -18.])
161:     array([ 0.,  1.,  2.,  3.])
162: 
163:     '''
164:     from .polynomial import polyadd, polysub, polymulx
165: 
166:     [c] = pu.as_series([c])
167:     n = len(c)
168:     if n == 1:
169:         return c
170:     else:
171:         c0 = c[-2]
172:         c1 = c[-1]
173:         # i is the current degree of c1
174:         for i in range(n - 1, 1, -1):
175:             tmp = c0
176:             c0 = polysub(c[i - 2], (c1*(i - 1))/i)
177:             c1 = polyadd(tmp, polysub((2*i - 1)*c1, polymulx(c1))/i)
178:         return polyadd(c0, polysub(c1, polymulx(c1)))
179: 
180: #
181: # These are constant arrays are of integer type so as to be compatible
182: # with the widest range of other types, such as Decimal.
183: #
184: 
185: # Laguerre
186: lagdomain = np.array([0, 1])
187: 
188: # Laguerre coefficients representing zero.
189: lagzero = np.array([0])
190: 
191: # Laguerre coefficients representing one.
192: lagone = np.array([1])
193: 
194: # Laguerre coefficients representing the identity x.
195: lagx = np.array([1, -1])
196: 
197: 
198: def lagline(off, scl):
199:     '''
200:     Laguerre series whose graph is a straight line.
201: 
202: 
203: 
204:     Parameters
205:     ----------
206:     off, scl : scalars
207:         The specified line is given by ``off + scl*x``.
208: 
209:     Returns
210:     -------
211:     y : ndarray
212:         This module's representation of the Laguerre series for
213:         ``off + scl*x``.
214: 
215:     See Also
216:     --------
217:     polyline, chebline
218: 
219:     Examples
220:     --------
221:     >>> from numpy.polynomial.laguerre import lagline, lagval
222:     >>> lagval(0,lagline(3, 2))
223:     3.0
224:     >>> lagval(1,lagline(3, 2))
225:     5.0
226: 
227:     '''
228:     if scl != 0:
229:         return np.array([off + scl, -scl])
230:     else:
231:         return np.array([off])
232: 
233: 
234: def lagfromroots(roots):
235:     '''
236:     Generate a Laguerre series with given roots.
237: 
238:     The function returns the coefficients of the polynomial
239: 
240:     .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
241: 
242:     in Laguerre form, where the `r_n` are the roots specified in `roots`.
243:     If a zero has multiplicity n, then it must appear in `roots` n times.
244:     For instance, if 2 is a root of multiplicity three and 3 is a root of
245:     multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
246:     roots can appear in any order.
247: 
248:     If the returned coefficients are `c`, then
249: 
250:     .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)
251: 
252:     The coefficient of the last term is not generally 1 for monic
253:     polynomials in Laguerre form.
254: 
255:     Parameters
256:     ----------
257:     roots : array_like
258:         Sequence containing the roots.
259: 
260:     Returns
261:     -------
262:     out : ndarray
263:         1-D array of coefficients.  If all roots are real then `out` is a
264:         real array, if some of the roots are complex, then `out` is complex
265:         even if all the coefficients in the result are real (see Examples
266:         below).
267: 
268:     See Also
269:     --------
270:     polyfromroots, legfromroots, chebfromroots, hermfromroots,
271:     hermefromroots.
272: 
273:     Examples
274:     --------
275:     >>> from numpy.polynomial.laguerre import lagfromroots, lagval
276:     >>> coef = lagfromroots((-1, 0, 1))
277:     >>> lagval((-1, 0, 1), coef)
278:     array([ 0.,  0.,  0.])
279:     >>> coef = lagfromroots((-1j, 1j))
280:     >>> lagval((-1j, 1j), coef)
281:     array([ 0.+0.j,  0.+0.j])
282: 
283:     '''
284:     if len(roots) == 0:
285:         return np.ones(1)
286:     else:
287:         [roots] = pu.as_series([roots], trim=False)
288:         roots.sort()
289:         p = [lagline(-r, 1) for r in roots]
290:         n = len(p)
291:         while n > 1:
292:             m, r = divmod(n, 2)
293:             tmp = [lagmul(p[i], p[i+m]) for i in range(m)]
294:             if r:
295:                 tmp[0] = lagmul(tmp[0], p[-1])
296:             p = tmp
297:             n = m
298:         return p[0]
299: 
300: 
301: def lagadd(c1, c2):
302:     '''
303:     Add one Laguerre series to another.
304: 
305:     Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
306:     are sequences of coefficients ordered from lowest order term to
307:     highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
308: 
309:     Parameters
310:     ----------
311:     c1, c2 : array_like
312:         1-D arrays of Laguerre series coefficients ordered from low to
313:         high.
314: 
315:     Returns
316:     -------
317:     out : ndarray
318:         Array representing the Laguerre series of their sum.
319: 
320:     See Also
321:     --------
322:     lagsub, lagmul, lagdiv, lagpow
323: 
324:     Notes
325:     -----
326:     Unlike multiplication, division, etc., the sum of two Laguerre series
327:     is a Laguerre series (without having to "reproject" the result onto
328:     the basis set) so addition, just like that of "standard" polynomials,
329:     is simply "component-wise."
330: 
331:     Examples
332:     --------
333:     >>> from numpy.polynomial.laguerre import lagadd
334:     >>> lagadd([1, 2, 3], [1, 2, 3, 4])
335:     array([ 2.,  4.,  6.,  4.])
336: 
337: 
338:     '''
339:     # c1, c2 are trimmed copies
340:     [c1, c2] = pu.as_series([c1, c2])
341:     if len(c1) > len(c2):
342:         c1[:c2.size] += c2
343:         ret = c1
344:     else:
345:         c2[:c1.size] += c1
346:         ret = c2
347:     return pu.trimseq(ret)
348: 
349: 
350: def lagsub(c1, c2):
351:     '''
352:     Subtract one Laguerre series from another.
353: 
354:     Returns the difference of two Laguerre series `c1` - `c2`.  The
355:     sequences of coefficients are from lowest order term to highest, i.e.,
356:     [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
357: 
358:     Parameters
359:     ----------
360:     c1, c2 : array_like
361:         1-D arrays of Laguerre series coefficients ordered from low to
362:         high.
363: 
364:     Returns
365:     -------
366:     out : ndarray
367:         Of Laguerre series coefficients representing their difference.
368: 
369:     See Also
370:     --------
371:     lagadd, lagmul, lagdiv, lagpow
372: 
373:     Notes
374:     -----
375:     Unlike multiplication, division, etc., the difference of two Laguerre
376:     series is a Laguerre series (without having to "reproject" the result
377:     onto the basis set) so subtraction, just like that of "standard"
378:     polynomials, is simply "component-wise."
379: 
380:     Examples
381:     --------
382:     >>> from numpy.polynomial.laguerre import lagsub
383:     >>> lagsub([1, 2, 3, 4], [1, 2, 3])
384:     array([ 0.,  0.,  0.,  4.])
385: 
386:     '''
387:     # c1, c2 are trimmed copies
388:     [c1, c2] = pu.as_series([c1, c2])
389:     if len(c1) > len(c2):
390:         c1[:c2.size] -= c2
391:         ret = c1
392:     else:
393:         c2 = -c2
394:         c2[:c1.size] += c1
395:         ret = c2
396:     return pu.trimseq(ret)
397: 
398: 
399: def lagmulx(c):
400:     '''Multiply a Laguerre series by x.
401: 
402:     Multiply the Laguerre series `c` by x, where x is the independent
403:     variable.
404: 
405: 
406:     Parameters
407:     ----------
408:     c : array_like
409:         1-D array of Laguerre series coefficients ordered from low to
410:         high.
411: 
412:     Returns
413:     -------
414:     out : ndarray
415:         Array representing the result of the multiplication.
416: 
417:     Notes
418:     -----
419:     The multiplication uses the recursion relationship for Laguerre
420:     polynomials in the form
421: 
422:     .. math::
423: 
424:     xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))
425: 
426:     Examples
427:     --------
428:     >>> from numpy.polynomial.laguerre import lagmulx
429:     >>> lagmulx([1, 2, 3])
430:     array([ -1.,  -1.,  11.,  -9.])
431: 
432:     '''
433:     # c is a trimmed copy
434:     [c] = pu.as_series([c])
435:     # The zero series needs special treatment
436:     if len(c) == 1 and c[0] == 0:
437:         return c
438: 
439:     prd = np.empty(len(c) + 1, dtype=c.dtype)
440:     prd[0] = c[0]
441:     prd[1] = -c[0]
442:     for i in range(1, len(c)):
443:         prd[i + 1] = -c[i]*(i + 1)
444:         prd[i] += c[i]*(2*i + 1)
445:         prd[i - 1] -= c[i]*i
446:     return prd
447: 
448: 
449: def lagmul(c1, c2):
450:     '''
451:     Multiply one Laguerre series by another.
452: 
453:     Returns the product of two Laguerre series `c1` * `c2`.  The arguments
454:     are sequences of coefficients, from lowest order "term" to highest,
455:     e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
456: 
457:     Parameters
458:     ----------
459:     c1, c2 : array_like
460:         1-D arrays of Laguerre series coefficients ordered from low to
461:         high.
462: 
463:     Returns
464:     -------
465:     out : ndarray
466:         Of Laguerre series coefficients representing their product.
467: 
468:     See Also
469:     --------
470:     lagadd, lagsub, lagdiv, lagpow
471: 
472:     Notes
473:     -----
474:     In general, the (polynomial) product of two C-series results in terms
475:     that are not in the Laguerre polynomial basis set.  Thus, to express
476:     the product as a Laguerre series, it is necessary to "reproject" the
477:     product onto said basis set, which may produce "unintuitive" (but
478:     correct) results; see Examples section below.
479: 
480:     Examples
481:     --------
482:     >>> from numpy.polynomial.laguerre import lagmul
483:     >>> lagmul([1, 2, 3], [0, 1, 2])
484:     array([  8., -13.,  38., -51.,  36.])
485: 
486:     '''
487:     # s1, s2 are trimmed copies
488:     [c1, c2] = pu.as_series([c1, c2])
489: 
490:     if len(c1) > len(c2):
491:         c = c2
492:         xs = c1
493:     else:
494:         c = c1
495:         xs = c2
496: 
497:     if len(c) == 1:
498:         c0 = c[0]*xs
499:         c1 = 0
500:     elif len(c) == 2:
501:         c0 = c[0]*xs
502:         c1 = c[1]*xs
503:     else:
504:         nd = len(c)
505:         c0 = c[-2]*xs
506:         c1 = c[-1]*xs
507:         for i in range(3, len(c) + 1):
508:             tmp = c0
509:             nd = nd - 1
510:             c0 = lagsub(c[-i]*xs, (c1*(nd - 1))/nd)
511:             c1 = lagadd(tmp, lagsub((2*nd - 1)*c1, lagmulx(c1))/nd)
512:     return lagadd(c0, lagsub(c1, lagmulx(c1)))
513: 
514: 
515: def lagdiv(c1, c2):
516:     '''
517:     Divide one Laguerre series by another.
518: 
519:     Returns the quotient-with-remainder of two Laguerre series
520:     `c1` / `c2`.  The arguments are sequences of coefficients from lowest
521:     order "term" to highest, e.g., [1,2,3] represents the series
522:     ``P_0 + 2*P_1 + 3*P_2``.
523: 
524:     Parameters
525:     ----------
526:     c1, c2 : array_like
527:         1-D arrays of Laguerre series coefficients ordered from low to
528:         high.
529: 
530:     Returns
531:     -------
532:     [quo, rem] : ndarrays
533:         Of Laguerre series coefficients representing the quotient and
534:         remainder.
535: 
536:     See Also
537:     --------
538:     lagadd, lagsub, lagmul, lagpow
539: 
540:     Notes
541:     -----
542:     In general, the (polynomial) division of one Laguerre series by another
543:     results in quotient and remainder terms that are not in the Laguerre
544:     polynomial basis set.  Thus, to express these results as a Laguerre
545:     series, it is necessary to "reproject" the results onto the Laguerre
546:     basis set, which may produce "unintuitive" (but correct) results; see
547:     Examples section below.
548: 
549:     Examples
550:     --------
551:     >>> from numpy.polynomial.laguerre import lagdiv
552:     >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
553:     (array([ 1.,  2.,  3.]), array([ 0.]))
554:     >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
555:     (array([ 1.,  2.,  3.]), array([ 1.,  1.]))
556: 
557:     '''
558:     # c1, c2 are trimmed copies
559:     [c1, c2] = pu.as_series([c1, c2])
560:     if c2[-1] == 0:
561:         raise ZeroDivisionError()
562: 
563:     lc1 = len(c1)
564:     lc2 = len(c2)
565:     if lc1 < lc2:
566:         return c1[:1]*0, c1
567:     elif lc2 == 1:
568:         return c1/c2[-1], c1[:1]*0
569:     else:
570:         quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
571:         rem = c1
572:         for i in range(lc1 - lc2, - 1, -1):
573:             p = lagmul([0]*i + [1], c2)
574:             q = rem[-1]/p[-1]
575:             rem = rem[:-1] - q*p[:-1]
576:             quo[i] = q
577:         return quo, pu.trimseq(rem)
578: 
579: 
580: def lagpow(c, pow, maxpower=16):
581:     '''Raise a Laguerre series to a power.
582: 
583:     Returns the Laguerre series `c` raised to the power `pow`. The
584:     argument `c` is a sequence of coefficients ordered from low to high.
585:     i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``
586: 
587:     Parameters
588:     ----------
589:     c : array_like
590:         1-D array of Laguerre series coefficients ordered from low to
591:         high.
592:     pow : integer
593:         Power to which the series will be raised
594:     maxpower : integer, optional
595:         Maximum power allowed. This is mainly to limit growth of the series
596:         to unmanageable size. Default is 16
597: 
598:     Returns
599:     -------
600:     coef : ndarray
601:         Laguerre series of power.
602: 
603:     See Also
604:     --------
605:     lagadd, lagsub, lagmul, lagdiv
606: 
607:     Examples
608:     --------
609:     >>> from numpy.polynomial.laguerre import lagpow
610:     >>> lagpow([1, 2, 3], 2)
611:     array([ 14., -16.,  56., -72.,  54.])
612: 
613:     '''
614:     # c is a trimmed copy
615:     [c] = pu.as_series([c])
616:     power = int(pow)
617:     if power != pow or power < 0:
618:         raise ValueError("Power must be a non-negative integer.")
619:     elif maxpower is not None and power > maxpower:
620:         raise ValueError("Power is too large")
621:     elif power == 0:
622:         return np.array([1], dtype=c.dtype)
623:     elif power == 1:
624:         return c
625:     else:
626:         # This can be made more efficient by using powers of two
627:         # in the usual way.
628:         prd = c
629:         for i in range(2, power + 1):
630:             prd = lagmul(prd, c)
631:         return prd
632: 
633: 
634: def lagder(c, m=1, scl=1, axis=0):
635:     '''
636:     Differentiate a Laguerre series.
637: 
638:     Returns the Laguerre series coefficients `c` differentiated `m` times
639:     along `axis`.  At each iteration the result is multiplied by `scl` (the
640:     scaling factor is for use in a linear change of variable). The argument
641:     `c` is an array of coefficients from low to high degree along each
642:     axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
643:     while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
644:     2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
645:     ``y``.
646: 
647:     Parameters
648:     ----------
649:     c : array_like
650:         Array of Laguerre series coefficients. If `c` is multidimensional
651:         the different axis correspond to different variables with the
652:         degree in each axis given by the corresponding index.
653:     m : int, optional
654:         Number of derivatives taken, must be non-negative. (Default: 1)
655:     scl : scalar, optional
656:         Each differentiation is multiplied by `scl`.  The end result is
657:         multiplication by ``scl**m``.  This is for use in a linear change of
658:         variable. (Default: 1)
659:     axis : int, optional
660:         Axis over which the derivative is taken. (Default: 0).
661: 
662:         .. versionadded:: 1.7.0
663: 
664:     Returns
665:     -------
666:     der : ndarray
667:         Laguerre series of the derivative.
668: 
669:     See Also
670:     --------
671:     lagint
672: 
673:     Notes
674:     -----
675:     In general, the result of differentiating a Laguerre series does not
676:     resemble the same operation on a power series. Thus the result of this
677:     function may be "unintuitive," albeit correct; see Examples section
678:     below.
679: 
680:     Examples
681:     --------
682:     >>> from numpy.polynomial.laguerre import lagder
683:     >>> lagder([ 1.,  1.,  1., -3.])
684:     array([ 1.,  2.,  3.])
685:     >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)
686:     array([ 1.,  2.,  3.])
687: 
688:     '''
689:     c = np.array(c, ndmin=1, copy=1)
690:     if c.dtype.char in '?bBhHiIlLqQpP':
691:         c = c.astype(np.double)
692:     cnt, iaxis = [int(t) for t in [m, axis]]
693: 
694:     if cnt != m:
695:         raise ValueError("The order of derivation must be integer")
696:     if cnt < 0:
697:         raise ValueError("The order of derivation must be non-negative")
698:     if iaxis != axis:
699:         raise ValueError("The axis must be integer")
700:     if not -c.ndim <= iaxis < c.ndim:
701:         raise ValueError("The axis is out of range")
702:     if iaxis < 0:
703:         iaxis += c.ndim
704: 
705:     if cnt == 0:
706:         return c
707: 
708:     c = np.rollaxis(c, iaxis)
709:     n = len(c)
710:     if cnt >= n:
711:         c = c[:1]*0
712:     else:
713:         for i in range(cnt):
714:             n = n - 1
715:             c *= scl
716:             der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
717:             for j in range(n, 1, -1):
718:                 der[j - 1] = -c[j]
719:                 c[j - 1] += c[j]
720:             der[0] = -c[1]
721:             c = der
722:     c = np.rollaxis(c, 0, iaxis + 1)
723:     return c
724: 
725: 
726: def lagint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
727:     '''
728:     Integrate a Laguerre series.
729: 
730:     Returns the Laguerre series coefficients `c` integrated `m` times from
731:     `lbnd` along `axis`. At each iteration the resulting series is
732:     **multiplied** by `scl` and an integration constant, `k`, is added.
733:     The scaling factor is for use in a linear change of variable.  ("Buyer
734:     beware": note that, depending on what one is doing, one may want `scl`
735:     to be the reciprocal of what one might expect; for more information,
736:     see the Notes section below.)  The argument `c` is an array of
737:     coefficients from low to high degree along each axis, e.g., [1,2,3]
738:     represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
739:     represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
740:     2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
741: 
742: 
743:     Parameters
744:     ----------
745:     c : array_like
746:         Array of Laguerre series coefficients. If `c` is multidimensional
747:         the different axis correspond to different variables with the
748:         degree in each axis given by the corresponding index.
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
770:         Laguerre series coefficients of the integral.
771: 
772:     Raises
773:     ------
774:     ValueError
775:         If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
776:         ``np.isscalar(scl) == False``.
777: 
778:     See Also
779:     --------
780:     lagder
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
797:     >>> from numpy.polynomial.laguerre import lagint
798:     >>> lagint([1,2,3])
799:     array([ 1.,  1.,  1., -3.])
800:     >>> lagint([1,2,3], m=2)
801:     array([ 1.,  0.,  0., -4.,  3.])
802:     >>> lagint([1,2,3], k=1)
803:     array([ 2.,  1.,  1., -3.])
804:     >>> lagint([1,2,3], lbnd=-1)
805:     array([ 11.5,   1. ,   1. ,  -3. ])
806:     >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)
807:     array([ 11.16666667,  -5.        ,  -3.        ,   2.        ])
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
842:             tmp[0] = c[0]
843:             tmp[1] = -c[0]
844:             for j in range(1, n):
845:                 tmp[j] += c[j]
846:                 tmp[j + 1] = -c[j]
847:             tmp[0] += k[i] - lagval(lbnd, tmp)
848:             c = tmp
849:     c = np.rollaxis(c, 0, iaxis + 1)
850:     return c
851: 
852: 
853: def lagval(x, c, tensor=True):
854:     '''
855:     Evaluate a Laguerre series at points x.
856: 
857:     If `c` is of length `n + 1`, this function returns the value:
858: 
859:     .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)
860: 
861:     The parameter `x` is converted to an array only if it is a tuple or a
862:     list, otherwise it is treated as a scalar. In either case, either `x`
863:     or its elements must support multiplication and addition both with
864:     themselves and with the elements of `c`.
865: 
866:     If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
867:     `c` is multidimensional, then the shape of the result depends on the
868:     value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
869:     x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
870:     scalars have shape (,).
871: 
872:     Trailing zeros in the coefficients will be used in the evaluation, so
873:     they should be avoided if efficiency is a concern.
874: 
875:     Parameters
876:     ----------
877:     x : array_like, compatible object
878:         If `x` is a list or tuple, it is converted to an ndarray, otherwise
879:         it is left unchanged and treated as a scalar. In either case, `x`
880:         or its elements must support addition and multiplication with
881:         with themselves and with the elements of `c`.
882:     c : array_like
883:         Array of coefficients ordered so that the coefficients for terms of
884:         degree n are contained in c[n]. If `c` is multidimensional the
885:         remaining indices enumerate multiple polynomials. In the two
886:         dimensional case the coefficients may be thought of as stored in
887:         the columns of `c`.
888:     tensor : boolean, optional
889:         If True, the shape of the coefficient array is extended with ones
890:         on the right, one for each dimension of `x`. Scalars have dimension 0
891:         for this action. The result is that every column of coefficients in
892:         `c` is evaluated for every element of `x`. If False, `x` is broadcast
893:         over the columns of `c` for the evaluation.  This keyword is useful
894:         when `c` is multidimensional. The default value is True.
895: 
896:         .. versionadded:: 1.7.0
897: 
898:     Returns
899:     -------
900:     values : ndarray, algebra_like
901:         The shape of the return value is described above.
902: 
903:     See Also
904:     --------
905:     lagval2d, laggrid2d, lagval3d, laggrid3d
906: 
907:     Notes
908:     -----
909:     The evaluation uses Clenshaw recursion, aka synthetic division.
910: 
911:     Examples
912:     --------
913:     >>> from numpy.polynomial.laguerre import lagval
914:     >>> coef = [1,2,3]
915:     >>> lagval(1, coef)
916:     -0.5
917:     >>> lagval([[1,2],[3,4]], coef)
918:     array([[-0.5, -4. ],
919:            [-4.5, -2. ]])
920: 
921:     '''
922:     c = np.array(c, ndmin=1, copy=0)
923:     if c.dtype.char in '?bBhHiIlLqQpP':
924:         c = c.astype(np.double)
925:     if isinstance(x, (tuple, list)):
926:         x = np.asarray(x)
927:     if isinstance(x, np.ndarray) and tensor:
928:         c = c.reshape(c.shape + (1,)*x.ndim)
929: 
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
943:             c0 = c[-i] - (c1*(nd - 1))/nd
944:             c1 = tmp + (c1*((2*nd - 1) - x))/nd
945:     return c0 + c1*(1 - x)
946: 
947: 
948: def lagval2d(x, y, c):
949:     '''
950:     Evaluate a 2-D Laguerre series at points (x, y).
951: 
952:     This function returns the values:
953: 
954:     .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)
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
986:     lagval, laggrid2d, lagval3d, laggrid3d
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
999:     c = lagval(x, c)
1000:     c = lagval(y, c, tensor=False)
1001:     return c
1002: 
1003: 
1004: def laggrid2d(x, y, c):
1005:     '''
1006:     Evaluate a 2-D Laguerre series on the Cartesian product of x and y.
1007: 
1008:     This function returns the values:
1009: 
1010:     .. math:: p(a,b) = \sum_{i,j} c_{i,j} * L_i(a) * L_j(b)
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
1023:     x.shape + y.shape.
1024: 
1025:     Parameters
1026:     ----------
1027:     x, y : array_like, compatible objects
1028:         The two dimensional series is evaluated at the points in the
1029:         Cartesian product of `x` and `y`.  If `x` or `y` is a list or
1030:         tuple, it is first converted to an ndarray, otherwise it is left
1031:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
1032:     c : array_like
1033:         Array of coefficients ordered so that the coefficient of the term of
1034:         multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
1035:         greater than two the remaining indices enumerate multiple sets of
1036:         coefficients.
1037: 
1038:     Returns
1039:     -------
1040:     values : ndarray, compatible object
1041:         The values of the two dimensional Chebyshev series at points in the
1042:         Cartesian product of `x` and `y`.
1043: 
1044:     See Also
1045:     --------
1046:     lagval, lagval2d, lagval3d, laggrid3d
1047: 
1048:     Notes
1049:     -----
1050: 
1051:     .. versionadded::1.7.0
1052: 
1053:     '''
1054:     c = lagval(x, c)
1055:     c = lagval(y, c)
1056:     return c
1057: 
1058: 
1059: def lagval3d(x, y, z, c):
1060:     '''
1061:     Evaluate a 3-D Laguerre series at points (x, y, z).
1062: 
1063:     This function returns the values:
1064: 
1065:     .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)
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
1094:         The values of the multidimension polynomial on points formed with
1095:         triples of corresponding values from `x`, `y`, and `z`.
1096: 
1097:     See Also
1098:     --------
1099:     lagval, lagval2d, laggrid2d, laggrid3d
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
1112:     c = lagval(x, c)
1113:     c = lagval(y, c, tensor=False)
1114:     c = lagval(z, c, tensor=False)
1115:     return c
1116: 
1117: 
1118: def laggrid3d(x, y, z, c):
1119:     '''
1120:     Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.
1121: 
1122:     This function returns the values:
1123: 
1124:     .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)
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
1163:     lagval, lagval2d, laggrid2d, lagval3d
1164: 
1165:     Notes
1166:     -----
1167: 
1168:     .. versionadded::1.7.0
1169: 
1170:     '''
1171:     c = lagval(x, c)
1172:     c = lagval(y, c)
1173:     c = lagval(z, c)
1174:     return c
1175: 
1176: 
1177: def lagvander(x, deg):
1178:     '''Pseudo-Vandermonde matrix of given degree.
1179: 
1180:     Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
1181:     `x`. The pseudo-Vandermonde matrix is defined by
1182: 
1183:     .. math:: V[..., i] = L_i(x)
1184: 
1185:     where `0 <= i <= deg`. The leading indices of `V` index the elements of
1186:     `x` and the last index is the degree of the Laguerre polynomial.
1187: 
1188:     If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
1189:     array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and
1190:     ``lagval(x, c)`` are the same up to roundoff. This equivalence is
1191:     useful both for least squares fitting and for the evaluation of a large
1192:     number of Laguerre series of the same degree and sample points.
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
1208:         corresponding Laguerre polynomial.  The dtype will be the same as
1209:         the converted `x`.
1210: 
1211:     Examples
1212:     --------
1213:     >>> from numpy.polynomial.laguerre import lagvander
1214:     >>> x = np.array([0, 1, 2])
1215:     >>> lagvander(x, 3)
1216:     array([[ 1.        ,  1.        ,  1.        ,  1.        ],
1217:            [ 1.        ,  0.        , -0.5       , -0.66666667],
1218:            [ 1.        , -1.        , -1.        , -0.33333333]])
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
1233:         v[1] = 1 - x
1234:         for i in range(2, ideg + 1):
1235:             v[i] = (v[i-1]*(2*i - 1 - x) - v[i-2]*(i - 1))/i
1236:     return np.rollaxis(v, 0, v.ndim)
1237: 
1238: 
1239: def lagvander2d(x, y, deg):
1240:     '''Pseudo-Vandermonde matrix of given degrees.
1241: 
1242:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1243:     points `(x, y)`. The pseudo-Vandermonde matrix is defined by
1244: 
1245:     .. math:: V[..., deg[1]*i + j] = L_i(x) * L_j(y),
1246: 
1247:     where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
1248:     `V` index the points `(x, y)` and the last index encodes the degrees of
1249:     the Laguerre polynomials.
1250: 
1251:     If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
1252:     correspond to the elements of a 2-D coefficient array `c` of shape
1253:     (xdeg + 1, ydeg + 1) in the order
1254: 
1255:     .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
1256: 
1257:     and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same
1258:     up to roundoff. This equivalence is useful both for least squares
1259:     fitting and for the evaluation of a large number of 2-D Laguerre
1260:     series of the same degrees and sample points.
1261: 
1262:     Parameters
1263:     ----------
1264:     x, y : array_like
1265:         Arrays of point coordinates, all of the same shape. The dtypes
1266:         will be converted to either float64 or complex128 depending on
1267:         whether any of the elements are complex. Scalars are converted to
1268:         1-D arrays.
1269:     deg : list of ints
1270:         List of maximum degrees of the form [x_deg, y_deg].
1271: 
1272:     Returns
1273:     -------
1274:     vander2d : ndarray
1275:         The shape of the returned matrix is ``x.shape + (order,)``, where
1276:         :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
1277:         as the converted `x` and `y`.
1278: 
1279:     See Also
1280:     --------
1281:     lagvander, lagvander3d. lagval2d, lagval3d
1282: 
1283:     Notes
1284:     -----
1285: 
1286:     .. versionadded::1.7.0
1287: 
1288:     '''
1289:     ideg = [int(d) for d in deg]
1290:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1291:     if is_valid != [1, 1]:
1292:         raise ValueError("degrees must be non-negative integers")
1293:     degx, degy = ideg
1294:     x, y = np.array((x, y), copy=0) + 0.0
1295: 
1296:     vx = lagvander(x, degx)
1297:     vy = lagvander(y, degy)
1298:     v = vx[..., None]*vy[..., None,:]
1299:     return v.reshape(v.shape[:-2] + (-1,))
1300: 
1301: 
1302: def lagvander3d(x, y, z, deg):
1303:     '''Pseudo-Vandermonde matrix of given degrees.
1304: 
1305:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1306:     points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
1307:     then The pseudo-Vandermonde matrix is defined by
1308: 
1309:     .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),
1310: 
1311:     where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
1312:     indices of `V` index the points `(x, y, z)` and the last index encodes
1313:     the degrees of the Laguerre polynomials.
1314: 
1315:     If ``V = lagvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
1316:     of `V` correspond to the elements of a 3-D coefficient array `c` of
1317:     shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
1318: 
1319:     .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
1320: 
1321:     and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the
1322:     same up to roundoff. This equivalence is useful both for least squares
1323:     fitting and for the evaluation of a large number of 3-D Laguerre
1324:     series of the same degrees and sample points.
1325: 
1326:     Parameters
1327:     ----------
1328:     x, y, z : array_like
1329:         Arrays of point coordinates, all of the same shape. The dtypes will
1330:         be converted to either float64 or complex128 depending on whether
1331:         any of the elements are complex. Scalars are converted to 1-D
1332:         arrays.
1333:     deg : list of ints
1334:         List of maximum degrees of the form [x_deg, y_deg, z_deg].
1335: 
1336:     Returns
1337:     -------
1338:     vander3d : ndarray
1339:         The shape of the returned matrix is ``x.shape + (order,)``, where
1340:         :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
1341:         be the same as the converted `x`, `y`, and `z`.
1342: 
1343:     See Also
1344:     --------
1345:     lagvander, lagvander3d. lagval2d, lagval3d
1346: 
1347:     Notes
1348:     -----
1349: 
1350:     .. versionadded::1.7.0
1351: 
1352:     '''
1353:     ideg = [int(d) for d in deg]
1354:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1355:     if is_valid != [1, 1, 1]:
1356:         raise ValueError("degrees must be non-negative integers")
1357:     degx, degy, degz = ideg
1358:     x, y, z = np.array((x, y, z), copy=0) + 0.0
1359: 
1360:     vx = lagvander(x, degx)
1361:     vy = lagvander(y, degy)
1362:     vz = lagvander(z, degz)
1363:     v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
1364:     return v.reshape(v.shape[:-3] + (-1,))
1365: 
1366: 
1367: def lagfit(x, y, deg, rcond=None, full=False, w=None):
1368:     '''
1369:     Least squares fit of Laguerre series to data.
1370: 
1371:     Return the coefficients of a Laguerre series of degree `deg` that is the
1372:     least squares fit to the data values `y` given at points `x`. If `y` is
1373:     1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
1374:     fits are done, one for each column of `y`, and the resulting
1375:     coefficients are stored in the corresponding columns of a 2-D return.
1376:     The fitted polynomial(s) are in the form
1377: 
1378:     .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),
1379: 
1380:     where `n` is `deg`.
1381: 
1382:     Parameters
1383:     ----------
1384:     x : array_like, shape (M,)
1385:         x-coordinates of the M sample points ``(x[i], y[i])``.
1386:     y : array_like, shape (M,) or (M, K)
1387:         y-coordinates of the sample points. Several data sets of sample
1388:         points sharing the same x-coordinates can be fitted at once by
1389:         passing in a 2D-array that contains one dataset per column.
1390:     deg : int or 1-D array_like
1391:         Degree(s) of the fitting polynomials. If `deg` is a single integer
1392:         all terms up to and including the `deg`'th term are included in the
1393:         fit. For Numpy versions >= 1.11 a list of integers specifying the
1394:         degrees of the terms to include may be used instead.
1395:     rcond : float, optional
1396:         Relative condition number of the fit. Singular values smaller than
1397:         this relative to the largest singular value will be ignored. The
1398:         default value is len(x)*eps, where eps is the relative precision of
1399:         the float type, about 2e-16 in most cases.
1400:     full : bool, optional
1401:         Switch determining nature of return value. When it is False (the
1402:         default) just the coefficients are returned, when True diagnostic
1403:         information from the singular value decomposition is also returned.
1404:     w : array_like, shape (`M`,), optional
1405:         Weights. If not None, the contribution of each point
1406:         ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
1407:         weights are chosen so that the errors of the products ``w[i]*y[i]``
1408:         all have the same variance.  The default value is None.
1409: 
1410:     Returns
1411:     -------
1412:     coef : ndarray, shape (M,) or (M, K)
1413:         Laguerre coefficients ordered from low to high. If `y` was 2-D,
1414:         the coefficients for the data in column k  of `y` are in column
1415:         `k`.
1416: 
1417:     [residuals, rank, singular_values, rcond] : list
1418:         These values are only returned if `full` = True
1419: 
1420:         resid -- sum of squared residuals of the least squares fit
1421:         rank -- the numerical rank of the scaled Vandermonde matrix
1422:         sv -- singular values of the scaled Vandermonde matrix
1423:         rcond -- value of `rcond`.
1424: 
1425:         For more details, see `linalg.lstsq`.
1426: 
1427:     Warns
1428:     -----
1429:     RankWarning
1430:         The rank of the coefficient matrix in the least-squares fit is
1431:         deficient. The warning is only raised if `full` = False.  The
1432:         warnings can be turned off by
1433: 
1434:         >>> import warnings
1435:         >>> warnings.simplefilter('ignore', RankWarning)
1436: 
1437:     See Also
1438:     --------
1439:     chebfit, legfit, polyfit, hermfit, hermefit
1440:     lagval : Evaluates a Laguerre series.
1441:     lagvander : pseudo Vandermonde matrix of Laguerre series.
1442:     lagweight : Laguerre weight function.
1443:     linalg.lstsq : Computes a least-squares fit from the matrix.
1444:     scipy.interpolate.UnivariateSpline : Computes spline fits.
1445: 
1446:     Notes
1447:     -----
1448:     The solution is the coefficients of the Laguerre series `p` that
1449:     minimizes the sum of the weighted squared errors
1450: 
1451:     .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
1452: 
1453:     where the :math:`w_j` are the weights. This problem is solved by
1454:     setting up as the (typically) overdetermined matrix equation
1455: 
1456:     .. math:: V(x) * c = w * y,
1457: 
1458:     where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
1459:     coefficients to be solved for, `w` are the weights, and `y` are the
1460:     observed values.  This equation is then solved using the singular value
1461:     decomposition of `V`.
1462: 
1463:     If some of the singular values of `V` are so small that they are
1464:     neglected, then a `RankWarning` will be issued. This means that the
1465:     coefficient values may be poorly determined. Using a lower order fit
1466:     will usually get rid of the warning.  The `rcond` parameter can also be
1467:     set to a value smaller than its default, but the resulting fit may be
1468:     spurious and have large contributions from roundoff error.
1469: 
1470:     Fits using Laguerre series are probably most useful when the data can
1471:     be approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the Laguerre
1472:     weight. In that case the weight ``sqrt(w(x[i])`` should be used
1473:     together with data values ``y[i]/sqrt(w(x[i])``. The weight function is
1474:     available as `lagweight`.
1475: 
1476:     References
1477:     ----------
1478:     .. [1] Wikipedia, "Curve fitting",
1479:            http://en.wikipedia.org/wiki/Curve_fitting
1480: 
1481:     Examples
1482:     --------
1483:     >>> from numpy.polynomial.laguerre import lagfit, lagval
1484:     >>> x = np.linspace(0, 10)
1485:     >>> err = np.random.randn(len(x))/10
1486:     >>> y = lagval(x, [1, 2, 3]) + err
1487:     >>> lagfit(x, y, 2)
1488:     array([ 0.96971004,  2.00193749,  3.00288744])
1489: 
1490:     '''
1491:     x = np.asarray(x) + 0.0
1492:     y = np.asarray(y) + 0.0
1493:     deg = np.asarray(deg)
1494: 
1495:     # check arguments.
1496:     if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
1497:         raise TypeError("deg must be an int or non-empty 1-D array of int")
1498:     if deg.min() < 0:
1499:         raise ValueError("expected deg >= 0")
1500:     if x.ndim != 1:
1501:         raise TypeError("expected 1D vector for x")
1502:     if x.size == 0:
1503:         raise TypeError("expected non-empty vector for x")
1504:     if y.ndim < 1 or y.ndim > 2:
1505:         raise TypeError("expected 1D or 2D array for y")
1506:     if len(x) != len(y):
1507:         raise TypeError("expected x and y to have same length")
1508: 
1509:     if deg.ndim == 0:
1510:         lmax = deg
1511:         order = lmax + 1
1512:         van = lagvander(x, lmax)
1513:     else:
1514:         deg = np.sort(deg)
1515:         lmax = deg[-1]
1516:         order = len(deg)
1517:         van = lagvander(x, lmax)[:, deg]
1518: 
1519:     # set up the least squares matrices in transposed form
1520:     lhs = van.T
1521:     rhs = y.T
1522:     if w is not None:
1523:         w = np.asarray(w) + 0.0
1524:         if w.ndim != 1:
1525:             raise TypeError("expected 1D vector for w")
1526:         if len(x) != len(w):
1527:             raise TypeError("expected x and w to have same length")
1528:         # apply weights. Don't use inplace operations as they
1529:         # can cause problems with NA.
1530:         lhs = lhs * w
1531:         rhs = rhs * w
1532: 
1533:     # set rcond
1534:     if rcond is None:
1535:         rcond = len(x)*np.finfo(x.dtype).eps
1536: 
1537:     # Determine the norms of the design matrix columns.
1538:     if issubclass(lhs.dtype.type, np.complexfloating):
1539:         scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
1540:     else:
1541:         scl = np.sqrt(np.square(lhs).sum(1))
1542:     scl[scl == 0] = 1
1543: 
1544:     # Solve the least squares problem.
1545:     c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
1546:     c = (c.T/scl).T
1547: 
1548:     # Expand c to include non-fitted coefficients which are set to zero
1549:     if deg.ndim > 0:
1550:         if c.ndim == 2:
1551:             cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
1552:         else:
1553:             cc = np.zeros(lmax+1, dtype=c.dtype)
1554:         cc[deg] = c
1555:         c = cc
1556: 
1557:     # warn on rank reduction
1558:     if rank != order and not full:
1559:         msg = "The fit may be poorly conditioned"
1560:         warnings.warn(msg, pu.RankWarning)
1561: 
1562:     if full:
1563:         return c, [resids, rank, s, rcond]
1564:     else:
1565:         return c
1566: 
1567: 
1568: def lagcompanion(c):
1569:     '''
1570:     Return the companion matrix of c.
1571: 
1572:     The usual companion matrix of the Laguerre polynomials is already
1573:     symmetric when `c` is a basis Laguerre polynomial, so no scaling is
1574:     applied.
1575: 
1576:     Parameters
1577:     ----------
1578:     c : array_like
1579:         1-D array of Laguerre series coefficients ordered from low to high
1580:         degree.
1581: 
1582:     Returns
1583:     -------
1584:     mat : ndarray
1585:         Companion matrix of dimensions (deg, deg).
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
1598:         return np.array([[1 + c[0]/c[1]]])
1599: 
1600:     n = len(c) - 1
1601:     mat = np.zeros((n, n), dtype=c.dtype)
1602:     top = mat.reshape(-1)[1::n+1]
1603:     mid = mat.reshape(-1)[0::n+1]
1604:     bot = mat.reshape(-1)[n::n+1]
1605:     top[...] = -np.arange(1, n)
1606:     mid[...] = 2.*np.arange(n) + 1.
1607:     bot[...] = top
1608:     mat[:, -1] += (c[:-1]/c[-1])*n
1609:     return mat
1610: 
1611: 
1612: def lagroots(c):
1613:     '''
1614:     Compute the roots of a Laguerre series.
1615: 
1616:     Return the roots (a.k.a. "zeros") of the polynomial
1617: 
1618:     .. math:: p(x) = \\sum_i c[i] * L_i(x).
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
1633:     polyroots, legroots, chebroots, hermroots, hermeroots
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
1645:     The Laguerre series basis polynomials aren't powers of `x` so the
1646:     results of this function may seem unintuitive.
1647: 
1648:     Examples
1649:     --------
1650:     >>> from numpy.polynomial.laguerre import lagroots, lagfromroots
1651:     >>> coef = lagfromroots([0, 1, 2])
1652:     >>> coef
1653:     array([  2.,  -8.,  12.,  -6.])
1654:     >>> lagroots(coef)
1655:     array([ -4.44089210e-16,   1.00000000e+00,   2.00000000e+00])
1656: 
1657:     '''
1658:     # c is a trimmed copy
1659:     [c] = pu.as_series([c])
1660:     if len(c) <= 1:
1661:         return np.array([], dtype=c.dtype)
1662:     if len(c) == 2:
1663:         return np.array([1 + c[0]/c[1]])
1664: 
1665:     m = lagcompanion(c)
1666:     r = la.eigvals(m)
1667:     r.sort()
1668:     return r
1669: 
1670: 
1671: def laggauss(deg):
1672:     '''
1673:     Gauss-Laguerre quadrature.
1674: 
1675:     Computes the sample points and weights for Gauss-Laguerre quadrature.
1676:     These sample points and weights will correctly integrate polynomials of
1677:     degree :math:`2*deg - 1` or less over the interval :math:`[0, \inf]`
1678:     with the weight function :math:`f(x) = \exp(-x)`.
1679: 
1680:     Parameters
1681:     ----------
1682:     deg : int
1683:         Number of sample points and weights. It must be >= 1.
1684: 
1685:     Returns
1686:     -------
1687:     x : ndarray
1688:         1-D ndarray containing the sample points.
1689:     y : ndarray
1690:         1-D ndarray containing the weights.
1691: 
1692:     Notes
1693:     -----
1694: 
1695:     .. versionadded::1.7.0
1696: 
1697:     The results have only been tested up to degree 100 higher degrees may
1698:     be problematic. The weights are determined by using the fact that
1699: 
1700:     .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))
1701: 
1702:     where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
1703:     is the k'th root of :math:`L_n`, and then scaling the results to get
1704:     the right value when integrating 1.
1705: 
1706:     '''
1707:     ideg = int(deg)
1708:     if ideg != deg or ideg < 1:
1709:         raise ValueError("deg must be a non-negative integer")
1710: 
1711:     # first approximation of roots. We use the fact that the companion
1712:     # matrix is symmetric in this case in order to obtain better zeros.
1713:     c = np.array([0]*deg + [1])
1714:     m = lagcompanion(c)
1715:     x = la.eigvalsh(m)
1716: 
1717:     # improve roots by one application of Newton
1718:     dy = lagval(x, c)
1719:     df = lagval(x, lagder(c))
1720:     x -= dy/df
1721: 
1722:     # compute the weights. We scale the factor to avoid possible numerical
1723:     # overflow.
1724:     fm = lagval(x, c[1:])
1725:     fm /= np.abs(fm).max()
1726:     df /= np.abs(df).max()
1727:     w = 1/(fm * df)
1728: 
1729:     # scale w to get the right value, 1 in this case
1730:     w /= w.sum()
1731: 
1732:     return x, w
1733: 
1734: 
1735: def lagweight(x):
1736:     '''Weight function of the Laguerre polynomials.
1737: 
1738:     The weight function is :math:`exp(-x)` and the interval of integration
1739:     is :math:`[0, \inf]`. The Laguerre polynomials are orthogonal, but not
1740:     normalized, with respect to this weight function.
1741: 
1742:     Parameters
1743:     ----------
1744:     x : array_like
1745:        Values at which the weight function will be computed.
1746: 
1747:     Returns
1748:     -------
1749:     w : ndarray
1750:        The weight function at `x`.
1751: 
1752:     Notes
1753:     -----
1754: 
1755:     .. versionadded::1.7.0
1756: 
1757:     '''
1758:     w = np.exp(-x)
1759:     return w
1760: 
1761: #
1762: # Laguerre series class
1763: #
1764: 
1765: class Laguerre(ABCPolyBase):
1766:     '''A Laguerre series class.
1767: 
1768:     The Laguerre class provides the standard Python numerical methods
1769:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
1770:     attributes and methods listed in the `ABCPolyBase` documentation.
1771: 
1772:     Parameters
1773:     ----------
1774:     coef : array_like
1775:         Laguerre coefficients in order of increasing degree, i.e,
1776:         ``(1, 2, 3)`` gives ``1*L_0(x) + 2*L_1(X) + 3*L_2(x)``.
1777:     domain : (2,) array_like, optional
1778:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
1779:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
1780:         The default value is [0, 1].
1781:     window : (2,) array_like, optional
1782:         Window, see `domain` for its use. The default value is [0, 1].
1783: 
1784:         .. versionadded:: 1.6.0
1785: 
1786:     '''
1787:     # Virtual Functions
1788:     _add = staticmethod(lagadd)
1789:     _sub = staticmethod(lagsub)
1790:     _mul = staticmethod(lagmul)
1791:     _div = staticmethod(lagdiv)
1792:     _pow = staticmethod(lagpow)
1793:     _val = staticmethod(lagval)
1794:     _int = staticmethod(lagint)
1795:     _der = staticmethod(lagder)
1796:     _fit = staticmethod(lagfit)
1797:     _line = staticmethod(lagline)
1798:     _roots = staticmethod(lagroots)
1799:     _fromroots = staticmethod(lagfromroots)
1800: 
1801:     # Virtual properties
1802:     nickname = 'lag'
1803:     domain = np.array(lagdomain)
1804:     window = np.array(lagdomain)
1805: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_170817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\nObjects for dealing with Laguerre series.\n\nThis module provides a number of objects (mostly functions) useful for\ndealing with Laguerre series, including a `Laguerre` class that\nencapsulates the usual arithmetic operations.  (General information\non how this module represents and works with such polynomials is in the\ndocstring for its "parent" sub-package, `numpy.polynomial`).\n\nConstants\n---------\n- `lagdomain` -- Laguerre series default domain, [-1,1].\n- `lagzero` -- Laguerre series that evaluates identically to 0.\n- `lagone` -- Laguerre series that evaluates identically to 1.\n- `lagx` -- Laguerre series for the identity map, ``f(x) = x``.\n\nArithmetic\n----------\n- `lagmulx` -- multiply a Laguerre series in ``P_i(x)`` by ``x``.\n- `lagadd` -- add two Laguerre series.\n- `lagsub` -- subtract one Laguerre series from another.\n- `lagmul` -- multiply two Laguerre series.\n- `lagdiv` -- divide one Laguerre series by another.\n- `lagval` -- evaluate a Laguerre series at given points.\n- `lagval2d` -- evaluate a 2D Laguerre series at given points.\n- `lagval3d` -- evaluate a 3D Laguerre series at given points.\n- `laggrid2d` -- evaluate a 2D Laguerre series on a Cartesian product.\n- `laggrid3d` -- evaluate a 3D Laguerre series on a Cartesian product.\n\nCalculus\n--------\n- `lagder` -- differentiate a Laguerre series.\n- `lagint` -- integrate a Laguerre series.\n\nMisc Functions\n--------------\n- `lagfromroots` -- create a Laguerre series with specified roots.\n- `lagroots` -- find the roots of a Laguerre series.\n- `lagvander` -- Vandermonde-like matrix for Laguerre polynomials.\n- `lagvander2d` -- Vandermonde-like matrix for 2D power series.\n- `lagvander3d` -- Vandermonde-like matrix for 3D power series.\n- `laggauss` -- Gauss-Laguerre quadrature, points and weights.\n- `lagweight` -- Laguerre weight function.\n- `lagcompanion` -- symmetrized companion matrix in Laguerre form.\n- `lagfit` -- least-squares fit returning a Laguerre series.\n- `lagtrim` -- trim leading coefficients from a Laguerre series.\n- `lagline` -- Laguerre series of given straight line.\n- `lag2poly` -- convert a Laguerre series to a polynomial.\n- `poly2lag` -- convert a polynomial to a Laguerre series.\n\nClasses\n-------\n- `Laguerre` -- A Laguerre series class.\n\nSee also\n--------\n`numpy.polynomial`\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 0))

# 'import warnings' statement (line 62)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 63, 0))

# 'import numpy' statement (line 63)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_170818 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy')

if (type(import_170818) is not StypyTypeError):

    if (import_170818 != 'pyd_module'):
        __import__(import_170818)
        sys_modules_170819 = sys.modules[import_170818]
        import_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'np', sys_modules_170819.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy', import_170818)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 0))

# 'import numpy.linalg' statement (line 64)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_170820 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'numpy.linalg')

if (type(import_170820) is not StypyTypeError):

    if (import_170820 != 'pyd_module'):
        __import__(import_170820)
        sys_modules_170821 = sys.modules[import_170820]
        import_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'la', sys_modules_170821.module_type_store, module_type_store)
    else:
        import numpy.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'la', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'numpy.linalg', import_170820)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'from numpy.polynomial import pu' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_170822 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial')

if (type(import_170822) is not StypyTypeError):

    if (import_170822 != 'pyd_module'):
        __import__(import_170822)
        sys_modules_170823 = sys.modules[import_170822]
        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', sys_modules_170823.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 66, 0), __file__, sys_modules_170823, sys_modules_170823.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', import_170822)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 67, 0))

# 'from numpy.polynomial._polybase import ABCPolyBase' statement (line 67)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_170824 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase')

if (type(import_170824) is not StypyTypeError):

    if (import_170824 != 'pyd_module'):
        __import__(import_170824)
        sys_modules_170825 = sys.modules[import_170824]
        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', sys_modules_170825.module_type_store, module_type_store, ['ABCPolyBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 67, 0), __file__, sys_modules_170825, sys_modules_170825.module_type_store, module_type_store)
    else:
        from numpy.polynomial._polybase import ABCPolyBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', None, module_type_store, ['ABCPolyBase'], [ABCPolyBase])

else:
    # Assigning a type to the variable 'numpy.polynomial._polybase' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', import_170824)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 69):

# Assigning a List to a Name (line 69):
__all__ = ['lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd', 'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder', 'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander', 'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d', 'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion', 'laggauss', 'lagweight']
module_type_store.set_exportable_members(['lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd', 'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder', 'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander', 'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d', 'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion', 'laggauss', 'lagweight'])

# Obtaining an instance of the builtin type 'list' (line 69)
list_170826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
str_170827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'lagzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170827)
# Adding element type (line 69)
str_170828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 15), 'str', 'lagone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170828)
# Adding element type (line 69)
str_170829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'str', 'lagx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170829)
# Adding element type (line 69)
str_170830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 33), 'str', 'lagdomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170830)
# Adding element type (line 69)
str_170831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 46), 'str', 'lagline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170831)
# Adding element type (line 69)
str_170832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 57), 'str', 'lagadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170832)
# Adding element type (line 69)
str_170833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'lagsub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170833)
# Adding element type (line 69)
str_170834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'str', 'lagmulx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170834)
# Adding element type (line 69)
str_170835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'str', 'lagmul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170835)
# Adding element type (line 69)
str_170836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'str', 'lagdiv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170836)
# Adding element type (line 69)
str_170837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'str', 'lagpow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170837)
# Adding element type (line 69)
str_170838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 55), 'str', 'lagval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170838)
# Adding element type (line 69)
str_170839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 65), 'str', 'lagder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170839)
# Adding element type (line 69)
str_170840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'lagint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170840)
# Adding element type (line 69)
str_170841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 14), 'str', 'lag2poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170841)
# Adding element type (line 69)
str_170842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'str', 'poly2lag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170842)
# Adding element type (line 69)
str_170843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'str', 'lagfromroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170843)
# Adding element type (line 69)
str_170844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 54), 'str', 'lagvander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170844)
# Adding element type (line 69)
str_170845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'lagfit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170845)
# Adding element type (line 69)
str_170846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 14), 'str', 'lagtrim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170846)
# Adding element type (line 69)
str_170847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'str', 'lagroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170847)
# Adding element type (line 69)
str_170848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 37), 'str', 'Laguerre')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170848)
# Adding element type (line 69)
str_170849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 49), 'str', 'lagval2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170849)
# Adding element type (line 69)
str_170850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 61), 'str', 'lagval3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170850)
# Adding element type (line 69)
str_170851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'laggrid2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170851)
# Adding element type (line 69)
str_170852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'str', 'laggrid3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170852)
# Adding element type (line 69)
str_170853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'str', 'lagvander2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170853)
# Adding element type (line 69)
str_170854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 45), 'str', 'lagvander3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170854)
# Adding element type (line 69)
str_170855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 60), 'str', 'lagcompanion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170855)
# Adding element type (line 69)
str_170856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'laggauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170856)
# Adding element type (line 69)
str_170857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'str', 'lagweight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_170826, str_170857)

# Assigning a type to the variable '__all__' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '__all__', list_170826)

# Assigning a Attribute to a Name (line 77):

# Assigning a Attribute to a Name (line 77):
# Getting the type of 'pu' (line 77)
pu_170858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 10), 'pu')
# Obtaining the member 'trimcoef' of a type (line 77)
trimcoef_170859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 10), pu_170858, 'trimcoef')
# Assigning a type to the variable 'lagtrim' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'lagtrim', trimcoef_170859)

@norecursion
def poly2lag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poly2lag'
    module_type_store = module_type_store.open_function_context('poly2lag', 80, 0, False)
    
    # Passed parameters checking function
    poly2lag.stypy_localization = localization
    poly2lag.stypy_type_of_self = None
    poly2lag.stypy_type_store = module_type_store
    poly2lag.stypy_function_name = 'poly2lag'
    poly2lag.stypy_param_names_list = ['pol']
    poly2lag.stypy_varargs_param_name = None
    poly2lag.stypy_kwargs_param_name = None
    poly2lag.stypy_call_defaults = defaults
    poly2lag.stypy_call_varargs = varargs
    poly2lag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poly2lag', ['pol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poly2lag', localization, ['pol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poly2lag(...)' code ##################

    str_170860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    poly2lag(pol)\n\n    Convert a polynomial to a Laguerre series.\n\n    Convert an array representing the coefficients of a polynomial (relative\n    to the "standard" basis) ordered from lowest degree to highest, to an\n    array of the coefficients of the equivalent Laguerre series, ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    pol : array_like\n        1-D array containing the polynomial coefficients\n\n    Returns\n    -------\n    c : ndarray\n        1-D array containing the coefficients of the equivalent Laguerre\n        series.\n\n    See Also\n    --------\n    lag2poly\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import poly2lag\n    >>> poly2lag(np.arange(4))\n    array([ 23., -63.,  58., -18.])\n\n    ')
    
    # Assigning a Call to a List (line 118):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_170863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    # Adding element type (line 118)
    # Getting the type of 'pol' (line 118)
    pol_170864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'pol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 25), list_170863, pol_170864)
    
    # Processing the call keyword arguments (line 118)
    kwargs_170865 = {}
    # Getting the type of 'pu' (line 118)
    pu_170861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 118)
    as_series_170862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), pu_170861, 'as_series')
    # Calling as_series(args, kwargs) (line 118)
    as_series_call_result_170866 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), as_series_170862, *[list_170863], **kwargs_170865)
    
    # Assigning a type to the variable 'call_assignment_170762' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_170762', as_series_call_result_170866)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170870 = {}
    # Getting the type of 'call_assignment_170762' (line 118)
    call_assignment_170762_170867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_170762', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___170868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_170762_170867, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170871 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170868, *[int_170869], **kwargs_170870)
    
    # Assigning a type to the variable 'call_assignment_170763' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_170763', getitem___call_result_170871)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_170763' (line 118)
    call_assignment_170763_170872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_170763')
    # Assigning a type to the variable 'pol' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 5), 'pol', call_assignment_170763_170872)
    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    
    # Call to len(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'pol' (line 119)
    pol_170874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'pol', False)
    # Processing the call keyword arguments (line 119)
    kwargs_170875 = {}
    # Getting the type of 'len' (line 119)
    len_170873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 10), 'len', False)
    # Calling len(args, kwargs) (line 119)
    len_call_result_170876 = invoke(stypy.reporting.localization.Localization(__file__, 119, 10), len_170873, *[pol_170874], **kwargs_170875)
    
    int_170877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'int')
    # Applying the binary operator '-' (line 119)
    result_sub_170878 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 10), '-', len_call_result_170876, int_170877)
    
    # Assigning a type to the variable 'deg' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'deg', result_sub_170878)
    
    # Assigning a Num to a Name (line 120):
    
    # Assigning a Num to a Name (line 120):
    int_170879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 10), 'int')
    # Assigning a type to the variable 'res' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'res', int_170879)
    
    
    # Call to range(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'deg' (line 121)
    deg_170881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'deg', False)
    int_170882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'int')
    int_170883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'int')
    # Processing the call keyword arguments (line 121)
    kwargs_170884 = {}
    # Getting the type of 'range' (line 121)
    range_170880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'range', False)
    # Calling range(args, kwargs) (line 121)
    range_call_result_170885 = invoke(stypy.reporting.localization.Localization(__file__, 121, 13), range_170880, *[deg_170881, int_170882, int_170883], **kwargs_170884)
    
    # Testing the type of a for loop iterable (line 121)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 4), range_call_result_170885)
    # Getting the type of the for loop variable (line 121)
    for_loop_var_170886 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 4), range_call_result_170885)
    # Assigning a type to the variable 'i' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'i', for_loop_var_170886)
    # SSA begins for a for statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to lagadd(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to lagmulx(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'res' (line 122)
    res_170889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'res', False)
    # Processing the call keyword arguments (line 122)
    kwargs_170890 = {}
    # Getting the type of 'lagmulx' (line 122)
    lagmulx_170888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'lagmulx', False)
    # Calling lagmulx(args, kwargs) (line 122)
    lagmulx_call_result_170891 = invoke(stypy.reporting.localization.Localization(__file__, 122, 21), lagmulx_170888, *[res_170889], **kwargs_170890)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 122)
    i_170892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'i', False)
    # Getting the type of 'pol' (line 122)
    pol_170893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 'pol', False)
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___170894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 35), pol_170893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_170895 = invoke(stypy.reporting.localization.Localization(__file__, 122, 35), getitem___170894, i_170892)
    
    # Processing the call keyword arguments (line 122)
    kwargs_170896 = {}
    # Getting the type of 'lagadd' (line 122)
    lagadd_170887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'lagadd', False)
    # Calling lagadd(args, kwargs) (line 122)
    lagadd_call_result_170897 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), lagadd_170887, *[lagmulx_call_result_170891, subscript_call_result_170895], **kwargs_170896)
    
    # Assigning a type to the variable 'res' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'res', lagadd_call_result_170897)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 123)
    res_170898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', res_170898)
    
    # ################# End of 'poly2lag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poly2lag' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_170899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poly2lag'
    return stypy_return_type_170899

# Assigning a type to the variable 'poly2lag' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'poly2lag', poly2lag)

@norecursion
def lag2poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lag2poly'
    module_type_store = module_type_store.open_function_context('lag2poly', 126, 0, False)
    
    # Passed parameters checking function
    lag2poly.stypy_localization = localization
    lag2poly.stypy_type_of_self = None
    lag2poly.stypy_type_store = module_type_store
    lag2poly.stypy_function_name = 'lag2poly'
    lag2poly.stypy_param_names_list = ['c']
    lag2poly.stypy_varargs_param_name = None
    lag2poly.stypy_kwargs_param_name = None
    lag2poly.stypy_call_defaults = defaults
    lag2poly.stypy_call_varargs = varargs
    lag2poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lag2poly', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lag2poly', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lag2poly(...)' code ##################

    str_170900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Convert a Laguerre series to a polynomial.\n\n    Convert an array representing the coefficients of a Laguerre series,\n    ordered from lowest degree to highest, to an array of the coefficients\n    of the equivalent polynomial (relative to the "standard" basis) ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array containing the Laguerre series coefficients, ordered\n        from lowest order term to highest.\n\n    Returns\n    -------\n    pol : ndarray\n        1-D array containing the coefficients of the equivalent polynomial\n        (relative to the "standard" basis) ordered from lowest order term\n        to highest.\n\n    See Also\n    --------\n    poly2lag\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lag2poly\n    >>> lag2poly([ 23., -63.,  58., -18.])\n    array([ 0.,  1.,  2.,  3.])\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 164, 4))
    
    # 'from numpy.polynomial.polynomial import polyadd, polysub, polymulx' statement (line 164)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_170901 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial')

    if (type(import_170901) is not StypyTypeError):

        if (import_170901 != 'pyd_module'):
            __import__(import_170901)
            sys_modules_170902 = sys.modules[import_170901]
            import_from_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial', sys_modules_170902.module_type_store, module_type_store, ['polyadd', 'polysub', 'polymulx'])
            nest_module(stypy.reporting.localization.Localization(__file__, 164, 4), __file__, sys_modules_170902, sys_modules_170902.module_type_store, module_type_store)
        else:
            from numpy.polynomial.polynomial import polyadd, polysub, polymulx

            import_from_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial', None, module_type_store, ['polyadd', 'polysub', 'polymulx'], [polyadd, polysub, polymulx])

    else:
        # Assigning a type to the variable 'numpy.polynomial.polynomial' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial', import_170901)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Assigning a Call to a List (line 166):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 166)
    # Processing the call arguments (line 166)
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_170905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    # Adding element type (line 166)
    # Getting the type of 'c' (line 166)
    c_170906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 23), list_170905, c_170906)
    
    # Processing the call keyword arguments (line 166)
    kwargs_170907 = {}
    # Getting the type of 'pu' (line 166)
    pu_170903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 166)
    as_series_170904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 10), pu_170903, 'as_series')
    # Calling as_series(args, kwargs) (line 166)
    as_series_call_result_170908 = invoke(stypy.reporting.localization.Localization(__file__, 166, 10), as_series_170904, *[list_170905], **kwargs_170907)
    
    # Assigning a type to the variable 'call_assignment_170764' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_170764', as_series_call_result_170908)
    
    # Assigning a Call to a Name (line 166):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170912 = {}
    # Getting the type of 'call_assignment_170764' (line 166)
    call_assignment_170764_170909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_170764', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___170910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 4), call_assignment_170764_170909, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170913 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170910, *[int_170911], **kwargs_170912)
    
    # Assigning a type to the variable 'call_assignment_170765' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_170765', getitem___call_result_170913)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'call_assignment_170765' (line 166)
    call_assignment_170765_170914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_170765')
    # Assigning a type to the variable 'c' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 5), 'c', call_assignment_170765_170914)
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'c' (line 167)
    c_170916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'c', False)
    # Processing the call keyword arguments (line 167)
    kwargs_170917 = {}
    # Getting the type of 'len' (line 167)
    len_170915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_170918 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), len_170915, *[c_170916], **kwargs_170917)
    
    # Assigning a type to the variable 'n' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'n', len_call_result_170918)
    
    
    # Getting the type of 'n' (line 168)
    n_170919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 7), 'n')
    int_170920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'int')
    # Applying the binary operator '==' (line 168)
    result_eq_170921 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 7), '==', n_170919, int_170920)
    
    # Testing the type of an if condition (line 168)
    if_condition_170922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), result_eq_170921)
    # Assigning a type to the variable 'if_condition_170922' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_170922', if_condition_170922)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 169)
    c_170923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', c_170923)
    # SSA branch for the else part of an if statement (line 168)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 171):
    
    # Assigning a Subscript to a Name (line 171):
    
    # Obtaining the type of the subscript
    int_170924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 15), 'int')
    # Getting the type of 'c' (line 171)
    c_170925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___170926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 13), c_170925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_170927 = invoke(stypy.reporting.localization.Localization(__file__, 171, 13), getitem___170926, int_170924)
    
    # Assigning a type to the variable 'c0' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'c0', subscript_call_result_170927)
    
    # Assigning a Subscript to a Name (line 172):
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_170928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'int')
    # Getting the type of 'c' (line 172)
    c_170929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___170930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 13), c_170929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_170931 = invoke(stypy.reporting.localization.Localization(__file__, 172, 13), getitem___170930, int_170928)
    
    # Assigning a type to the variable 'c1' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'c1', subscript_call_result_170931)
    
    
    # Call to range(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'n' (line 174)
    n_170933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'n', False)
    int_170934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 27), 'int')
    # Applying the binary operator '-' (line 174)
    result_sub_170935 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 23), '-', n_170933, int_170934)
    
    int_170936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 30), 'int')
    int_170937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 33), 'int')
    # Processing the call keyword arguments (line 174)
    kwargs_170938 = {}
    # Getting the type of 'range' (line 174)
    range_170932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'range', False)
    # Calling range(args, kwargs) (line 174)
    range_call_result_170939 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), range_170932, *[result_sub_170935, int_170936, int_170937], **kwargs_170938)
    
    # Testing the type of a for loop iterable (line 174)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 174, 8), range_call_result_170939)
    # Getting the type of the for loop variable (line 174)
    for_loop_var_170940 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 174, 8), range_call_result_170939)
    # Assigning a type to the variable 'i' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'i', for_loop_var_170940)
    # SSA begins for a for statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 175):
    
    # Assigning a Name to a Name (line 175):
    # Getting the type of 'c0' (line 175)
    c0_170941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'tmp', c0_170941)
    
    # Assigning a Call to a Name (line 176):
    
    # Assigning a Call to a Name (line 176):
    
    # Call to polysub(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 176)
    i_170943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'i', False)
    int_170944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'int')
    # Applying the binary operator '-' (line 176)
    result_sub_170945 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 27), '-', i_170943, int_170944)
    
    # Getting the type of 'c' (line 176)
    c_170946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___170947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), c_170946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_170948 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), getitem___170947, result_sub_170945)
    
    # Getting the type of 'c1' (line 176)
    c1_170949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'c1', False)
    # Getting the type of 'i' (line 176)
    i_170950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 40), 'i', False)
    int_170951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 44), 'int')
    # Applying the binary operator '-' (line 176)
    result_sub_170952 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 40), '-', i_170950, int_170951)
    
    # Applying the binary operator '*' (line 176)
    result_mul_170953 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 36), '*', c1_170949, result_sub_170952)
    
    # Getting the type of 'i' (line 176)
    i_170954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 48), 'i', False)
    # Applying the binary operator 'div' (line 176)
    result_div_170955 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 35), 'div', result_mul_170953, i_170954)
    
    # Processing the call keyword arguments (line 176)
    kwargs_170956 = {}
    # Getting the type of 'polysub' (line 176)
    polysub_170942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'polysub', False)
    # Calling polysub(args, kwargs) (line 176)
    polysub_call_result_170957 = invoke(stypy.reporting.localization.Localization(__file__, 176, 17), polysub_170942, *[subscript_call_result_170948, result_div_170955], **kwargs_170956)
    
    # Assigning a type to the variable 'c0' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'c0', polysub_call_result_170957)
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to polyadd(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'tmp' (line 177)
    tmp_170959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'tmp', False)
    
    # Call to polysub(...): (line 177)
    # Processing the call arguments (line 177)
    int_170961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 39), 'int')
    # Getting the type of 'i' (line 177)
    i_170962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 41), 'i', False)
    # Applying the binary operator '*' (line 177)
    result_mul_170963 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 39), '*', int_170961, i_170962)
    
    int_170964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 45), 'int')
    # Applying the binary operator '-' (line 177)
    result_sub_170965 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 39), '-', result_mul_170963, int_170964)
    
    # Getting the type of 'c1' (line 177)
    c1_170966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 48), 'c1', False)
    # Applying the binary operator '*' (line 177)
    result_mul_170967 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 38), '*', result_sub_170965, c1_170966)
    
    
    # Call to polymulx(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'c1' (line 177)
    c1_170969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 61), 'c1', False)
    # Processing the call keyword arguments (line 177)
    kwargs_170970 = {}
    # Getting the type of 'polymulx' (line 177)
    polymulx_170968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 177)
    polymulx_call_result_170971 = invoke(stypy.reporting.localization.Localization(__file__, 177, 52), polymulx_170968, *[c1_170969], **kwargs_170970)
    
    # Processing the call keyword arguments (line 177)
    kwargs_170972 = {}
    # Getting the type of 'polysub' (line 177)
    polysub_170960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), 'polysub', False)
    # Calling polysub(args, kwargs) (line 177)
    polysub_call_result_170973 = invoke(stypy.reporting.localization.Localization(__file__, 177, 30), polysub_170960, *[result_mul_170967, polymulx_call_result_170971], **kwargs_170972)
    
    # Getting the type of 'i' (line 177)
    i_170974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 66), 'i', False)
    # Applying the binary operator 'div' (line 177)
    result_div_170975 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 30), 'div', polysub_call_result_170973, i_170974)
    
    # Processing the call keyword arguments (line 177)
    kwargs_170976 = {}
    # Getting the type of 'polyadd' (line 177)
    polyadd_170958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 177)
    polyadd_call_result_170977 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), polyadd_170958, *[tmp_170959, result_div_170975], **kwargs_170976)
    
    # Assigning a type to the variable 'c1' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'c1', polyadd_call_result_170977)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to polyadd(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'c0' (line 178)
    c0_170979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'c0', False)
    
    # Call to polysub(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'c1' (line 178)
    c1_170981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'c1', False)
    
    # Call to polymulx(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'c1' (line 178)
    c1_170983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 48), 'c1', False)
    # Processing the call keyword arguments (line 178)
    kwargs_170984 = {}
    # Getting the type of 'polymulx' (line 178)
    polymulx_170982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 39), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 178)
    polymulx_call_result_170985 = invoke(stypy.reporting.localization.Localization(__file__, 178, 39), polymulx_170982, *[c1_170983], **kwargs_170984)
    
    # Processing the call keyword arguments (line 178)
    kwargs_170986 = {}
    # Getting the type of 'polysub' (line 178)
    polysub_170980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'polysub', False)
    # Calling polysub(args, kwargs) (line 178)
    polysub_call_result_170987 = invoke(stypy.reporting.localization.Localization(__file__, 178, 27), polysub_170980, *[c1_170981, polymulx_call_result_170985], **kwargs_170986)
    
    # Processing the call keyword arguments (line 178)
    kwargs_170988 = {}
    # Getting the type of 'polyadd' (line 178)
    polyadd_170978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 178)
    polyadd_call_result_170989 = invoke(stypy.reporting.localization.Localization(__file__, 178, 15), polyadd_170978, *[c0_170979, polysub_call_result_170987], **kwargs_170988)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', polyadd_call_result_170989)
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lag2poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lag2poly' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_170990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lag2poly'
    return stypy_return_type_170990

# Assigning a type to the variable 'lag2poly' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'lag2poly', lag2poly)

# Assigning a Call to a Name (line 186):

# Assigning a Call to a Name (line 186):

# Call to array(...): (line 186)
# Processing the call arguments (line 186)

# Obtaining an instance of the builtin type 'list' (line 186)
list_170993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 186)
# Adding element type (line 186)
int_170994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), list_170993, int_170994)
# Adding element type (line 186)
int_170995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), list_170993, int_170995)

# Processing the call keyword arguments (line 186)
kwargs_170996 = {}
# Getting the type of 'np' (line 186)
np_170991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'np', False)
# Obtaining the member 'array' of a type (line 186)
array_170992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), np_170991, 'array')
# Calling array(args, kwargs) (line 186)
array_call_result_170997 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), array_170992, *[list_170993], **kwargs_170996)

# Assigning a type to the variable 'lagdomain' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'lagdomain', array_call_result_170997)

# Assigning a Call to a Name (line 189):

# Assigning a Call to a Name (line 189):

# Call to array(...): (line 189)
# Processing the call arguments (line 189)

# Obtaining an instance of the builtin type 'list' (line 189)
list_171000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 189)
# Adding element type (line 189)
int_171001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 19), list_171000, int_171001)

# Processing the call keyword arguments (line 189)
kwargs_171002 = {}
# Getting the type of 'np' (line 189)
np_170998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 10), 'np', False)
# Obtaining the member 'array' of a type (line 189)
array_170999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 10), np_170998, 'array')
# Calling array(args, kwargs) (line 189)
array_call_result_171003 = invoke(stypy.reporting.localization.Localization(__file__, 189, 10), array_170999, *[list_171000], **kwargs_171002)

# Assigning a type to the variable 'lagzero' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'lagzero', array_call_result_171003)

# Assigning a Call to a Name (line 192):

# Assigning a Call to a Name (line 192):

# Call to array(...): (line 192)
# Processing the call arguments (line 192)

# Obtaining an instance of the builtin type 'list' (line 192)
list_171006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 192)
# Adding element type (line 192)
int_171007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 18), list_171006, int_171007)

# Processing the call keyword arguments (line 192)
kwargs_171008 = {}
# Getting the type of 'np' (line 192)
np_171004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 9), 'np', False)
# Obtaining the member 'array' of a type (line 192)
array_171005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 9), np_171004, 'array')
# Calling array(args, kwargs) (line 192)
array_call_result_171009 = invoke(stypy.reporting.localization.Localization(__file__, 192, 9), array_171005, *[list_171006], **kwargs_171008)

# Assigning a type to the variable 'lagone' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'lagone', array_call_result_171009)

# Assigning a Call to a Name (line 195):

# Assigning a Call to a Name (line 195):

# Call to array(...): (line 195)
# Processing the call arguments (line 195)

# Obtaining an instance of the builtin type 'list' (line 195)
list_171012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 195)
# Adding element type (line 195)
int_171013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_171012, int_171013)
# Adding element type (line 195)
int_171014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_171012, int_171014)

# Processing the call keyword arguments (line 195)
kwargs_171015 = {}
# Getting the type of 'np' (line 195)
np_171010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 7), 'np', False)
# Obtaining the member 'array' of a type (line 195)
array_171011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 7), np_171010, 'array')
# Calling array(args, kwargs) (line 195)
array_call_result_171016 = invoke(stypy.reporting.localization.Localization(__file__, 195, 7), array_171011, *[list_171012], **kwargs_171015)

# Assigning a type to the variable 'lagx' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'lagx', array_call_result_171016)

@norecursion
def lagline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagline'
    module_type_store = module_type_store.open_function_context('lagline', 198, 0, False)
    
    # Passed parameters checking function
    lagline.stypy_localization = localization
    lagline.stypy_type_of_self = None
    lagline.stypy_type_store = module_type_store
    lagline.stypy_function_name = 'lagline'
    lagline.stypy_param_names_list = ['off', 'scl']
    lagline.stypy_varargs_param_name = None
    lagline.stypy_kwargs_param_name = None
    lagline.stypy_call_defaults = defaults
    lagline.stypy_call_varargs = varargs
    lagline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagline', ['off', 'scl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagline', localization, ['off', 'scl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagline(...)' code ##################

    str_171017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'str', "\n    Laguerre series whose graph is a straight line.\n\n\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The specified line is given by ``off + scl*x``.\n\n    Returns\n    -------\n    y : ndarray\n        This module's representation of the Laguerre series for\n        ``off + scl*x``.\n\n    See Also\n    --------\n    polyline, chebline\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagline, lagval\n    >>> lagval(0,lagline(3, 2))\n    3.0\n    >>> lagval(1,lagline(3, 2))\n    5.0\n\n    ")
    
    
    # Getting the type of 'scl' (line 228)
    scl_171018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 7), 'scl')
    int_171019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 14), 'int')
    # Applying the binary operator '!=' (line 228)
    result_ne_171020 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 7), '!=', scl_171018, int_171019)
    
    # Testing the type of an if condition (line 228)
    if_condition_171021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 4), result_ne_171020)
    # Assigning a type to the variable 'if_condition_171021' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'if_condition_171021', if_condition_171021)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 229)
    # Processing the call arguments (line 229)
    
    # Obtaining an instance of the builtin type 'list' (line 229)
    list_171024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 229)
    # Adding element type (line 229)
    # Getting the type of 'off' (line 229)
    off_171025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'off', False)
    # Getting the type of 'scl' (line 229)
    scl_171026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 31), 'scl', False)
    # Applying the binary operator '+' (line 229)
    result_add_171027 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 25), '+', off_171025, scl_171026)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 24), list_171024, result_add_171027)
    # Adding element type (line 229)
    
    # Getting the type of 'scl' (line 229)
    scl_171028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 37), 'scl', False)
    # Applying the 'usub' unary operator (line 229)
    result___neg___171029 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 36), 'usub', scl_171028)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 24), list_171024, result___neg___171029)
    
    # Processing the call keyword arguments (line 229)
    kwargs_171030 = {}
    # Getting the type of 'np' (line 229)
    np_171022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 229)
    array_171023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), np_171022, 'array')
    # Calling array(args, kwargs) (line 229)
    array_call_result_171031 = invoke(stypy.reporting.localization.Localization(__file__, 229, 15), array_171023, *[list_171024], **kwargs_171030)
    
    # Assigning a type to the variable 'stypy_return_type' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', array_call_result_171031)
    # SSA branch for the else part of an if statement (line 228)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Obtaining an instance of the builtin type 'list' (line 231)
    list_171034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 231)
    # Adding element type (line 231)
    # Getting the type of 'off' (line 231)
    off_171035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 24), list_171034, off_171035)
    
    # Processing the call keyword arguments (line 231)
    kwargs_171036 = {}
    # Getting the type of 'np' (line 231)
    np_171032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 231)
    array_171033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 15), np_171032, 'array')
    # Calling array(args, kwargs) (line 231)
    array_call_result_171037 = invoke(stypy.reporting.localization.Localization(__file__, 231, 15), array_171033, *[list_171034], **kwargs_171036)
    
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', array_call_result_171037)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lagline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagline' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_171038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171038)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagline'
    return stypy_return_type_171038

# Assigning a type to the variable 'lagline' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'lagline', lagline)

@norecursion
def lagfromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagfromroots'
    module_type_store = module_type_store.open_function_context('lagfromroots', 234, 0, False)
    
    # Passed parameters checking function
    lagfromroots.stypy_localization = localization
    lagfromroots.stypy_type_of_self = None
    lagfromroots.stypy_type_store = module_type_store
    lagfromroots.stypy_function_name = 'lagfromroots'
    lagfromroots.stypy_param_names_list = ['roots']
    lagfromroots.stypy_varargs_param_name = None
    lagfromroots.stypy_kwargs_param_name = None
    lagfromroots.stypy_call_defaults = defaults
    lagfromroots.stypy_call_varargs = varargs
    lagfromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagfromroots', ['roots'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagfromroots', localization, ['roots'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagfromroots(...)' code ##################

    str_171039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, (-1)), 'str', '\n    Generate a Laguerre series with given roots.\n\n    The function returns the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    in Laguerre form, where the `r_n` are the roots specified in `roots`.\n    If a zero has multiplicity n, then it must appear in `roots` n times.\n    For instance, if 2 is a root of multiplicity three and 3 is a root of\n    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The\n    roots can appear in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)\n\n    The coefficient of the last term is not generally 1 for monic\n    polynomials in Laguerre form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of coefficients.  If all roots are real then `out` is a\n        real array, if some of the roots are complex, then `out` is complex\n        even if all the coefficients in the result are real (see Examples\n        below).\n\n    See Also\n    --------\n    polyfromroots, legfromroots, chebfromroots, hermfromroots,\n    hermefromroots.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagfromroots, lagval\n    >>> coef = lagfromroots((-1, 0, 1))\n    >>> lagval((-1, 0, 1), coef)\n    array([ 0.,  0.,  0.])\n    >>> coef = lagfromroots((-1j, 1j))\n    >>> lagval((-1j, 1j), coef)\n    array([ 0.+0.j,  0.+0.j])\n\n    ')
    
    
    
    # Call to len(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'roots' (line 284)
    roots_171041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'roots', False)
    # Processing the call keyword arguments (line 284)
    kwargs_171042 = {}
    # Getting the type of 'len' (line 284)
    len_171040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 7), 'len', False)
    # Calling len(args, kwargs) (line 284)
    len_call_result_171043 = invoke(stypy.reporting.localization.Localization(__file__, 284, 7), len_171040, *[roots_171041], **kwargs_171042)
    
    int_171044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 21), 'int')
    # Applying the binary operator '==' (line 284)
    result_eq_171045 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 7), '==', len_call_result_171043, int_171044)
    
    # Testing the type of an if condition (line 284)
    if_condition_171046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 4), result_eq_171045)
    # Assigning a type to the variable 'if_condition_171046' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'if_condition_171046', if_condition_171046)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 285)
    # Processing the call arguments (line 285)
    int_171049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 23), 'int')
    # Processing the call keyword arguments (line 285)
    kwargs_171050 = {}
    # Getting the type of 'np' (line 285)
    np_171047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 285)
    ones_171048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), np_171047, 'ones')
    # Calling ones(args, kwargs) (line 285)
    ones_call_result_171051 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), ones_171048, *[int_171049], **kwargs_171050)
    
    # Assigning a type to the variable 'stypy_return_type' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', ones_call_result_171051)
    # SSA branch for the else part of an if statement (line 284)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a List (line 287):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 287)
    # Processing the call arguments (line 287)
    
    # Obtaining an instance of the builtin type 'list' (line 287)
    list_171054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 287)
    # Adding element type (line 287)
    # Getting the type of 'roots' (line 287)
    roots_171055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), list_171054, roots_171055)
    
    # Processing the call keyword arguments (line 287)
    # Getting the type of 'False' (line 287)
    False_171056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 45), 'False', False)
    keyword_171057 = False_171056
    kwargs_171058 = {'trim': keyword_171057}
    # Getting the type of 'pu' (line 287)
    pu_171052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 18), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 287)
    as_series_171053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 18), pu_171052, 'as_series')
    # Calling as_series(args, kwargs) (line 287)
    as_series_call_result_171059 = invoke(stypy.reporting.localization.Localization(__file__, 287, 18), as_series_171053, *[list_171054], **kwargs_171058)
    
    # Assigning a type to the variable 'call_assignment_170766' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'call_assignment_170766', as_series_call_result_171059)
    
    # Assigning a Call to a Name (line 287):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
    # Processing the call keyword arguments
    kwargs_171063 = {}
    # Getting the type of 'call_assignment_170766' (line 287)
    call_assignment_170766_171060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'call_assignment_170766', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___171061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), call_assignment_170766_171060, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171064 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171061, *[int_171062], **kwargs_171063)
    
    # Assigning a type to the variable 'call_assignment_170767' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'call_assignment_170767', getitem___call_result_171064)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'call_assignment_170767' (line 287)
    call_assignment_170767_171065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'call_assignment_170767')
    # Assigning a type to the variable 'roots' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 9), 'roots', call_assignment_170767_171065)
    
    # Call to sort(...): (line 288)
    # Processing the call keyword arguments (line 288)
    kwargs_171068 = {}
    # Getting the type of 'roots' (line 288)
    roots_171066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'roots', False)
    # Obtaining the member 'sort' of a type (line 288)
    sort_171067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), roots_171066, 'sort')
    # Calling sort(args, kwargs) (line 288)
    sort_call_result_171069 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), sort_171067, *[], **kwargs_171068)
    
    
    # Assigning a ListComp to a Name (line 289):
    
    # Assigning a ListComp to a Name (line 289):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'roots' (line 289)
    roots_171076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 37), 'roots')
    comprehension_171077 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 13), roots_171076)
    # Assigning a type to the variable 'r' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'r', comprehension_171077)
    
    # Call to lagline(...): (line 289)
    # Processing the call arguments (line 289)
    
    # Getting the type of 'r' (line 289)
    r_171071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'r', False)
    # Applying the 'usub' unary operator (line 289)
    result___neg___171072 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 21), 'usub', r_171071)
    
    int_171073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 25), 'int')
    # Processing the call keyword arguments (line 289)
    kwargs_171074 = {}
    # Getting the type of 'lagline' (line 289)
    lagline_171070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'lagline', False)
    # Calling lagline(args, kwargs) (line 289)
    lagline_call_result_171075 = invoke(stypy.reporting.localization.Localization(__file__, 289, 13), lagline_171070, *[result___neg___171072, int_171073], **kwargs_171074)
    
    list_171078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 13), list_171078, lagline_call_result_171075)
    # Assigning a type to the variable 'p' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'p', list_171078)
    
    # Assigning a Call to a Name (line 290):
    
    # Assigning a Call to a Name (line 290):
    
    # Call to len(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'p' (line 290)
    p_171080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'p', False)
    # Processing the call keyword arguments (line 290)
    kwargs_171081 = {}
    # Getting the type of 'len' (line 290)
    len_171079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'len', False)
    # Calling len(args, kwargs) (line 290)
    len_call_result_171082 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), len_171079, *[p_171080], **kwargs_171081)
    
    # Assigning a type to the variable 'n' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'n', len_call_result_171082)
    
    
    # Getting the type of 'n' (line 291)
    n_171083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'n')
    int_171084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'int')
    # Applying the binary operator '>' (line 291)
    result_gt_171085 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 14), '>', n_171083, int_171084)
    
    # Testing the type of an if condition (line 291)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 8), result_gt_171085)
    # SSA begins for while statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 292):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'n' (line 292)
    n_171087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'n', False)
    int_171088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 29), 'int')
    # Processing the call keyword arguments (line 292)
    kwargs_171089 = {}
    # Getting the type of 'divmod' (line 292)
    divmod_171086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 292)
    divmod_call_result_171090 = invoke(stypy.reporting.localization.Localization(__file__, 292, 19), divmod_171086, *[n_171087, int_171088], **kwargs_171089)
    
    # Assigning a type to the variable 'call_assignment_170768' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170768', divmod_call_result_171090)
    
    # Assigning a Call to a Name (line 292):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'int')
    # Processing the call keyword arguments
    kwargs_171094 = {}
    # Getting the type of 'call_assignment_170768' (line 292)
    call_assignment_170768_171091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170768', False)
    # Obtaining the member '__getitem__' of a type (line 292)
    getitem___171092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), call_assignment_170768_171091, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171095 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171092, *[int_171093], **kwargs_171094)
    
    # Assigning a type to the variable 'call_assignment_170769' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170769', getitem___call_result_171095)
    
    # Assigning a Name to a Name (line 292):
    # Getting the type of 'call_assignment_170769' (line 292)
    call_assignment_170769_171096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170769')
    # Assigning a type to the variable 'm' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'm', call_assignment_170769_171096)
    
    # Assigning a Call to a Name (line 292):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'int')
    # Processing the call keyword arguments
    kwargs_171100 = {}
    # Getting the type of 'call_assignment_170768' (line 292)
    call_assignment_170768_171097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170768', False)
    # Obtaining the member '__getitem__' of a type (line 292)
    getitem___171098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), call_assignment_170768_171097, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171101 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171098, *[int_171099], **kwargs_171100)
    
    # Assigning a type to the variable 'call_assignment_170770' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170770', getitem___call_result_171101)
    
    # Assigning a Name to a Name (line 292):
    # Getting the type of 'call_assignment_170770' (line 292)
    call_assignment_170770_171102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'call_assignment_170770')
    # Assigning a type to the variable 'r' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'r', call_assignment_170770_171102)
    
    # Assigning a ListComp to a Name (line 293):
    
    # Assigning a ListComp to a Name (line 293):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'm' (line 293)
    m_171117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 55), 'm', False)
    # Processing the call keyword arguments (line 293)
    kwargs_171118 = {}
    # Getting the type of 'range' (line 293)
    range_171116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 49), 'range', False)
    # Calling range(args, kwargs) (line 293)
    range_call_result_171119 = invoke(stypy.reporting.localization.Localization(__file__, 293, 49), range_171116, *[m_171117], **kwargs_171118)
    
    comprehension_171120 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 19), range_call_result_171119)
    # Assigning a type to the variable 'i' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'i', comprehension_171120)
    
    # Call to lagmul(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 293)
    i_171104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 28), 'i', False)
    # Getting the type of 'p' (line 293)
    p_171105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 293)
    getitem___171106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 26), p_171105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 293)
    subscript_call_result_171107 = invoke(stypy.reporting.localization.Localization(__file__, 293, 26), getitem___171106, i_171104)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 293)
    i_171108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 34), 'i', False)
    # Getting the type of 'm' (line 293)
    m_171109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 36), 'm', False)
    # Applying the binary operator '+' (line 293)
    result_add_171110 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 34), '+', i_171108, m_171109)
    
    # Getting the type of 'p' (line 293)
    p_171111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 32), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 293)
    getitem___171112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 32), p_171111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 293)
    subscript_call_result_171113 = invoke(stypy.reporting.localization.Localization(__file__, 293, 32), getitem___171112, result_add_171110)
    
    # Processing the call keyword arguments (line 293)
    kwargs_171114 = {}
    # Getting the type of 'lagmul' (line 293)
    lagmul_171103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'lagmul', False)
    # Calling lagmul(args, kwargs) (line 293)
    lagmul_call_result_171115 = invoke(stypy.reporting.localization.Localization(__file__, 293, 19), lagmul_171103, *[subscript_call_result_171107, subscript_call_result_171113], **kwargs_171114)
    
    list_171121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 19), list_171121, lagmul_call_result_171115)
    # Assigning a type to the variable 'tmp' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'tmp', list_171121)
    
    # Getting the type of 'r' (line 294)
    r_171122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'r')
    # Testing the type of an if condition (line 294)
    if_condition_171123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), r_171122)
    # Assigning a type to the variable 'if_condition_171123' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_171123', if_condition_171123)
    # SSA begins for if statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 295):
    
    # Assigning a Call to a Subscript (line 295):
    
    # Call to lagmul(...): (line 295)
    # Processing the call arguments (line 295)
    
    # Obtaining the type of the subscript
    int_171125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 36), 'int')
    # Getting the type of 'tmp' (line 295)
    tmp_171126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 32), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___171127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 32), tmp_171126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_171128 = invoke(stypy.reporting.localization.Localization(__file__, 295, 32), getitem___171127, int_171125)
    
    
    # Obtaining the type of the subscript
    int_171129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 42), 'int')
    # Getting the type of 'p' (line 295)
    p_171130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 40), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___171131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 40), p_171130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_171132 = invoke(stypy.reporting.localization.Localization(__file__, 295, 40), getitem___171131, int_171129)
    
    # Processing the call keyword arguments (line 295)
    kwargs_171133 = {}
    # Getting the type of 'lagmul' (line 295)
    lagmul_171124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'lagmul', False)
    # Calling lagmul(args, kwargs) (line 295)
    lagmul_call_result_171134 = invoke(stypy.reporting.localization.Localization(__file__, 295, 25), lagmul_171124, *[subscript_call_result_171128, subscript_call_result_171132], **kwargs_171133)
    
    # Getting the type of 'tmp' (line 295)
    tmp_171135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'tmp')
    int_171136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'int')
    # Storing an element on a container (line 295)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), tmp_171135, (int_171136, lagmul_call_result_171134))
    # SSA join for if statement (line 294)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 296):
    
    # Assigning a Name to a Name (line 296):
    # Getting the type of 'tmp' (line 296)
    tmp_171137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'tmp')
    # Assigning a type to the variable 'p' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'p', tmp_171137)
    
    # Assigning a Name to a Name (line 297):
    
    # Assigning a Name to a Name (line 297):
    # Getting the type of 'm' (line 297)
    m_171138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'm')
    # Assigning a type to the variable 'n' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'n', m_171138)
    # SSA join for while statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_171139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 17), 'int')
    # Getting the type of 'p' (line 298)
    p_171140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'p')
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___171141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), p_171140, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_171142 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), getitem___171141, int_171139)
    
    # Assigning a type to the variable 'stypy_return_type' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type', subscript_call_result_171142)
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lagfromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagfromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 234)
    stypy_return_type_171143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171143)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagfromroots'
    return stypy_return_type_171143

# Assigning a type to the variable 'lagfromroots' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'lagfromroots', lagfromroots)

@norecursion
def lagadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagadd'
    module_type_store = module_type_store.open_function_context('lagadd', 301, 0, False)
    
    # Passed parameters checking function
    lagadd.stypy_localization = localization
    lagadd.stypy_type_of_self = None
    lagadd.stypy_type_store = module_type_store
    lagadd.stypy_function_name = 'lagadd'
    lagadd.stypy_param_names_list = ['c1', 'c2']
    lagadd.stypy_varargs_param_name = None
    lagadd.stypy_kwargs_param_name = None
    lagadd.stypy_call_defaults = defaults
    lagadd.stypy_call_varargs = varargs
    lagadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagadd', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagadd', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagadd(...)' code ##################

    str_171144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, (-1)), 'str', '\n    Add one Laguerre series to another.\n\n    Returns the sum of two Laguerre series `c1` + `c2`.  The arguments\n    are sequences of coefficients ordered from lowest order term to\n    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the Laguerre series of their sum.\n\n    See Also\n    --------\n    lagsub, lagmul, lagdiv, lagpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the sum of two Laguerre series\n    is a Laguerre series (without having to "reproject" the result onto\n    the basis set) so addition, just like that of "standard" polynomials,\n    is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagadd\n    >>> lagadd([1, 2, 3], [1, 2, 3, 4])\n    array([ 2.,  4.,  6.,  4.])\n\n\n    ')
    
    # Assigning a Call to a List (line 340):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 340)
    # Processing the call arguments (line 340)
    
    # Obtaining an instance of the builtin type 'list' (line 340)
    list_171147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 340)
    # Adding element type (line 340)
    # Getting the type of 'c1' (line 340)
    c1_171148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 28), list_171147, c1_171148)
    # Adding element type (line 340)
    # Getting the type of 'c2' (line 340)
    c2_171149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 28), list_171147, c2_171149)
    
    # Processing the call keyword arguments (line 340)
    kwargs_171150 = {}
    # Getting the type of 'pu' (line 340)
    pu_171145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 340)
    as_series_171146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 15), pu_171145, 'as_series')
    # Calling as_series(args, kwargs) (line 340)
    as_series_call_result_171151 = invoke(stypy.reporting.localization.Localization(__file__, 340, 15), as_series_171146, *[list_171147], **kwargs_171150)
    
    # Assigning a type to the variable 'call_assignment_170771' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170771', as_series_call_result_171151)
    
    # Assigning a Call to a Name (line 340):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171155 = {}
    # Getting the type of 'call_assignment_170771' (line 340)
    call_assignment_170771_171152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170771', False)
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___171153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 4), call_assignment_170771_171152, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171156 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171153, *[int_171154], **kwargs_171155)
    
    # Assigning a type to the variable 'call_assignment_170772' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170772', getitem___call_result_171156)
    
    # Assigning a Name to a Name (line 340):
    # Getting the type of 'call_assignment_170772' (line 340)
    call_assignment_170772_171157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170772')
    # Assigning a type to the variable 'c1' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 5), 'c1', call_assignment_170772_171157)
    
    # Assigning a Call to a Name (line 340):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171161 = {}
    # Getting the type of 'call_assignment_170771' (line 340)
    call_assignment_170771_171158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170771', False)
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___171159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 4), call_assignment_170771_171158, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171162 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171159, *[int_171160], **kwargs_171161)
    
    # Assigning a type to the variable 'call_assignment_170773' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170773', getitem___call_result_171162)
    
    # Assigning a Name to a Name (line 340):
    # Getting the type of 'call_assignment_170773' (line 340)
    call_assignment_170773_171163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'call_assignment_170773')
    # Assigning a type to the variable 'c2' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 9), 'c2', call_assignment_170773_171163)
    
    
    
    # Call to len(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'c1' (line 341)
    c1_171165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 11), 'c1', False)
    # Processing the call keyword arguments (line 341)
    kwargs_171166 = {}
    # Getting the type of 'len' (line 341)
    len_171164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 7), 'len', False)
    # Calling len(args, kwargs) (line 341)
    len_call_result_171167 = invoke(stypy.reporting.localization.Localization(__file__, 341, 7), len_171164, *[c1_171165], **kwargs_171166)
    
    
    # Call to len(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'c2' (line 341)
    c2_171169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'c2', False)
    # Processing the call keyword arguments (line 341)
    kwargs_171170 = {}
    # Getting the type of 'len' (line 341)
    len_171168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'len', False)
    # Calling len(args, kwargs) (line 341)
    len_call_result_171171 = invoke(stypy.reporting.localization.Localization(__file__, 341, 17), len_171168, *[c2_171169], **kwargs_171170)
    
    # Applying the binary operator '>' (line 341)
    result_gt_171172 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 7), '>', len_call_result_171167, len_call_result_171171)
    
    # Testing the type of an if condition (line 341)
    if_condition_171173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 4), result_gt_171172)
    # Assigning a type to the variable 'if_condition_171173' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'if_condition_171173', if_condition_171173)
    # SSA begins for if statement (line 341)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 342)
    c1_171174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 342)
    c2_171175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'c2')
    # Obtaining the member 'size' of a type (line 342)
    size_171176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), c2_171175, 'size')
    slice_171177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 342, 8), None, size_171176, None)
    # Getting the type of 'c1' (line 342)
    c1_171178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___171179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), c1_171178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_171180 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), getitem___171179, slice_171177)
    
    # Getting the type of 'c2' (line 342)
    c2_171181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 24), 'c2')
    # Applying the binary operator '+=' (line 342)
    result_iadd_171182 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 8), '+=', subscript_call_result_171180, c2_171181)
    # Getting the type of 'c1' (line 342)
    c1_171183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'c1')
    # Getting the type of 'c2' (line 342)
    c2_171184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'c2')
    # Obtaining the member 'size' of a type (line 342)
    size_171185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), c2_171184, 'size')
    slice_171186 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 342, 8), None, size_171185, None)
    # Storing an element on a container (line 342)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 8), c1_171183, (slice_171186, result_iadd_171182))
    
    
    # Assigning a Name to a Name (line 343):
    
    # Assigning a Name to a Name (line 343):
    # Getting the type of 'c1' (line 343)
    c1_171187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'ret', c1_171187)
    # SSA branch for the else part of an if statement (line 341)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'c2' (line 345)
    c2_171188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 345)
    c1_171189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'c1')
    # Obtaining the member 'size' of a type (line 345)
    size_171190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), c1_171189, 'size')
    slice_171191 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 345, 8), None, size_171190, None)
    # Getting the type of 'c2' (line 345)
    c2_171192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___171193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), c2_171192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_171194 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___171193, slice_171191)
    
    # Getting the type of 'c1' (line 345)
    c1_171195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'c1')
    # Applying the binary operator '+=' (line 345)
    result_iadd_171196 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 8), '+=', subscript_call_result_171194, c1_171195)
    # Getting the type of 'c2' (line 345)
    c2_171197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'c2')
    # Getting the type of 'c1' (line 345)
    c1_171198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'c1')
    # Obtaining the member 'size' of a type (line 345)
    size_171199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), c1_171198, 'size')
    slice_171200 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 345, 8), None, size_171199, None)
    # Storing an element on a container (line 345)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), c2_171197, (slice_171200, result_iadd_171196))
    
    
    # Assigning a Name to a Name (line 346):
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'c2' (line 346)
    c2_171201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'ret', c2_171201)
    # SSA join for if statement (line 341)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'ret' (line 347)
    ret_171204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'ret', False)
    # Processing the call keyword arguments (line 347)
    kwargs_171205 = {}
    # Getting the type of 'pu' (line 347)
    pu_171202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 347)
    trimseq_171203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 11), pu_171202, 'trimseq')
    # Calling trimseq(args, kwargs) (line 347)
    trimseq_call_result_171206 = invoke(stypy.reporting.localization.Localization(__file__, 347, 11), trimseq_171203, *[ret_171204], **kwargs_171205)
    
    # Assigning a type to the variable 'stypy_return_type' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type', trimseq_call_result_171206)
    
    # ################# End of 'lagadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagadd' in the type store
    # Getting the type of 'stypy_return_type' (line 301)
    stypy_return_type_171207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171207)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagadd'
    return stypy_return_type_171207

# Assigning a type to the variable 'lagadd' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'lagadd', lagadd)

@norecursion
def lagsub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagsub'
    module_type_store = module_type_store.open_function_context('lagsub', 350, 0, False)
    
    # Passed parameters checking function
    lagsub.stypy_localization = localization
    lagsub.stypy_type_of_self = None
    lagsub.stypy_type_store = module_type_store
    lagsub.stypy_function_name = 'lagsub'
    lagsub.stypy_param_names_list = ['c1', 'c2']
    lagsub.stypy_varargs_param_name = None
    lagsub.stypy_kwargs_param_name = None
    lagsub.stypy_call_defaults = defaults
    lagsub.stypy_call_varargs = varargs
    lagsub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagsub', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagsub', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagsub(...)' code ##################

    str_171208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, (-1)), 'str', '\n    Subtract one Laguerre series from another.\n\n    Returns the difference of two Laguerre series `c1` - `c2`.  The\n    sequences of coefficients are from lowest order term to highest, i.e.,\n    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Laguerre series coefficients representing their difference.\n\n    See Also\n    --------\n    lagadd, lagmul, lagdiv, lagpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the difference of two Laguerre\n    series is a Laguerre series (without having to "reproject" the result\n    onto the basis set) so subtraction, just like that of "standard"\n    polynomials, is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagsub\n    >>> lagsub([1, 2, 3, 4], [1, 2, 3])\n    array([ 0.,  0.,  0.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 388):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Obtaining an instance of the builtin type 'list' (line 388)
    list_171211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 388)
    # Adding element type (line 388)
    # Getting the type of 'c1' (line 388)
    c1_171212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 28), list_171211, c1_171212)
    # Adding element type (line 388)
    # Getting the type of 'c2' (line 388)
    c2_171213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 28), list_171211, c2_171213)
    
    # Processing the call keyword arguments (line 388)
    kwargs_171214 = {}
    # Getting the type of 'pu' (line 388)
    pu_171209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 388)
    as_series_171210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), pu_171209, 'as_series')
    # Calling as_series(args, kwargs) (line 388)
    as_series_call_result_171215 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), as_series_171210, *[list_171211], **kwargs_171214)
    
    # Assigning a type to the variable 'call_assignment_170774' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170774', as_series_call_result_171215)
    
    # Assigning a Call to a Name (line 388):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171219 = {}
    # Getting the type of 'call_assignment_170774' (line 388)
    call_assignment_170774_171216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170774', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___171217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 4), call_assignment_170774_171216, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171220 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171217, *[int_171218], **kwargs_171219)
    
    # Assigning a type to the variable 'call_assignment_170775' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170775', getitem___call_result_171220)
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'call_assignment_170775' (line 388)
    call_assignment_170775_171221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170775')
    # Assigning a type to the variable 'c1' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 5), 'c1', call_assignment_170775_171221)
    
    # Assigning a Call to a Name (line 388):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171225 = {}
    # Getting the type of 'call_assignment_170774' (line 388)
    call_assignment_170774_171222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170774', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___171223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 4), call_assignment_170774_171222, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171226 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171223, *[int_171224], **kwargs_171225)
    
    # Assigning a type to the variable 'call_assignment_170776' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170776', getitem___call_result_171226)
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'call_assignment_170776' (line 388)
    call_assignment_170776_171227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'call_assignment_170776')
    # Assigning a type to the variable 'c2' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 9), 'c2', call_assignment_170776_171227)
    
    
    
    # Call to len(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'c1' (line 389)
    c1_171229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'c1', False)
    # Processing the call keyword arguments (line 389)
    kwargs_171230 = {}
    # Getting the type of 'len' (line 389)
    len_171228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 'len', False)
    # Calling len(args, kwargs) (line 389)
    len_call_result_171231 = invoke(stypy.reporting.localization.Localization(__file__, 389, 7), len_171228, *[c1_171229], **kwargs_171230)
    
    
    # Call to len(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'c2' (line 389)
    c2_171233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'c2', False)
    # Processing the call keyword arguments (line 389)
    kwargs_171234 = {}
    # Getting the type of 'len' (line 389)
    len_171232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'len', False)
    # Calling len(args, kwargs) (line 389)
    len_call_result_171235 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), len_171232, *[c2_171233], **kwargs_171234)
    
    # Applying the binary operator '>' (line 389)
    result_gt_171236 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), '>', len_call_result_171231, len_call_result_171235)
    
    # Testing the type of an if condition (line 389)
    if_condition_171237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 4), result_gt_171236)
    # Assigning a type to the variable 'if_condition_171237' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'if_condition_171237', if_condition_171237)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 390)
    c1_171238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 390)
    c2_171239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'c2')
    # Obtaining the member 'size' of a type (line 390)
    size_171240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), c2_171239, 'size')
    slice_171241 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 390, 8), None, size_171240, None)
    # Getting the type of 'c1' (line 390)
    c1_171242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___171243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), c1_171242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
    subscript_call_result_171244 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), getitem___171243, slice_171241)
    
    # Getting the type of 'c2' (line 390)
    c2_171245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'c2')
    # Applying the binary operator '-=' (line 390)
    result_isub_171246 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 8), '-=', subscript_call_result_171244, c2_171245)
    # Getting the type of 'c1' (line 390)
    c1_171247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'c1')
    # Getting the type of 'c2' (line 390)
    c2_171248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'c2')
    # Obtaining the member 'size' of a type (line 390)
    size_171249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), c2_171248, 'size')
    slice_171250 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 390, 8), None, size_171249, None)
    # Storing an element on a container (line 390)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 8), c1_171247, (slice_171250, result_isub_171246))
    
    
    # Assigning a Name to a Name (line 391):
    
    # Assigning a Name to a Name (line 391):
    # Getting the type of 'c1' (line 391)
    c1_171251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'ret', c1_171251)
    # SSA branch for the else part of an if statement (line 389)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a UnaryOp to a Name (line 393):
    
    # Assigning a UnaryOp to a Name (line 393):
    
    # Getting the type of 'c2' (line 393)
    c2_171252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 14), 'c2')
    # Applying the 'usub' unary operator (line 393)
    result___neg___171253 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 13), 'usub', c2_171252)
    
    # Assigning a type to the variable 'c2' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'c2', result___neg___171253)
    
    # Getting the type of 'c2' (line 394)
    c2_171254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 394)
    c1_171255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'c1')
    # Obtaining the member 'size' of a type (line 394)
    size_171256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), c1_171255, 'size')
    slice_171257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 394, 8), None, size_171256, None)
    # Getting the type of 'c2' (line 394)
    c2_171258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___171259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), c2_171258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_171260 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), getitem___171259, slice_171257)
    
    # Getting the type of 'c1' (line 394)
    c1_171261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'c1')
    # Applying the binary operator '+=' (line 394)
    result_iadd_171262 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 8), '+=', subscript_call_result_171260, c1_171261)
    # Getting the type of 'c2' (line 394)
    c2_171263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'c2')
    # Getting the type of 'c1' (line 394)
    c1_171264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'c1')
    # Obtaining the member 'size' of a type (line 394)
    size_171265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), c1_171264, 'size')
    slice_171266 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 394, 8), None, size_171265, None)
    # Storing an element on a container (line 394)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 8), c2_171263, (slice_171266, result_iadd_171262))
    
    
    # Assigning a Name to a Name (line 395):
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'c2' (line 395)
    c2_171267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'ret', c2_171267)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'ret' (line 396)
    ret_171270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'ret', False)
    # Processing the call keyword arguments (line 396)
    kwargs_171271 = {}
    # Getting the type of 'pu' (line 396)
    pu_171268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 396)
    trimseq_171269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 11), pu_171268, 'trimseq')
    # Calling trimseq(args, kwargs) (line 396)
    trimseq_call_result_171272 = invoke(stypy.reporting.localization.Localization(__file__, 396, 11), trimseq_171269, *[ret_171270], **kwargs_171271)
    
    # Assigning a type to the variable 'stypy_return_type' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type', trimseq_call_result_171272)
    
    # ################# End of 'lagsub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagsub' in the type store
    # Getting the type of 'stypy_return_type' (line 350)
    stypy_return_type_171273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagsub'
    return stypy_return_type_171273

# Assigning a type to the variable 'lagsub' (line 350)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'lagsub', lagsub)

@norecursion
def lagmulx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagmulx'
    module_type_store = module_type_store.open_function_context('lagmulx', 399, 0, False)
    
    # Passed parameters checking function
    lagmulx.stypy_localization = localization
    lagmulx.stypy_type_of_self = None
    lagmulx.stypy_type_store = module_type_store
    lagmulx.stypy_function_name = 'lagmulx'
    lagmulx.stypy_param_names_list = ['c']
    lagmulx.stypy_varargs_param_name = None
    lagmulx.stypy_kwargs_param_name = None
    lagmulx.stypy_call_defaults = defaults
    lagmulx.stypy_call_varargs = varargs
    lagmulx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagmulx', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagmulx', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagmulx(...)' code ##################

    str_171274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, (-1)), 'str', 'Multiply a Laguerre series by x.\n\n    Multiply the Laguerre series `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    Notes\n    -----\n    The multiplication uses the recursion relationship for Laguerre\n    polynomials in the form\n\n    .. math::\n\n    xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagmulx\n    >>> lagmulx([1, 2, 3])\n    array([ -1.,  -1.,  11.,  -9.])\n\n    ')
    
    # Assigning a Call to a List (line 434):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 434)
    # Processing the call arguments (line 434)
    
    # Obtaining an instance of the builtin type 'list' (line 434)
    list_171277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 434)
    # Adding element type (line 434)
    # Getting the type of 'c' (line 434)
    c_171278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 23), list_171277, c_171278)
    
    # Processing the call keyword arguments (line 434)
    kwargs_171279 = {}
    # Getting the type of 'pu' (line 434)
    pu_171275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 434)
    as_series_171276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 10), pu_171275, 'as_series')
    # Calling as_series(args, kwargs) (line 434)
    as_series_call_result_171280 = invoke(stypy.reporting.localization.Localization(__file__, 434, 10), as_series_171276, *[list_171277], **kwargs_171279)
    
    # Assigning a type to the variable 'call_assignment_170777' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_170777', as_series_call_result_171280)
    
    # Assigning a Call to a Name (line 434):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171284 = {}
    # Getting the type of 'call_assignment_170777' (line 434)
    call_assignment_170777_171281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_170777', False)
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___171282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 4), call_assignment_170777_171281, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171285 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171282, *[int_171283], **kwargs_171284)
    
    # Assigning a type to the variable 'call_assignment_170778' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_170778', getitem___call_result_171285)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'call_assignment_170778' (line 434)
    call_assignment_170778_171286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'call_assignment_170778')
    # Assigning a type to the variable 'c' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 5), 'c', call_assignment_170778_171286)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'c' (line 436)
    c_171288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'c', False)
    # Processing the call keyword arguments (line 436)
    kwargs_171289 = {}
    # Getting the type of 'len' (line 436)
    len_171287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 7), 'len', False)
    # Calling len(args, kwargs) (line 436)
    len_call_result_171290 = invoke(stypy.reporting.localization.Localization(__file__, 436, 7), len_171287, *[c_171288], **kwargs_171289)
    
    int_171291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 17), 'int')
    # Applying the binary operator '==' (line 436)
    result_eq_171292 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 7), '==', len_call_result_171290, int_171291)
    
    
    
    # Obtaining the type of the subscript
    int_171293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 25), 'int')
    # Getting the type of 'c' (line 436)
    c_171294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'c')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___171295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 23), c_171294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_171296 = invoke(stypy.reporting.localization.Localization(__file__, 436, 23), getitem___171295, int_171293)
    
    int_171297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 31), 'int')
    # Applying the binary operator '==' (line 436)
    result_eq_171298 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 23), '==', subscript_call_result_171296, int_171297)
    
    # Applying the binary operator 'and' (line 436)
    result_and_keyword_171299 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 7), 'and', result_eq_171292, result_eq_171298)
    
    # Testing the type of an if condition (line 436)
    if_condition_171300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 4), result_and_keyword_171299)
    # Assigning a type to the variable 'if_condition_171300' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'if_condition_171300', if_condition_171300)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 437)
    c_171301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', c_171301)
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 439):
    
    # Assigning a Call to a Name (line 439):
    
    # Call to empty(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Call to len(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'c' (line 439)
    c_171305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'c', False)
    # Processing the call keyword arguments (line 439)
    kwargs_171306 = {}
    # Getting the type of 'len' (line 439)
    len_171304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 19), 'len', False)
    # Calling len(args, kwargs) (line 439)
    len_call_result_171307 = invoke(stypy.reporting.localization.Localization(__file__, 439, 19), len_171304, *[c_171305], **kwargs_171306)
    
    int_171308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 28), 'int')
    # Applying the binary operator '+' (line 439)
    result_add_171309 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 19), '+', len_call_result_171307, int_171308)
    
    # Processing the call keyword arguments (line 439)
    # Getting the type of 'c' (line 439)
    c_171310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 37), 'c', False)
    # Obtaining the member 'dtype' of a type (line 439)
    dtype_171311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 37), c_171310, 'dtype')
    keyword_171312 = dtype_171311
    kwargs_171313 = {'dtype': keyword_171312}
    # Getting the type of 'np' (line 439)
    np_171302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 439)
    empty_171303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 10), np_171302, 'empty')
    # Calling empty(args, kwargs) (line 439)
    empty_call_result_171314 = invoke(stypy.reporting.localization.Localization(__file__, 439, 10), empty_171303, *[result_add_171309], **kwargs_171313)
    
    # Assigning a type to the variable 'prd' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'prd', empty_call_result_171314)
    
    # Assigning a Subscript to a Subscript (line 440):
    
    # Assigning a Subscript to a Subscript (line 440):
    
    # Obtaining the type of the subscript
    int_171315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 15), 'int')
    # Getting the type of 'c' (line 440)
    c_171316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___171317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 13), c_171316, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_171318 = invoke(stypy.reporting.localization.Localization(__file__, 440, 13), getitem___171317, int_171315)
    
    # Getting the type of 'prd' (line 440)
    prd_171319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'prd')
    int_171320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 8), 'int')
    # Storing an element on a container (line 440)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 4), prd_171319, (int_171320, subscript_call_result_171318))
    
    # Assigning a UnaryOp to a Subscript (line 441):
    
    # Assigning a UnaryOp to a Subscript (line 441):
    
    
    # Obtaining the type of the subscript
    int_171321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 16), 'int')
    # Getting the type of 'c' (line 441)
    c_171322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'c')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___171323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 14), c_171322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_171324 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), getitem___171323, int_171321)
    
    # Applying the 'usub' unary operator (line 441)
    result___neg___171325 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 13), 'usub', subscript_call_result_171324)
    
    # Getting the type of 'prd' (line 441)
    prd_171326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'prd')
    int_171327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 8), 'int')
    # Storing an element on a container (line 441)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 4), prd_171326, (int_171327, result___neg___171325))
    
    
    # Call to range(...): (line 442)
    # Processing the call arguments (line 442)
    int_171329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'int')
    
    # Call to len(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'c' (line 442)
    c_171331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 26), 'c', False)
    # Processing the call keyword arguments (line 442)
    kwargs_171332 = {}
    # Getting the type of 'len' (line 442)
    len_171330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 22), 'len', False)
    # Calling len(args, kwargs) (line 442)
    len_call_result_171333 = invoke(stypy.reporting.localization.Localization(__file__, 442, 22), len_171330, *[c_171331], **kwargs_171332)
    
    # Processing the call keyword arguments (line 442)
    kwargs_171334 = {}
    # Getting the type of 'range' (line 442)
    range_171328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 13), 'range', False)
    # Calling range(args, kwargs) (line 442)
    range_call_result_171335 = invoke(stypy.reporting.localization.Localization(__file__, 442, 13), range_171328, *[int_171329, len_call_result_171333], **kwargs_171334)
    
    # Testing the type of a for loop iterable (line 442)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 442, 4), range_call_result_171335)
    # Getting the type of the for loop variable (line 442)
    for_loop_var_171336 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 442, 4), range_call_result_171335)
    # Assigning a type to the variable 'i' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'i', for_loop_var_171336)
    # SSA begins for a for statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 443):
    
    # Assigning a BinOp to a Subscript (line 443):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 443)
    i_171337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'i')
    # Getting the type of 'c' (line 443)
    c_171338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 443)
    getitem___171339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 22), c_171338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 443)
    subscript_call_result_171340 = invoke(stypy.reporting.localization.Localization(__file__, 443, 22), getitem___171339, i_171337)
    
    # Applying the 'usub' unary operator (line 443)
    result___neg___171341 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 21), 'usub', subscript_call_result_171340)
    
    # Getting the type of 'i' (line 443)
    i_171342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 28), 'i')
    int_171343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 32), 'int')
    # Applying the binary operator '+' (line 443)
    result_add_171344 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 28), '+', i_171342, int_171343)
    
    # Applying the binary operator '*' (line 443)
    result_mul_171345 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 21), '*', result___neg___171341, result_add_171344)
    
    # Getting the type of 'prd' (line 443)
    prd_171346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'prd')
    # Getting the type of 'i' (line 443)
    i_171347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'i')
    int_171348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 16), 'int')
    # Applying the binary operator '+' (line 443)
    result_add_171349 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 12), '+', i_171347, int_171348)
    
    # Storing an element on a container (line 443)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 8), prd_171346, (result_add_171349, result_mul_171345))
    
    # Getting the type of 'prd' (line 444)
    prd_171350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'prd')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 444)
    i_171351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'i')
    # Getting the type of 'prd' (line 444)
    prd_171352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'prd')
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___171353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), prd_171352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_171354 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), getitem___171353, i_171351)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 444)
    i_171355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'i')
    # Getting the type of 'c' (line 444)
    c_171356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 18), 'c')
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___171357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 18), c_171356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_171358 = invoke(stypy.reporting.localization.Localization(__file__, 444, 18), getitem___171357, i_171355)
    
    int_171359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 24), 'int')
    # Getting the type of 'i' (line 444)
    i_171360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 26), 'i')
    # Applying the binary operator '*' (line 444)
    result_mul_171361 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 24), '*', int_171359, i_171360)
    
    int_171362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 30), 'int')
    # Applying the binary operator '+' (line 444)
    result_add_171363 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 24), '+', result_mul_171361, int_171362)
    
    # Applying the binary operator '*' (line 444)
    result_mul_171364 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 18), '*', subscript_call_result_171358, result_add_171363)
    
    # Applying the binary operator '+=' (line 444)
    result_iadd_171365 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 8), '+=', subscript_call_result_171354, result_mul_171364)
    # Getting the type of 'prd' (line 444)
    prd_171366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'prd')
    # Getting the type of 'i' (line 444)
    i_171367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'i')
    # Storing an element on a container (line 444)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 8), prd_171366, (i_171367, result_iadd_171365))
    
    
    # Getting the type of 'prd' (line 445)
    prd_171368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'prd')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 445)
    i_171369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'i')
    int_171370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'int')
    # Applying the binary operator '-' (line 445)
    result_sub_171371 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 12), '-', i_171369, int_171370)
    
    # Getting the type of 'prd' (line 445)
    prd_171372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'prd')
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___171373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), prd_171372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_171374 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___171373, result_sub_171371)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 445)
    i_171375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'i')
    # Getting the type of 'c' (line 445)
    c_171376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___171377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 22), c_171376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_171378 = invoke(stypy.reporting.localization.Localization(__file__, 445, 22), getitem___171377, i_171375)
    
    # Getting the type of 'i' (line 445)
    i_171379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 27), 'i')
    # Applying the binary operator '*' (line 445)
    result_mul_171380 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 22), '*', subscript_call_result_171378, i_171379)
    
    # Applying the binary operator '-=' (line 445)
    result_isub_171381 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 8), '-=', subscript_call_result_171374, result_mul_171380)
    # Getting the type of 'prd' (line 445)
    prd_171382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'prd')
    # Getting the type of 'i' (line 445)
    i_171383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'i')
    int_171384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'int')
    # Applying the binary operator '-' (line 445)
    result_sub_171385 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 12), '-', i_171383, int_171384)
    
    # Storing an element on a container (line 445)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 8), prd_171382, (result_sub_171385, result_isub_171381))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 446)
    prd_171386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 11), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type', prd_171386)
    
    # ################# End of 'lagmulx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagmulx' in the type store
    # Getting the type of 'stypy_return_type' (line 399)
    stypy_return_type_171387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171387)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagmulx'
    return stypy_return_type_171387

# Assigning a type to the variable 'lagmulx' (line 399)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'lagmulx', lagmulx)

@norecursion
def lagmul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagmul'
    module_type_store = module_type_store.open_function_context('lagmul', 449, 0, False)
    
    # Passed parameters checking function
    lagmul.stypy_localization = localization
    lagmul.stypy_type_of_self = None
    lagmul.stypy_type_store = module_type_store
    lagmul.stypy_function_name = 'lagmul'
    lagmul.stypy_param_names_list = ['c1', 'c2']
    lagmul.stypy_varargs_param_name = None
    lagmul.stypy_kwargs_param_name = None
    lagmul.stypy_call_defaults = defaults
    lagmul.stypy_call_varargs = varargs
    lagmul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagmul', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagmul', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagmul(...)' code ##################

    str_171388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, (-1)), 'str', '\n    Multiply one Laguerre series by another.\n\n    Returns the product of two Laguerre series `c1` * `c2`.  The arguments\n    are sequences of coefficients, from lowest order "term" to highest,\n    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Laguerre series coefficients representing their product.\n\n    See Also\n    --------\n    lagadd, lagsub, lagdiv, lagpow\n\n    Notes\n    -----\n    In general, the (polynomial) product of two C-series results in terms\n    that are not in the Laguerre polynomial basis set.  Thus, to express\n    the product as a Laguerre series, it is necessary to "reproject" the\n    product onto said basis set, which may produce "unintuitive" (but\n    correct) results; see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagmul\n    >>> lagmul([1, 2, 3], [0, 1, 2])\n    array([  8., -13.,  38., -51.,  36.])\n\n    ')
    
    # Assigning a Call to a List (line 488):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Obtaining an instance of the builtin type 'list' (line 488)
    list_171391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 488)
    # Adding element type (line 488)
    # Getting the type of 'c1' (line 488)
    c1_171392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 28), list_171391, c1_171392)
    # Adding element type (line 488)
    # Getting the type of 'c2' (line 488)
    c2_171393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 28), list_171391, c2_171393)
    
    # Processing the call keyword arguments (line 488)
    kwargs_171394 = {}
    # Getting the type of 'pu' (line 488)
    pu_171389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 488)
    as_series_171390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 15), pu_171389, 'as_series')
    # Calling as_series(args, kwargs) (line 488)
    as_series_call_result_171395 = invoke(stypy.reporting.localization.Localization(__file__, 488, 15), as_series_171390, *[list_171391], **kwargs_171394)
    
    # Assigning a type to the variable 'call_assignment_170779' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170779', as_series_call_result_171395)
    
    # Assigning a Call to a Name (line 488):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171399 = {}
    # Getting the type of 'call_assignment_170779' (line 488)
    call_assignment_170779_171396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170779', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___171397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 4), call_assignment_170779_171396, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171400 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171397, *[int_171398], **kwargs_171399)
    
    # Assigning a type to the variable 'call_assignment_170780' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170780', getitem___call_result_171400)
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'call_assignment_170780' (line 488)
    call_assignment_170780_171401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170780')
    # Assigning a type to the variable 'c1' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 5), 'c1', call_assignment_170780_171401)
    
    # Assigning a Call to a Name (line 488):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171405 = {}
    # Getting the type of 'call_assignment_170779' (line 488)
    call_assignment_170779_171402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170779', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___171403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 4), call_assignment_170779_171402, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171406 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171403, *[int_171404], **kwargs_171405)
    
    # Assigning a type to the variable 'call_assignment_170781' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170781', getitem___call_result_171406)
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'call_assignment_170781' (line 488)
    call_assignment_170781_171407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'call_assignment_170781')
    # Assigning a type to the variable 'c2' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 9), 'c2', call_assignment_170781_171407)
    
    
    
    # Call to len(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'c1' (line 490)
    c1_171409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'c1', False)
    # Processing the call keyword arguments (line 490)
    kwargs_171410 = {}
    # Getting the type of 'len' (line 490)
    len_171408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 7), 'len', False)
    # Calling len(args, kwargs) (line 490)
    len_call_result_171411 = invoke(stypy.reporting.localization.Localization(__file__, 490, 7), len_171408, *[c1_171409], **kwargs_171410)
    
    
    # Call to len(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'c2' (line 490)
    c2_171413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 21), 'c2', False)
    # Processing the call keyword arguments (line 490)
    kwargs_171414 = {}
    # Getting the type of 'len' (line 490)
    len_171412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 17), 'len', False)
    # Calling len(args, kwargs) (line 490)
    len_call_result_171415 = invoke(stypy.reporting.localization.Localization(__file__, 490, 17), len_171412, *[c2_171413], **kwargs_171414)
    
    # Applying the binary operator '>' (line 490)
    result_gt_171416 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 7), '>', len_call_result_171411, len_call_result_171415)
    
    # Testing the type of an if condition (line 490)
    if_condition_171417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 4), result_gt_171416)
    # Assigning a type to the variable 'if_condition_171417' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'if_condition_171417', if_condition_171417)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 491):
    
    # Assigning a Name to a Name (line 491):
    # Getting the type of 'c2' (line 491)
    c2_171418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'c2')
    # Assigning a type to the variable 'c' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'c', c2_171418)
    
    # Assigning a Name to a Name (line 492):
    
    # Assigning a Name to a Name (line 492):
    # Getting the type of 'c1' (line 492)
    c1_171419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 13), 'c1')
    # Assigning a type to the variable 'xs' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'xs', c1_171419)
    # SSA branch for the else part of an if statement (line 490)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 494):
    
    # Assigning a Name to a Name (line 494):
    # Getting the type of 'c1' (line 494)
    c1_171420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'c1')
    # Assigning a type to the variable 'c' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'c', c1_171420)
    
    # Assigning a Name to a Name (line 495):
    
    # Assigning a Name to a Name (line 495):
    # Getting the type of 'c2' (line 495)
    c2_171421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 13), 'c2')
    # Assigning a type to the variable 'xs' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'xs', c2_171421)
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 'c' (line 497)
    c_171423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 11), 'c', False)
    # Processing the call keyword arguments (line 497)
    kwargs_171424 = {}
    # Getting the type of 'len' (line 497)
    len_171422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 7), 'len', False)
    # Calling len(args, kwargs) (line 497)
    len_call_result_171425 = invoke(stypy.reporting.localization.Localization(__file__, 497, 7), len_171422, *[c_171423], **kwargs_171424)
    
    int_171426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 17), 'int')
    # Applying the binary operator '==' (line 497)
    result_eq_171427 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 7), '==', len_call_result_171425, int_171426)
    
    # Testing the type of an if condition (line 497)
    if_condition_171428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 4), result_eq_171427)
    # Assigning a type to the variable 'if_condition_171428' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'if_condition_171428', if_condition_171428)
    # SSA begins for if statement (line 497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 498):
    
    # Assigning a BinOp to a Name (line 498):
    
    # Obtaining the type of the subscript
    int_171429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 15), 'int')
    # Getting the type of 'c' (line 498)
    c_171430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___171431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 13), c_171430, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_171432 = invoke(stypy.reporting.localization.Localization(__file__, 498, 13), getitem___171431, int_171429)
    
    # Getting the type of 'xs' (line 498)
    xs_171433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 18), 'xs')
    # Applying the binary operator '*' (line 498)
    result_mul_171434 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 13), '*', subscript_call_result_171432, xs_171433)
    
    # Assigning a type to the variable 'c0' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'c0', result_mul_171434)
    
    # Assigning a Num to a Name (line 499):
    
    # Assigning a Num to a Name (line 499):
    int_171435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 13), 'int')
    # Assigning a type to the variable 'c1' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'c1', int_171435)
    # SSA branch for the else part of an if statement (line 497)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'c' (line 500)
    c_171437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 13), 'c', False)
    # Processing the call keyword arguments (line 500)
    kwargs_171438 = {}
    # Getting the type of 'len' (line 500)
    len_171436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 9), 'len', False)
    # Calling len(args, kwargs) (line 500)
    len_call_result_171439 = invoke(stypy.reporting.localization.Localization(__file__, 500, 9), len_171436, *[c_171437], **kwargs_171438)
    
    int_171440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 19), 'int')
    # Applying the binary operator '==' (line 500)
    result_eq_171441 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 9), '==', len_call_result_171439, int_171440)
    
    # Testing the type of an if condition (line 500)
    if_condition_171442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 9), result_eq_171441)
    # Assigning a type to the variable 'if_condition_171442' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 9), 'if_condition_171442', if_condition_171442)
    # SSA begins for if statement (line 500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 501):
    
    # Assigning a BinOp to a Name (line 501):
    
    # Obtaining the type of the subscript
    int_171443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 15), 'int')
    # Getting the type of 'c' (line 501)
    c_171444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___171445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 13), c_171444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_171446 = invoke(stypy.reporting.localization.Localization(__file__, 501, 13), getitem___171445, int_171443)
    
    # Getting the type of 'xs' (line 501)
    xs_171447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 18), 'xs')
    # Applying the binary operator '*' (line 501)
    result_mul_171448 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 13), '*', subscript_call_result_171446, xs_171447)
    
    # Assigning a type to the variable 'c0' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'c0', result_mul_171448)
    
    # Assigning a BinOp to a Name (line 502):
    
    # Assigning a BinOp to a Name (line 502):
    
    # Obtaining the type of the subscript
    int_171449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 15), 'int')
    # Getting the type of 'c' (line 502)
    c_171450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___171451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 13), c_171450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_171452 = invoke(stypy.reporting.localization.Localization(__file__, 502, 13), getitem___171451, int_171449)
    
    # Getting the type of 'xs' (line 502)
    xs_171453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 18), 'xs')
    # Applying the binary operator '*' (line 502)
    result_mul_171454 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 13), '*', subscript_call_result_171452, xs_171453)
    
    # Assigning a type to the variable 'c1' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'c1', result_mul_171454)
    # SSA branch for the else part of an if statement (line 500)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to len(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'c' (line 504)
    c_171456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'c', False)
    # Processing the call keyword arguments (line 504)
    kwargs_171457 = {}
    # Getting the type of 'len' (line 504)
    len_171455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 13), 'len', False)
    # Calling len(args, kwargs) (line 504)
    len_call_result_171458 = invoke(stypy.reporting.localization.Localization(__file__, 504, 13), len_171455, *[c_171456], **kwargs_171457)
    
    # Assigning a type to the variable 'nd' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'nd', len_call_result_171458)
    
    # Assigning a BinOp to a Name (line 505):
    
    # Assigning a BinOp to a Name (line 505):
    
    # Obtaining the type of the subscript
    int_171459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 15), 'int')
    # Getting the type of 'c' (line 505)
    c_171460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___171461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 13), c_171460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_171462 = invoke(stypy.reporting.localization.Localization(__file__, 505, 13), getitem___171461, int_171459)
    
    # Getting the type of 'xs' (line 505)
    xs_171463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 19), 'xs')
    # Applying the binary operator '*' (line 505)
    result_mul_171464 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 13), '*', subscript_call_result_171462, xs_171463)
    
    # Assigning a type to the variable 'c0' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'c0', result_mul_171464)
    
    # Assigning a BinOp to a Name (line 506):
    
    # Assigning a BinOp to a Name (line 506):
    
    # Obtaining the type of the subscript
    int_171465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 15), 'int')
    # Getting the type of 'c' (line 506)
    c_171466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___171467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 13), c_171466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_171468 = invoke(stypy.reporting.localization.Localization(__file__, 506, 13), getitem___171467, int_171465)
    
    # Getting the type of 'xs' (line 506)
    xs_171469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 19), 'xs')
    # Applying the binary operator '*' (line 506)
    result_mul_171470 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 13), '*', subscript_call_result_171468, xs_171469)
    
    # Assigning a type to the variable 'c1' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'c1', result_mul_171470)
    
    
    # Call to range(...): (line 507)
    # Processing the call arguments (line 507)
    int_171472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 23), 'int')
    
    # Call to len(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'c' (line 507)
    c_171474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 30), 'c', False)
    # Processing the call keyword arguments (line 507)
    kwargs_171475 = {}
    # Getting the type of 'len' (line 507)
    len_171473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 26), 'len', False)
    # Calling len(args, kwargs) (line 507)
    len_call_result_171476 = invoke(stypy.reporting.localization.Localization(__file__, 507, 26), len_171473, *[c_171474], **kwargs_171475)
    
    int_171477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 35), 'int')
    # Applying the binary operator '+' (line 507)
    result_add_171478 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 26), '+', len_call_result_171476, int_171477)
    
    # Processing the call keyword arguments (line 507)
    kwargs_171479 = {}
    # Getting the type of 'range' (line 507)
    range_171471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 17), 'range', False)
    # Calling range(args, kwargs) (line 507)
    range_call_result_171480 = invoke(stypy.reporting.localization.Localization(__file__, 507, 17), range_171471, *[int_171472, result_add_171478], **kwargs_171479)
    
    # Testing the type of a for loop iterable (line 507)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 507, 8), range_call_result_171480)
    # Getting the type of the for loop variable (line 507)
    for_loop_var_171481 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 507, 8), range_call_result_171480)
    # Assigning a type to the variable 'i' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'i', for_loop_var_171481)
    # SSA begins for a for statement (line 507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 508):
    
    # Assigning a Name to a Name (line 508):
    # Getting the type of 'c0' (line 508)
    c0_171482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'tmp', c0_171482)
    
    # Assigning a BinOp to a Name (line 509):
    
    # Assigning a BinOp to a Name (line 509):
    # Getting the type of 'nd' (line 509)
    nd_171483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'nd')
    int_171484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 22), 'int')
    # Applying the binary operator '-' (line 509)
    result_sub_171485 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 17), '-', nd_171483, int_171484)
    
    # Assigning a type to the variable 'nd' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'nd', result_sub_171485)
    
    # Assigning a Call to a Name (line 510):
    
    # Assigning a Call to a Name (line 510):
    
    # Call to lagsub(...): (line 510)
    # Processing the call arguments (line 510)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 510)
    i_171487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 27), 'i', False)
    # Applying the 'usub' unary operator (line 510)
    result___neg___171488 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 26), 'usub', i_171487)
    
    # Getting the type of 'c' (line 510)
    c_171489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___171490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 24), c_171489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_171491 = invoke(stypy.reporting.localization.Localization(__file__, 510, 24), getitem___171490, result___neg___171488)
    
    # Getting the type of 'xs' (line 510)
    xs_171492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 30), 'xs', False)
    # Applying the binary operator '*' (line 510)
    result_mul_171493 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 24), '*', subscript_call_result_171491, xs_171492)
    
    # Getting the type of 'c1' (line 510)
    c1_171494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 35), 'c1', False)
    # Getting the type of 'nd' (line 510)
    nd_171495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'nd', False)
    int_171496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 44), 'int')
    # Applying the binary operator '-' (line 510)
    result_sub_171497 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 39), '-', nd_171495, int_171496)
    
    # Applying the binary operator '*' (line 510)
    result_mul_171498 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 35), '*', c1_171494, result_sub_171497)
    
    # Getting the type of 'nd' (line 510)
    nd_171499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 48), 'nd', False)
    # Applying the binary operator 'div' (line 510)
    result_div_171500 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 34), 'div', result_mul_171498, nd_171499)
    
    # Processing the call keyword arguments (line 510)
    kwargs_171501 = {}
    # Getting the type of 'lagsub' (line 510)
    lagsub_171486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'lagsub', False)
    # Calling lagsub(args, kwargs) (line 510)
    lagsub_call_result_171502 = invoke(stypy.reporting.localization.Localization(__file__, 510, 17), lagsub_171486, *[result_mul_171493, result_div_171500], **kwargs_171501)
    
    # Assigning a type to the variable 'c0' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'c0', lagsub_call_result_171502)
    
    # Assigning a Call to a Name (line 511):
    
    # Assigning a Call to a Name (line 511):
    
    # Call to lagadd(...): (line 511)
    # Processing the call arguments (line 511)
    # Getting the type of 'tmp' (line 511)
    tmp_171504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 24), 'tmp', False)
    
    # Call to lagsub(...): (line 511)
    # Processing the call arguments (line 511)
    int_171506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 37), 'int')
    # Getting the type of 'nd' (line 511)
    nd_171507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 39), 'nd', False)
    # Applying the binary operator '*' (line 511)
    result_mul_171508 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 37), '*', int_171506, nd_171507)
    
    int_171509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 44), 'int')
    # Applying the binary operator '-' (line 511)
    result_sub_171510 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 37), '-', result_mul_171508, int_171509)
    
    # Getting the type of 'c1' (line 511)
    c1_171511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 47), 'c1', False)
    # Applying the binary operator '*' (line 511)
    result_mul_171512 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 36), '*', result_sub_171510, c1_171511)
    
    
    # Call to lagmulx(...): (line 511)
    # Processing the call arguments (line 511)
    # Getting the type of 'c1' (line 511)
    c1_171514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 59), 'c1', False)
    # Processing the call keyword arguments (line 511)
    kwargs_171515 = {}
    # Getting the type of 'lagmulx' (line 511)
    lagmulx_171513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 51), 'lagmulx', False)
    # Calling lagmulx(args, kwargs) (line 511)
    lagmulx_call_result_171516 = invoke(stypy.reporting.localization.Localization(__file__, 511, 51), lagmulx_171513, *[c1_171514], **kwargs_171515)
    
    # Processing the call keyword arguments (line 511)
    kwargs_171517 = {}
    # Getting the type of 'lagsub' (line 511)
    lagsub_171505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 29), 'lagsub', False)
    # Calling lagsub(args, kwargs) (line 511)
    lagsub_call_result_171518 = invoke(stypy.reporting.localization.Localization(__file__, 511, 29), lagsub_171505, *[result_mul_171512, lagmulx_call_result_171516], **kwargs_171517)
    
    # Getting the type of 'nd' (line 511)
    nd_171519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 64), 'nd', False)
    # Applying the binary operator 'div' (line 511)
    result_div_171520 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 29), 'div', lagsub_call_result_171518, nd_171519)
    
    # Processing the call keyword arguments (line 511)
    kwargs_171521 = {}
    # Getting the type of 'lagadd' (line 511)
    lagadd_171503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'lagadd', False)
    # Calling lagadd(args, kwargs) (line 511)
    lagadd_call_result_171522 = invoke(stypy.reporting.localization.Localization(__file__, 511, 17), lagadd_171503, *[tmp_171504, result_div_171520], **kwargs_171521)
    
    # Assigning a type to the variable 'c1' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'c1', lagadd_call_result_171522)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 500)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 497)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to lagadd(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'c0' (line 512)
    c0_171524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 18), 'c0', False)
    
    # Call to lagsub(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'c1' (line 512)
    c1_171526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 29), 'c1', False)
    
    # Call to lagmulx(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'c1' (line 512)
    c1_171528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 41), 'c1', False)
    # Processing the call keyword arguments (line 512)
    kwargs_171529 = {}
    # Getting the type of 'lagmulx' (line 512)
    lagmulx_171527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 33), 'lagmulx', False)
    # Calling lagmulx(args, kwargs) (line 512)
    lagmulx_call_result_171530 = invoke(stypy.reporting.localization.Localization(__file__, 512, 33), lagmulx_171527, *[c1_171528], **kwargs_171529)
    
    # Processing the call keyword arguments (line 512)
    kwargs_171531 = {}
    # Getting the type of 'lagsub' (line 512)
    lagsub_171525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 22), 'lagsub', False)
    # Calling lagsub(args, kwargs) (line 512)
    lagsub_call_result_171532 = invoke(stypy.reporting.localization.Localization(__file__, 512, 22), lagsub_171525, *[c1_171526, lagmulx_call_result_171530], **kwargs_171531)
    
    # Processing the call keyword arguments (line 512)
    kwargs_171533 = {}
    # Getting the type of 'lagadd' (line 512)
    lagadd_171523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'lagadd', False)
    # Calling lagadd(args, kwargs) (line 512)
    lagadd_call_result_171534 = invoke(stypy.reporting.localization.Localization(__file__, 512, 11), lagadd_171523, *[c0_171524, lagsub_call_result_171532], **kwargs_171533)
    
    # Assigning a type to the variable 'stypy_return_type' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type', lagadd_call_result_171534)
    
    # ################# End of 'lagmul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagmul' in the type store
    # Getting the type of 'stypy_return_type' (line 449)
    stypy_return_type_171535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171535)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagmul'
    return stypy_return_type_171535

# Assigning a type to the variable 'lagmul' (line 449)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), 'lagmul', lagmul)

@norecursion
def lagdiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagdiv'
    module_type_store = module_type_store.open_function_context('lagdiv', 515, 0, False)
    
    # Passed parameters checking function
    lagdiv.stypy_localization = localization
    lagdiv.stypy_type_of_self = None
    lagdiv.stypy_type_store = module_type_store
    lagdiv.stypy_function_name = 'lagdiv'
    lagdiv.stypy_param_names_list = ['c1', 'c2']
    lagdiv.stypy_varargs_param_name = None
    lagdiv.stypy_kwargs_param_name = None
    lagdiv.stypy_call_defaults = defaults
    lagdiv.stypy_call_varargs = varargs
    lagdiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagdiv', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagdiv', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagdiv(...)' code ##################

    str_171536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, (-1)), 'str', '\n    Divide one Laguerre series by another.\n\n    Returns the quotient-with-remainder of two Laguerre series\n    `c1` / `c2`.  The arguments are sequences of coefficients from lowest\n    order "term" to highest, e.g., [1,2,3] represents the series\n    ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    [quo, rem] : ndarrays\n        Of Laguerre series coefficients representing the quotient and\n        remainder.\n\n    See Also\n    --------\n    lagadd, lagsub, lagmul, lagpow\n\n    Notes\n    -----\n    In general, the (polynomial) division of one Laguerre series by another\n    results in quotient and remainder terms that are not in the Laguerre\n    polynomial basis set.  Thus, to express these results as a Laguerre\n    series, it is necessary to "reproject" the results onto the Laguerre\n    basis set, which may produce "unintuitive" (but correct) results; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagdiv\n    >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 0.]))\n    >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 1.,  1.]))\n\n    ')
    
    # Assigning a Call to a List (line 559):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 559)
    # Processing the call arguments (line 559)
    
    # Obtaining an instance of the builtin type 'list' (line 559)
    list_171539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 559)
    # Adding element type (line 559)
    # Getting the type of 'c1' (line 559)
    c1_171540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 28), list_171539, c1_171540)
    # Adding element type (line 559)
    # Getting the type of 'c2' (line 559)
    c2_171541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 28), list_171539, c2_171541)
    
    # Processing the call keyword arguments (line 559)
    kwargs_171542 = {}
    # Getting the type of 'pu' (line 559)
    pu_171537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 559)
    as_series_171538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 15), pu_171537, 'as_series')
    # Calling as_series(args, kwargs) (line 559)
    as_series_call_result_171543 = invoke(stypy.reporting.localization.Localization(__file__, 559, 15), as_series_171538, *[list_171539], **kwargs_171542)
    
    # Assigning a type to the variable 'call_assignment_170782' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170782', as_series_call_result_171543)
    
    # Assigning a Call to a Name (line 559):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171547 = {}
    # Getting the type of 'call_assignment_170782' (line 559)
    call_assignment_170782_171544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170782', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___171545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 4), call_assignment_170782_171544, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171548 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171545, *[int_171546], **kwargs_171547)
    
    # Assigning a type to the variable 'call_assignment_170783' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170783', getitem___call_result_171548)
    
    # Assigning a Name to a Name (line 559):
    # Getting the type of 'call_assignment_170783' (line 559)
    call_assignment_170783_171549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170783')
    # Assigning a type to the variable 'c1' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 5), 'c1', call_assignment_170783_171549)
    
    # Assigning a Call to a Name (line 559):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171553 = {}
    # Getting the type of 'call_assignment_170782' (line 559)
    call_assignment_170782_171550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170782', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___171551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 4), call_assignment_170782_171550, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171554 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171551, *[int_171552], **kwargs_171553)
    
    # Assigning a type to the variable 'call_assignment_170784' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170784', getitem___call_result_171554)
    
    # Assigning a Name to a Name (line 559):
    # Getting the type of 'call_assignment_170784' (line 559)
    call_assignment_170784_171555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_170784')
    # Assigning a type to the variable 'c2' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'c2', call_assignment_170784_171555)
    
    
    
    # Obtaining the type of the subscript
    int_171556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 10), 'int')
    # Getting the type of 'c2' (line 560)
    c2_171557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 7), 'c2')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___171558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 7), c2_171557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_171559 = invoke(stypy.reporting.localization.Localization(__file__, 560, 7), getitem___171558, int_171556)
    
    int_171560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 17), 'int')
    # Applying the binary operator '==' (line 560)
    result_eq_171561 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 7), '==', subscript_call_result_171559, int_171560)
    
    # Testing the type of an if condition (line 560)
    if_condition_171562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 4), result_eq_171561)
    # Assigning a type to the variable 'if_condition_171562' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'if_condition_171562', if_condition_171562)
    # SSA begins for if statement (line 560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ZeroDivisionError(...): (line 561)
    # Processing the call keyword arguments (line 561)
    kwargs_171564 = {}
    # Getting the type of 'ZeroDivisionError' (line 561)
    ZeroDivisionError_171563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 14), 'ZeroDivisionError', False)
    # Calling ZeroDivisionError(args, kwargs) (line 561)
    ZeroDivisionError_call_result_171565 = invoke(stypy.reporting.localization.Localization(__file__, 561, 14), ZeroDivisionError_171563, *[], **kwargs_171564)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 561, 8), ZeroDivisionError_call_result_171565, 'raise parameter', BaseException)
    # SSA join for if statement (line 560)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 563):
    
    # Assigning a Call to a Name (line 563):
    
    # Call to len(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'c1' (line 563)
    c1_171567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 14), 'c1', False)
    # Processing the call keyword arguments (line 563)
    kwargs_171568 = {}
    # Getting the type of 'len' (line 563)
    len_171566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 10), 'len', False)
    # Calling len(args, kwargs) (line 563)
    len_call_result_171569 = invoke(stypy.reporting.localization.Localization(__file__, 563, 10), len_171566, *[c1_171567], **kwargs_171568)
    
    # Assigning a type to the variable 'lc1' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'lc1', len_call_result_171569)
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to len(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'c2' (line 564)
    c2_171571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'c2', False)
    # Processing the call keyword arguments (line 564)
    kwargs_171572 = {}
    # Getting the type of 'len' (line 564)
    len_171570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 10), 'len', False)
    # Calling len(args, kwargs) (line 564)
    len_call_result_171573 = invoke(stypy.reporting.localization.Localization(__file__, 564, 10), len_171570, *[c2_171571], **kwargs_171572)
    
    # Assigning a type to the variable 'lc2' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'lc2', len_call_result_171573)
    
    
    # Getting the type of 'lc1' (line 565)
    lc1_171574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 7), 'lc1')
    # Getting the type of 'lc2' (line 565)
    lc2_171575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 13), 'lc2')
    # Applying the binary operator '<' (line 565)
    result_lt_171576 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 7), '<', lc1_171574, lc2_171575)
    
    # Testing the type of an if condition (line 565)
    if_condition_171577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 4), result_lt_171576)
    # Assigning a type to the variable 'if_condition_171577' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'if_condition_171577', if_condition_171577)
    # SSA begins for if statement (line 565)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 566)
    tuple_171578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 566)
    # Adding element type (line 566)
    
    # Obtaining the type of the subscript
    int_171579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 19), 'int')
    slice_171580 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 566, 15), None, int_171579, None)
    # Getting the type of 'c1' (line 566)
    c1_171581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___171582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 15), c1_171581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_171583 = invoke(stypy.reporting.localization.Localization(__file__, 566, 15), getitem___171582, slice_171580)
    
    int_171584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 22), 'int')
    # Applying the binary operator '*' (line 566)
    result_mul_171585 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 15), '*', subscript_call_result_171583, int_171584)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 15), tuple_171578, result_mul_171585)
    # Adding element type (line 566)
    # Getting the type of 'c1' (line 566)
    c1_171586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 25), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 15), tuple_171578, c1_171586)
    
    # Assigning a type to the variable 'stypy_return_type' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'stypy_return_type', tuple_171578)
    # SSA branch for the else part of an if statement (line 565)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lc2' (line 567)
    lc2_171587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 9), 'lc2')
    int_171588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 16), 'int')
    # Applying the binary operator '==' (line 567)
    result_eq_171589 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 9), '==', lc2_171587, int_171588)
    
    # Testing the type of an if condition (line 567)
    if_condition_171590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 9), result_eq_171589)
    # Assigning a type to the variable 'if_condition_171590' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 9), 'if_condition_171590', if_condition_171590)
    # SSA begins for if statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_171591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'c1' (line 568)
    c1_171592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'c1')
    
    # Obtaining the type of the subscript
    int_171593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 21), 'int')
    # Getting the type of 'c2' (line 568)
    c2_171594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 18), 'c2')
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___171595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 18), c2_171594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_171596 = invoke(stypy.reporting.localization.Localization(__file__, 568, 18), getitem___171595, int_171593)
    
    # Applying the binary operator 'div' (line 568)
    result_div_171597 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 15), 'div', c1_171592, subscript_call_result_171596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 15), tuple_171591, result_div_171597)
    # Adding element type (line 568)
    
    # Obtaining the type of the subscript
    int_171598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 30), 'int')
    slice_171599 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 568, 26), None, int_171598, None)
    # Getting the type of 'c1' (line 568)
    c1_171600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___171601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 26), c1_171600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_171602 = invoke(stypy.reporting.localization.Localization(__file__, 568, 26), getitem___171601, slice_171599)
    
    int_171603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 33), 'int')
    # Applying the binary operator '*' (line 568)
    result_mul_171604 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 26), '*', subscript_call_result_171602, int_171603)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 15), tuple_171591, result_mul_171604)
    
    # Assigning a type to the variable 'stypy_return_type' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'stypy_return_type', tuple_171591)
    # SSA branch for the else part of an if statement (line 567)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 570):
    
    # Assigning a Call to a Name (line 570):
    
    # Call to empty(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'lc1' (line 570)
    lc1_171607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 570)
    lc2_171608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 29), 'lc2', False)
    # Applying the binary operator '-' (line 570)
    result_sub_171609 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 23), '-', lc1_171607, lc2_171608)
    
    int_171610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 35), 'int')
    # Applying the binary operator '+' (line 570)
    result_add_171611 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 33), '+', result_sub_171609, int_171610)
    
    # Processing the call keyword arguments (line 570)
    # Getting the type of 'c1' (line 570)
    c1_171612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 44), 'c1', False)
    # Obtaining the member 'dtype' of a type (line 570)
    dtype_171613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 44), c1_171612, 'dtype')
    keyword_171614 = dtype_171613
    kwargs_171615 = {'dtype': keyword_171614}
    # Getting the type of 'np' (line 570)
    np_171605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 570)
    empty_171606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 14), np_171605, 'empty')
    # Calling empty(args, kwargs) (line 570)
    empty_call_result_171616 = invoke(stypy.reporting.localization.Localization(__file__, 570, 14), empty_171606, *[result_add_171611], **kwargs_171615)
    
    # Assigning a type to the variable 'quo' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'quo', empty_call_result_171616)
    
    # Assigning a Name to a Name (line 571):
    
    # Assigning a Name to a Name (line 571):
    # Getting the type of 'c1' (line 571)
    c1_171617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 14), 'c1')
    # Assigning a type to the variable 'rem' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'rem', c1_171617)
    
    
    # Call to range(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'lc1' (line 572)
    lc1_171619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 572)
    lc2_171620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 29), 'lc2', False)
    # Applying the binary operator '-' (line 572)
    result_sub_171621 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 23), '-', lc1_171619, lc2_171620)
    
    int_171622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 34), 'int')
    int_171623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 39), 'int')
    # Processing the call keyword arguments (line 572)
    kwargs_171624 = {}
    # Getting the type of 'range' (line 572)
    range_171618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 17), 'range', False)
    # Calling range(args, kwargs) (line 572)
    range_call_result_171625 = invoke(stypy.reporting.localization.Localization(__file__, 572, 17), range_171618, *[result_sub_171621, int_171622, int_171623], **kwargs_171624)
    
    # Testing the type of a for loop iterable (line 572)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 572, 8), range_call_result_171625)
    # Getting the type of the for loop variable (line 572)
    for_loop_var_171626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 572, 8), range_call_result_171625)
    # Assigning a type to the variable 'i' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'i', for_loop_var_171626)
    # SSA begins for a for statement (line 572)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to lagmul(...): (line 573)
    # Processing the call arguments (line 573)
    
    # Obtaining an instance of the builtin type 'list' (line 573)
    list_171628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 573)
    # Adding element type (line 573)
    int_171629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 23), list_171628, int_171629)
    
    # Getting the type of 'i' (line 573)
    i_171630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'i', False)
    # Applying the binary operator '*' (line 573)
    result_mul_171631 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 23), '*', list_171628, i_171630)
    
    
    # Obtaining an instance of the builtin type 'list' (line 573)
    list_171632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 573)
    # Adding element type (line 573)
    int_171633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 31), list_171632, int_171633)
    
    # Applying the binary operator '+' (line 573)
    result_add_171634 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 23), '+', result_mul_171631, list_171632)
    
    # Getting the type of 'c2' (line 573)
    c2_171635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 36), 'c2', False)
    # Processing the call keyword arguments (line 573)
    kwargs_171636 = {}
    # Getting the type of 'lagmul' (line 573)
    lagmul_171627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'lagmul', False)
    # Calling lagmul(args, kwargs) (line 573)
    lagmul_call_result_171637 = invoke(stypy.reporting.localization.Localization(__file__, 573, 16), lagmul_171627, *[result_add_171634, c2_171635], **kwargs_171636)
    
    # Assigning a type to the variable 'p' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'p', lagmul_call_result_171637)
    
    # Assigning a BinOp to a Name (line 574):
    
    # Assigning a BinOp to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_171638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 20), 'int')
    # Getting the type of 'rem' (line 574)
    rem_171639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'rem')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___171640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 16), rem_171639, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_171641 = invoke(stypy.reporting.localization.Localization(__file__, 574, 16), getitem___171640, int_171638)
    
    
    # Obtaining the type of the subscript
    int_171642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 26), 'int')
    # Getting the type of 'p' (line 574)
    p_171643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'p')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___171644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 24), p_171643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_171645 = invoke(stypy.reporting.localization.Localization(__file__, 574, 24), getitem___171644, int_171642)
    
    # Applying the binary operator 'div' (line 574)
    result_div_171646 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 16), 'div', subscript_call_result_171641, subscript_call_result_171645)
    
    # Assigning a type to the variable 'q' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'q', result_div_171646)
    
    # Assigning a BinOp to a Name (line 575):
    
    # Assigning a BinOp to a Name (line 575):
    
    # Obtaining the type of the subscript
    int_171647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 23), 'int')
    slice_171648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 18), None, int_171647, None)
    # Getting the type of 'rem' (line 575)
    rem_171649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 18), 'rem')
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___171650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 18), rem_171649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 575)
    subscript_call_result_171651 = invoke(stypy.reporting.localization.Localization(__file__, 575, 18), getitem___171650, slice_171648)
    
    # Getting the type of 'q' (line 575)
    q_171652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'q')
    
    # Obtaining the type of the subscript
    int_171653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 34), 'int')
    slice_171654 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 31), None, int_171653, None)
    # Getting the type of 'p' (line 575)
    p_171655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 31), 'p')
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___171656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 31), p_171655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 575)
    subscript_call_result_171657 = invoke(stypy.reporting.localization.Localization(__file__, 575, 31), getitem___171656, slice_171654)
    
    # Applying the binary operator '*' (line 575)
    result_mul_171658 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 29), '*', q_171652, subscript_call_result_171657)
    
    # Applying the binary operator '-' (line 575)
    result_sub_171659 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 18), '-', subscript_call_result_171651, result_mul_171658)
    
    # Assigning a type to the variable 'rem' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'rem', result_sub_171659)
    
    # Assigning a Name to a Subscript (line 576):
    
    # Assigning a Name to a Subscript (line 576):
    # Getting the type of 'q' (line 576)
    q_171660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 21), 'q')
    # Getting the type of 'quo' (line 576)
    quo_171661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'quo')
    # Getting the type of 'i' (line 576)
    i_171662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'i')
    # Storing an element on a container (line 576)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 12), quo_171661, (i_171662, q_171660))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 577)
    tuple_171663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 577)
    # Adding element type (line 577)
    # Getting the type of 'quo' (line 577)
    quo_171664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 15), 'quo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 15), tuple_171663, quo_171664)
    # Adding element type (line 577)
    
    # Call to trimseq(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'rem' (line 577)
    rem_171667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'rem', False)
    # Processing the call keyword arguments (line 577)
    kwargs_171668 = {}
    # Getting the type of 'pu' (line 577)
    pu_171665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 577)
    trimseq_171666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 20), pu_171665, 'trimseq')
    # Calling trimseq(args, kwargs) (line 577)
    trimseq_call_result_171669 = invoke(stypy.reporting.localization.Localization(__file__, 577, 20), trimseq_171666, *[rem_171667], **kwargs_171668)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 15), tuple_171663, trimseq_call_result_171669)
    
    # Assigning a type to the variable 'stypy_return_type' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'stypy_return_type', tuple_171663)
    # SSA join for if statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 565)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lagdiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagdiv' in the type store
    # Getting the type of 'stypy_return_type' (line 515)
    stypy_return_type_171670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171670)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagdiv'
    return stypy_return_type_171670

# Assigning a type to the variable 'lagdiv' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'lagdiv', lagdiv)

@norecursion
def lagpow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_171671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 28), 'int')
    defaults = [int_171671]
    # Create a new context for function 'lagpow'
    module_type_store = module_type_store.open_function_context('lagpow', 580, 0, False)
    
    # Passed parameters checking function
    lagpow.stypy_localization = localization
    lagpow.stypy_type_of_self = None
    lagpow.stypy_type_store = module_type_store
    lagpow.stypy_function_name = 'lagpow'
    lagpow.stypy_param_names_list = ['c', 'pow', 'maxpower']
    lagpow.stypy_varargs_param_name = None
    lagpow.stypy_kwargs_param_name = None
    lagpow.stypy_call_defaults = defaults
    lagpow.stypy_call_varargs = varargs
    lagpow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagpow', ['c', 'pow', 'maxpower'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagpow', localization, ['c', 'pow', 'maxpower'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagpow(...)' code ##################

    str_171672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, (-1)), 'str', 'Raise a Laguerre series to a power.\n\n    Returns the Laguerre series `c` raised to the power `pow`. The\n    argument `c` is a sequence of coefficients ordered from low to high.\n    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Laguerre series coefficients ordered from low to\n        high.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Laguerre series of power.\n\n    See Also\n    --------\n    lagadd, lagsub, lagmul, lagdiv\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagpow\n    >>> lagpow([1, 2, 3], 2)\n    array([ 14., -16.,  56., -72.,  54.])\n\n    ')
    
    # Assigning a Call to a List (line 615):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 615)
    # Processing the call arguments (line 615)
    
    # Obtaining an instance of the builtin type 'list' (line 615)
    list_171675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 615)
    # Adding element type (line 615)
    # Getting the type of 'c' (line 615)
    c_171676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 23), list_171675, c_171676)
    
    # Processing the call keyword arguments (line 615)
    kwargs_171677 = {}
    # Getting the type of 'pu' (line 615)
    pu_171673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 615)
    as_series_171674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 10), pu_171673, 'as_series')
    # Calling as_series(args, kwargs) (line 615)
    as_series_call_result_171678 = invoke(stypy.reporting.localization.Localization(__file__, 615, 10), as_series_171674, *[list_171675], **kwargs_171677)
    
    # Assigning a type to the variable 'call_assignment_170785' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_170785', as_series_call_result_171678)
    
    # Assigning a Call to a Name (line 615):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_171681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 4), 'int')
    # Processing the call keyword arguments
    kwargs_171682 = {}
    # Getting the type of 'call_assignment_170785' (line 615)
    call_assignment_170785_171679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_170785', False)
    # Obtaining the member '__getitem__' of a type (line 615)
    getitem___171680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 4), call_assignment_170785_171679, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_171683 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___171680, *[int_171681], **kwargs_171682)
    
    # Assigning a type to the variable 'call_assignment_170786' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_170786', getitem___call_result_171683)
    
    # Assigning a Name to a Name (line 615):
    # Getting the type of 'call_assignment_170786' (line 615)
    call_assignment_170786_171684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_170786')
    # Assigning a type to the variable 'c' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 5), 'c', call_assignment_170786_171684)
    
    # Assigning a Call to a Name (line 616):
    
    # Assigning a Call to a Name (line 616):
    
    # Call to int(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'pow' (line 616)
    pow_171686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'pow', False)
    # Processing the call keyword arguments (line 616)
    kwargs_171687 = {}
    # Getting the type of 'int' (line 616)
    int_171685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'int', False)
    # Calling int(args, kwargs) (line 616)
    int_call_result_171688 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), int_171685, *[pow_171686], **kwargs_171687)
    
    # Assigning a type to the variable 'power' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'power', int_call_result_171688)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'power' (line 617)
    power_171689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 7), 'power')
    # Getting the type of 'pow' (line 617)
    pow_171690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'pow')
    # Applying the binary operator '!=' (line 617)
    result_ne_171691 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 7), '!=', power_171689, pow_171690)
    
    
    # Getting the type of 'power' (line 617)
    power_171692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 23), 'power')
    int_171693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 31), 'int')
    # Applying the binary operator '<' (line 617)
    result_lt_171694 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 23), '<', power_171692, int_171693)
    
    # Applying the binary operator 'or' (line 617)
    result_or_keyword_171695 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 7), 'or', result_ne_171691, result_lt_171694)
    
    # Testing the type of an if condition (line 617)
    if_condition_171696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 4), result_or_keyword_171695)
    # Assigning a type to the variable 'if_condition_171696' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'if_condition_171696', if_condition_171696)
    # SSA begins for if statement (line 617)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 618)
    # Processing the call arguments (line 618)
    str_171698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 25), 'str', 'Power must be a non-negative integer.')
    # Processing the call keyword arguments (line 618)
    kwargs_171699 = {}
    # Getting the type of 'ValueError' (line 618)
    ValueError_171697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 618)
    ValueError_call_result_171700 = invoke(stypy.reporting.localization.Localization(__file__, 618, 14), ValueError_171697, *[str_171698], **kwargs_171699)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 618, 8), ValueError_call_result_171700, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 617)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxpower' (line 619)
    maxpower_171701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 9), 'maxpower')
    # Getting the type of 'None' (line 619)
    None_171702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 25), 'None')
    # Applying the binary operator 'isnot' (line 619)
    result_is_not_171703 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 9), 'isnot', maxpower_171701, None_171702)
    
    
    # Getting the type of 'power' (line 619)
    power_171704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 34), 'power')
    # Getting the type of 'maxpower' (line 619)
    maxpower_171705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 42), 'maxpower')
    # Applying the binary operator '>' (line 619)
    result_gt_171706 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 34), '>', power_171704, maxpower_171705)
    
    # Applying the binary operator 'and' (line 619)
    result_and_keyword_171707 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 9), 'and', result_is_not_171703, result_gt_171706)
    
    # Testing the type of an if condition (line 619)
    if_condition_171708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 9), result_and_keyword_171707)
    # Assigning a type to the variable 'if_condition_171708' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 9), 'if_condition_171708', if_condition_171708)
    # SSA begins for if statement (line 619)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 620)
    # Processing the call arguments (line 620)
    str_171710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 25), 'str', 'Power is too large')
    # Processing the call keyword arguments (line 620)
    kwargs_171711 = {}
    # Getting the type of 'ValueError' (line 620)
    ValueError_171709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 620)
    ValueError_call_result_171712 = invoke(stypy.reporting.localization.Localization(__file__, 620, 14), ValueError_171709, *[str_171710], **kwargs_171711)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 620, 8), ValueError_call_result_171712, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 619)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 621)
    power_171713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 9), 'power')
    int_171714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 18), 'int')
    # Applying the binary operator '==' (line 621)
    result_eq_171715 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 9), '==', power_171713, int_171714)
    
    # Testing the type of an if condition (line 621)
    if_condition_171716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 9), result_eq_171715)
    # Assigning a type to the variable 'if_condition_171716' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 9), 'if_condition_171716', if_condition_171716)
    # SSA begins for if statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 622)
    # Processing the call arguments (line 622)
    
    # Obtaining an instance of the builtin type 'list' (line 622)
    list_171719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 622)
    # Adding element type (line 622)
    int_171720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 24), list_171719, int_171720)
    
    # Processing the call keyword arguments (line 622)
    # Getting the type of 'c' (line 622)
    c_171721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 35), 'c', False)
    # Obtaining the member 'dtype' of a type (line 622)
    dtype_171722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 35), c_171721, 'dtype')
    keyword_171723 = dtype_171722
    kwargs_171724 = {'dtype': keyword_171723}
    # Getting the type of 'np' (line 622)
    np_171717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 622)
    array_171718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 15), np_171717, 'array')
    # Calling array(args, kwargs) (line 622)
    array_call_result_171725 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), array_171718, *[list_171719], **kwargs_171724)
    
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'stypy_return_type', array_call_result_171725)
    # SSA branch for the else part of an if statement (line 621)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 623)
    power_171726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 9), 'power')
    int_171727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 18), 'int')
    # Applying the binary operator '==' (line 623)
    result_eq_171728 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 9), '==', power_171726, int_171727)
    
    # Testing the type of an if condition (line 623)
    if_condition_171729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 9), result_eq_171728)
    # Assigning a type to the variable 'if_condition_171729' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 9), 'if_condition_171729', if_condition_171729)
    # SSA begins for if statement (line 623)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 624)
    c_171730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'stypy_return_type', c_171730)
    # SSA branch for the else part of an if statement (line 623)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 628):
    
    # Assigning a Name to a Name (line 628):
    # Getting the type of 'c' (line 628)
    c_171731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 14), 'c')
    # Assigning a type to the variable 'prd' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'prd', c_171731)
    
    
    # Call to range(...): (line 629)
    # Processing the call arguments (line 629)
    int_171733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 23), 'int')
    # Getting the type of 'power' (line 629)
    power_171734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 26), 'power', False)
    int_171735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 34), 'int')
    # Applying the binary operator '+' (line 629)
    result_add_171736 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 26), '+', power_171734, int_171735)
    
    # Processing the call keyword arguments (line 629)
    kwargs_171737 = {}
    # Getting the type of 'range' (line 629)
    range_171732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 17), 'range', False)
    # Calling range(args, kwargs) (line 629)
    range_call_result_171738 = invoke(stypy.reporting.localization.Localization(__file__, 629, 17), range_171732, *[int_171733, result_add_171736], **kwargs_171737)
    
    # Testing the type of a for loop iterable (line 629)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 629, 8), range_call_result_171738)
    # Getting the type of the for loop variable (line 629)
    for_loop_var_171739 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 629, 8), range_call_result_171738)
    # Assigning a type to the variable 'i' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'i', for_loop_var_171739)
    # SSA begins for a for statement (line 629)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 630):
    
    # Assigning a Call to a Name (line 630):
    
    # Call to lagmul(...): (line 630)
    # Processing the call arguments (line 630)
    # Getting the type of 'prd' (line 630)
    prd_171741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 25), 'prd', False)
    # Getting the type of 'c' (line 630)
    c_171742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 30), 'c', False)
    # Processing the call keyword arguments (line 630)
    kwargs_171743 = {}
    # Getting the type of 'lagmul' (line 630)
    lagmul_171740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 18), 'lagmul', False)
    # Calling lagmul(args, kwargs) (line 630)
    lagmul_call_result_171744 = invoke(stypy.reporting.localization.Localization(__file__, 630, 18), lagmul_171740, *[prd_171741, c_171742], **kwargs_171743)
    
    # Assigning a type to the variable 'prd' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'prd', lagmul_call_result_171744)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 631)
    prd_171745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 15), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'stypy_return_type', prd_171745)
    # SSA join for if statement (line 623)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 621)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 619)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 617)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lagpow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagpow' in the type store
    # Getting the type of 'stypy_return_type' (line 580)
    stypy_return_type_171746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171746)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagpow'
    return stypy_return_type_171746

# Assigning a type to the variable 'lagpow' (line 580)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 0), 'lagpow', lagpow)

@norecursion
def lagder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_171747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 16), 'int')
    int_171748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 23), 'int')
    int_171749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 31), 'int')
    defaults = [int_171747, int_171748, int_171749]
    # Create a new context for function 'lagder'
    module_type_store = module_type_store.open_function_context('lagder', 634, 0, False)
    
    # Passed parameters checking function
    lagder.stypy_localization = localization
    lagder.stypy_type_of_self = None
    lagder.stypy_type_store = module_type_store
    lagder.stypy_function_name = 'lagder'
    lagder.stypy_param_names_list = ['c', 'm', 'scl', 'axis']
    lagder.stypy_varargs_param_name = None
    lagder.stypy_kwargs_param_name = None
    lagder.stypy_call_defaults = defaults
    lagder.stypy_call_varargs = varargs
    lagder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagder', ['c', 'm', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagder', localization, ['c', 'm', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagder(...)' code ##################

    str_171750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, (-1)), 'str', '\n    Differentiate a Laguerre series.\n\n    Returns the Laguerre series coefficients `c` differentiated `m` times\n    along `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable). The argument\n    `c` is an array of coefficients from low to high degree along each\n    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``\n    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +\n    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is\n    ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Laguerre series coefficients. If `c` is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change of\n        variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Laguerre series of the derivative.\n\n    See Also\n    --------\n    lagint\n\n    Notes\n    -----\n    In general, the result of differentiating a Laguerre series does not\n    resemble the same operation on a power series. Thus the result of this\n    function may be "unintuitive," albeit correct; see Examples section\n    below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagder\n    >>> lagder([ 1.,  1.,  1., -3.])\n    array([ 1.,  2.,  3.])\n    >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)\n    array([ 1.,  2.,  3.])\n\n    ')
    
    # Assigning a Call to a Name (line 689):
    
    # Assigning a Call to a Name (line 689):
    
    # Call to array(...): (line 689)
    # Processing the call arguments (line 689)
    # Getting the type of 'c' (line 689)
    c_171753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 17), 'c', False)
    # Processing the call keyword arguments (line 689)
    int_171754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 26), 'int')
    keyword_171755 = int_171754
    int_171756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 34), 'int')
    keyword_171757 = int_171756
    kwargs_171758 = {'copy': keyword_171757, 'ndmin': keyword_171755}
    # Getting the type of 'np' (line 689)
    np_171751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 689)
    array_171752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), np_171751, 'array')
    # Calling array(args, kwargs) (line 689)
    array_call_result_171759 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), array_171752, *[c_171753], **kwargs_171758)
    
    # Assigning a type to the variable 'c' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'c', array_call_result_171759)
    
    
    # Getting the type of 'c' (line 690)
    c_171760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 690)
    dtype_171761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 7), c_171760, 'dtype')
    # Obtaining the member 'char' of a type (line 690)
    char_171762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 7), dtype_171761, 'char')
    str_171763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 690)
    result_contains_171764 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 7), 'in', char_171762, str_171763)
    
    # Testing the type of an if condition (line 690)
    if_condition_171765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 4), result_contains_171764)
    # Assigning a type to the variable 'if_condition_171765' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'if_condition_171765', if_condition_171765)
    # SSA begins for if statement (line 690)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 691):
    
    # Assigning a Call to a Name (line 691):
    
    # Call to astype(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'np' (line 691)
    np_171768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 691)
    double_171769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 21), np_171768, 'double')
    # Processing the call keyword arguments (line 691)
    kwargs_171770 = {}
    # Getting the type of 'c' (line 691)
    c_171766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 691)
    astype_171767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 12), c_171766, 'astype')
    # Calling astype(args, kwargs) (line 691)
    astype_call_result_171771 = invoke(stypy.reporting.localization.Localization(__file__, 691, 12), astype_171767, *[double_171769], **kwargs_171770)
    
    # Assigning a type to the variable 'c' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'c', astype_call_result_171771)
    # SSA join for if statement (line 690)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 692):
    
    # Assigning a Subscript to a Name (line 692):
    
    # Obtaining the type of the subscript
    int_171772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 692)
    list_171777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 692)
    # Adding element type (line 692)
    # Getting the type of 'm' (line 692)
    m_171778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), list_171777, m_171778)
    # Adding element type (line 692)
    # Getting the type of 'axis' (line 692)
    axis_171779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), list_171777, axis_171779)
    
    comprehension_171780 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 18), list_171777)
    # Assigning a type to the variable 't' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 't', comprehension_171780)
    
    # Call to int(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 't' (line 692)
    t_171774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 22), 't', False)
    # Processing the call keyword arguments (line 692)
    kwargs_171775 = {}
    # Getting the type of 'int' (line 692)
    int_171773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 'int', False)
    # Calling int(args, kwargs) (line 692)
    int_call_result_171776 = invoke(stypy.reporting.localization.Localization(__file__, 692, 18), int_171773, *[t_171774], **kwargs_171775)
    
    list_171781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 18), list_171781, int_call_result_171776)
    # Obtaining the member '__getitem__' of a type (line 692)
    getitem___171782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 4), list_171781, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 692)
    subscript_call_result_171783 = invoke(stypy.reporting.localization.Localization(__file__, 692, 4), getitem___171782, int_171772)
    
    # Assigning a type to the variable 'tuple_var_assignment_170787' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'tuple_var_assignment_170787', subscript_call_result_171783)
    
    # Assigning a Subscript to a Name (line 692):
    
    # Obtaining the type of the subscript
    int_171784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 692)
    list_171789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 692)
    # Adding element type (line 692)
    # Getting the type of 'm' (line 692)
    m_171790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), list_171789, m_171790)
    # Adding element type (line 692)
    # Getting the type of 'axis' (line 692)
    axis_171791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), list_171789, axis_171791)
    
    comprehension_171792 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 18), list_171789)
    # Assigning a type to the variable 't' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 't', comprehension_171792)
    
    # Call to int(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 't' (line 692)
    t_171786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 22), 't', False)
    # Processing the call keyword arguments (line 692)
    kwargs_171787 = {}
    # Getting the type of 'int' (line 692)
    int_171785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 'int', False)
    # Calling int(args, kwargs) (line 692)
    int_call_result_171788 = invoke(stypy.reporting.localization.Localization(__file__, 692, 18), int_171785, *[t_171786], **kwargs_171787)
    
    list_171793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 18), list_171793, int_call_result_171788)
    # Obtaining the member '__getitem__' of a type (line 692)
    getitem___171794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 4), list_171793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 692)
    subscript_call_result_171795 = invoke(stypy.reporting.localization.Localization(__file__, 692, 4), getitem___171794, int_171784)
    
    # Assigning a type to the variable 'tuple_var_assignment_170788' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'tuple_var_assignment_170788', subscript_call_result_171795)
    
    # Assigning a Name to a Name (line 692):
    # Getting the type of 'tuple_var_assignment_170787' (line 692)
    tuple_var_assignment_170787_171796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'tuple_var_assignment_170787')
    # Assigning a type to the variable 'cnt' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'cnt', tuple_var_assignment_170787_171796)
    
    # Assigning a Name to a Name (line 692):
    # Getting the type of 'tuple_var_assignment_170788' (line 692)
    tuple_var_assignment_170788_171797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'tuple_var_assignment_170788')
    # Assigning a type to the variable 'iaxis' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 9), 'iaxis', tuple_var_assignment_170788_171797)
    
    
    # Getting the type of 'cnt' (line 694)
    cnt_171798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 7), 'cnt')
    # Getting the type of 'm' (line 694)
    m_171799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 14), 'm')
    # Applying the binary operator '!=' (line 694)
    result_ne_171800 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 7), '!=', cnt_171798, m_171799)
    
    # Testing the type of an if condition (line 694)
    if_condition_171801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 4), result_ne_171800)
    # Assigning a type to the variable 'if_condition_171801' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'if_condition_171801', if_condition_171801)
    # SSA begins for if statement (line 694)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 695)
    # Processing the call arguments (line 695)
    str_171803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 25), 'str', 'The order of derivation must be integer')
    # Processing the call keyword arguments (line 695)
    kwargs_171804 = {}
    # Getting the type of 'ValueError' (line 695)
    ValueError_171802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 695)
    ValueError_call_result_171805 = invoke(stypy.reporting.localization.Localization(__file__, 695, 14), ValueError_171802, *[str_171803], **kwargs_171804)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 695, 8), ValueError_call_result_171805, 'raise parameter', BaseException)
    # SSA join for if statement (line 694)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 696)
    cnt_171806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 7), 'cnt')
    int_171807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 13), 'int')
    # Applying the binary operator '<' (line 696)
    result_lt_171808 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 7), '<', cnt_171806, int_171807)
    
    # Testing the type of an if condition (line 696)
    if_condition_171809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 4), result_lt_171808)
    # Assigning a type to the variable 'if_condition_171809' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'if_condition_171809', if_condition_171809)
    # SSA begins for if statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 697)
    # Processing the call arguments (line 697)
    str_171811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 25), 'str', 'The order of derivation must be non-negative')
    # Processing the call keyword arguments (line 697)
    kwargs_171812 = {}
    # Getting the type of 'ValueError' (line 697)
    ValueError_171810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 697)
    ValueError_call_result_171813 = invoke(stypy.reporting.localization.Localization(__file__, 697, 14), ValueError_171810, *[str_171811], **kwargs_171812)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 697, 8), ValueError_call_result_171813, 'raise parameter', BaseException)
    # SSA join for if statement (line 696)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 698)
    iaxis_171814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 7), 'iaxis')
    # Getting the type of 'axis' (line 698)
    axis_171815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'axis')
    # Applying the binary operator '!=' (line 698)
    result_ne_171816 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 7), '!=', iaxis_171814, axis_171815)
    
    # Testing the type of an if condition (line 698)
    if_condition_171817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 698, 4), result_ne_171816)
    # Assigning a type to the variable 'if_condition_171817' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'if_condition_171817', if_condition_171817)
    # SSA begins for if statement (line 698)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 699)
    # Processing the call arguments (line 699)
    str_171819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 699)
    kwargs_171820 = {}
    # Getting the type of 'ValueError' (line 699)
    ValueError_171818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 699)
    ValueError_call_result_171821 = invoke(stypy.reporting.localization.Localization(__file__, 699, 14), ValueError_171818, *[str_171819], **kwargs_171820)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 699, 8), ValueError_call_result_171821, 'raise parameter', BaseException)
    # SSA join for if statement (line 698)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 700)
    c_171822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 700)
    ndim_171823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 12), c_171822, 'ndim')
    # Applying the 'usub' unary operator (line 700)
    result___neg___171824 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 11), 'usub', ndim_171823)
    
    # Getting the type of 'iaxis' (line 700)
    iaxis_171825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 22), 'iaxis')
    # Applying the binary operator '<=' (line 700)
    result_le_171826 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 11), '<=', result___neg___171824, iaxis_171825)
    # Getting the type of 'c' (line 700)
    c_171827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 700)
    ndim_171828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 30), c_171827, 'ndim')
    # Applying the binary operator '<' (line 700)
    result_lt_171829 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 11), '<', iaxis_171825, ndim_171828)
    # Applying the binary operator '&' (line 700)
    result_and__171830 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 11), '&', result_le_171826, result_lt_171829)
    
    # Applying the 'not' unary operator (line 700)
    result_not__171831 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 7), 'not', result_and__171830)
    
    # Testing the type of an if condition (line 700)
    if_condition_171832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 4), result_not__171831)
    # Assigning a type to the variable 'if_condition_171832' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'if_condition_171832', if_condition_171832)
    # SSA begins for if statement (line 700)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 701)
    # Processing the call arguments (line 701)
    str_171834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 701)
    kwargs_171835 = {}
    # Getting the type of 'ValueError' (line 701)
    ValueError_171833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 701)
    ValueError_call_result_171836 = invoke(stypy.reporting.localization.Localization(__file__, 701, 14), ValueError_171833, *[str_171834], **kwargs_171835)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 701, 8), ValueError_call_result_171836, 'raise parameter', BaseException)
    # SSA join for if statement (line 700)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 702)
    iaxis_171837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 7), 'iaxis')
    int_171838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 15), 'int')
    # Applying the binary operator '<' (line 702)
    result_lt_171839 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 7), '<', iaxis_171837, int_171838)
    
    # Testing the type of an if condition (line 702)
    if_condition_171840 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 702, 4), result_lt_171839)
    # Assigning a type to the variable 'if_condition_171840' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'if_condition_171840', if_condition_171840)
    # SSA begins for if statement (line 702)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 703)
    iaxis_171841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'iaxis')
    # Getting the type of 'c' (line 703)
    c_171842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 703)
    ndim_171843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 17), c_171842, 'ndim')
    # Applying the binary operator '+=' (line 703)
    result_iadd_171844 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 8), '+=', iaxis_171841, ndim_171843)
    # Assigning a type to the variable 'iaxis' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'iaxis', result_iadd_171844)
    
    # SSA join for if statement (line 702)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 705)
    cnt_171845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 7), 'cnt')
    int_171846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 14), 'int')
    # Applying the binary operator '==' (line 705)
    result_eq_171847 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 7), '==', cnt_171845, int_171846)
    
    # Testing the type of an if condition (line 705)
    if_condition_171848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 4), result_eq_171847)
    # Assigning a type to the variable 'if_condition_171848' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'if_condition_171848', if_condition_171848)
    # SSA begins for if statement (line 705)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 706)
    c_171849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'stypy_return_type', c_171849)
    # SSA join for if statement (line 705)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 708):
    
    # Assigning a Call to a Name (line 708):
    
    # Call to rollaxis(...): (line 708)
    # Processing the call arguments (line 708)
    # Getting the type of 'c' (line 708)
    c_171852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 20), 'c', False)
    # Getting the type of 'iaxis' (line 708)
    iaxis_171853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 708)
    kwargs_171854 = {}
    # Getting the type of 'np' (line 708)
    np_171850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 708)
    rollaxis_171851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), np_171850, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 708)
    rollaxis_call_result_171855 = invoke(stypy.reporting.localization.Localization(__file__, 708, 8), rollaxis_171851, *[c_171852, iaxis_171853], **kwargs_171854)
    
    # Assigning a type to the variable 'c' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'c', rollaxis_call_result_171855)
    
    # Assigning a Call to a Name (line 709):
    
    # Assigning a Call to a Name (line 709):
    
    # Call to len(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'c' (line 709)
    c_171857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 12), 'c', False)
    # Processing the call keyword arguments (line 709)
    kwargs_171858 = {}
    # Getting the type of 'len' (line 709)
    len_171856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'len', False)
    # Calling len(args, kwargs) (line 709)
    len_call_result_171859 = invoke(stypy.reporting.localization.Localization(__file__, 709, 8), len_171856, *[c_171857], **kwargs_171858)
    
    # Assigning a type to the variable 'n' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'n', len_call_result_171859)
    
    
    # Getting the type of 'cnt' (line 710)
    cnt_171860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 7), 'cnt')
    # Getting the type of 'n' (line 710)
    n_171861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 14), 'n')
    # Applying the binary operator '>=' (line 710)
    result_ge_171862 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 7), '>=', cnt_171860, n_171861)
    
    # Testing the type of an if condition (line 710)
    if_condition_171863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 710, 4), result_ge_171862)
    # Assigning a type to the variable 'if_condition_171863' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'if_condition_171863', if_condition_171863)
    # SSA begins for if statement (line 710)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 711):
    
    # Assigning a BinOp to a Name (line 711):
    
    # Obtaining the type of the subscript
    int_171864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 15), 'int')
    slice_171865 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 711, 12), None, int_171864, None)
    # Getting the type of 'c' (line 711)
    c_171866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 711)
    getitem___171867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 12), c_171866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 711)
    subscript_call_result_171868 = invoke(stypy.reporting.localization.Localization(__file__, 711, 12), getitem___171867, slice_171865)
    
    int_171869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 18), 'int')
    # Applying the binary operator '*' (line 711)
    result_mul_171870 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 12), '*', subscript_call_result_171868, int_171869)
    
    # Assigning a type to the variable 'c' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'c', result_mul_171870)
    # SSA branch for the else part of an if statement (line 710)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 713)
    # Processing the call arguments (line 713)
    # Getting the type of 'cnt' (line 713)
    cnt_171872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'cnt', False)
    # Processing the call keyword arguments (line 713)
    kwargs_171873 = {}
    # Getting the type of 'range' (line 713)
    range_171871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 17), 'range', False)
    # Calling range(args, kwargs) (line 713)
    range_call_result_171874 = invoke(stypy.reporting.localization.Localization(__file__, 713, 17), range_171871, *[cnt_171872], **kwargs_171873)
    
    # Testing the type of a for loop iterable (line 713)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 713, 8), range_call_result_171874)
    # Getting the type of the for loop variable (line 713)
    for_loop_var_171875 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 713, 8), range_call_result_171874)
    # Assigning a type to the variable 'i' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'i', for_loop_var_171875)
    # SSA begins for a for statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 714):
    
    # Assigning a BinOp to a Name (line 714):
    # Getting the type of 'n' (line 714)
    n_171876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 16), 'n')
    int_171877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 20), 'int')
    # Applying the binary operator '-' (line 714)
    result_sub_171878 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 16), '-', n_171876, int_171877)
    
    # Assigning a type to the variable 'n' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 12), 'n', result_sub_171878)
    
    # Getting the type of 'c' (line 715)
    c_171879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 12), 'c')
    # Getting the type of 'scl' (line 715)
    scl_171880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 17), 'scl')
    # Applying the binary operator '*=' (line 715)
    result_imul_171881 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 12), '*=', c_171879, scl_171880)
    # Assigning a type to the variable 'c' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 12), 'c', result_imul_171881)
    
    
    # Assigning a Call to a Name (line 716):
    
    # Assigning a Call to a Name (line 716):
    
    # Call to empty(...): (line 716)
    # Processing the call arguments (line 716)
    
    # Obtaining an instance of the builtin type 'tuple' (line 716)
    tuple_171884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 716)
    # Adding element type (line 716)
    # Getting the type of 'n' (line 716)
    n_171885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 28), tuple_171884, n_171885)
    
    
    # Obtaining the type of the subscript
    int_171886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 42), 'int')
    slice_171887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 716, 34), int_171886, None, None)
    # Getting the type of 'c' (line 716)
    c_171888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 716)
    shape_171889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 34), c_171888, 'shape')
    # Obtaining the member '__getitem__' of a type (line 716)
    getitem___171890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 34), shape_171889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 716)
    subscript_call_result_171891 = invoke(stypy.reporting.localization.Localization(__file__, 716, 34), getitem___171890, slice_171887)
    
    # Applying the binary operator '+' (line 716)
    result_add_171892 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 27), '+', tuple_171884, subscript_call_result_171891)
    
    # Processing the call keyword arguments (line 716)
    # Getting the type of 'c' (line 716)
    c_171893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 716)
    dtype_171894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 53), c_171893, 'dtype')
    keyword_171895 = dtype_171894
    kwargs_171896 = {'dtype': keyword_171895}
    # Getting the type of 'np' (line 716)
    np_171882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 716)
    empty_171883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 18), np_171882, 'empty')
    # Calling empty(args, kwargs) (line 716)
    empty_call_result_171897 = invoke(stypy.reporting.localization.Localization(__file__, 716, 18), empty_171883, *[result_add_171892], **kwargs_171896)
    
    # Assigning a type to the variable 'der' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'der', empty_call_result_171897)
    
    
    # Call to range(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'n' (line 717)
    n_171899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 27), 'n', False)
    int_171900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 30), 'int')
    int_171901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 33), 'int')
    # Processing the call keyword arguments (line 717)
    kwargs_171902 = {}
    # Getting the type of 'range' (line 717)
    range_171898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 21), 'range', False)
    # Calling range(args, kwargs) (line 717)
    range_call_result_171903 = invoke(stypy.reporting.localization.Localization(__file__, 717, 21), range_171898, *[n_171899, int_171900, int_171901], **kwargs_171902)
    
    # Testing the type of a for loop iterable (line 717)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 717, 12), range_call_result_171903)
    # Getting the type of the for loop variable (line 717)
    for_loop_var_171904 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 717, 12), range_call_result_171903)
    # Assigning a type to the variable 'j' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 12), 'j', for_loop_var_171904)
    # SSA begins for a for statement (line 717)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 718):
    
    # Assigning a UnaryOp to a Subscript (line 718):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 718)
    j_171905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 32), 'j')
    # Getting the type of 'c' (line 718)
    c_171906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 30), 'c')
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___171907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 30), c_171906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_171908 = invoke(stypy.reporting.localization.Localization(__file__, 718, 30), getitem___171907, j_171905)
    
    # Applying the 'usub' unary operator (line 718)
    result___neg___171909 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 29), 'usub', subscript_call_result_171908)
    
    # Getting the type of 'der' (line 718)
    der_171910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 16), 'der')
    # Getting the type of 'j' (line 718)
    j_171911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 20), 'j')
    int_171912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 24), 'int')
    # Applying the binary operator '-' (line 718)
    result_sub_171913 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 20), '-', j_171911, int_171912)
    
    # Storing an element on a container (line 718)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 16), der_171910, (result_sub_171913, result___neg___171909))
    
    # Getting the type of 'c' (line 719)
    c_171914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'c')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 719)
    j_171915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 18), 'j')
    int_171916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 22), 'int')
    # Applying the binary operator '-' (line 719)
    result_sub_171917 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 18), '-', j_171915, int_171916)
    
    # Getting the type of 'c' (line 719)
    c_171918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'c')
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___171919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 16), c_171918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_171920 = invoke(stypy.reporting.localization.Localization(__file__, 719, 16), getitem___171919, result_sub_171917)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 719)
    j_171921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 30), 'j')
    # Getting the type of 'c' (line 719)
    c_171922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 28), 'c')
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___171923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 28), c_171922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_171924 = invoke(stypy.reporting.localization.Localization(__file__, 719, 28), getitem___171923, j_171921)
    
    # Applying the binary operator '+=' (line 719)
    result_iadd_171925 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 16), '+=', subscript_call_result_171920, subscript_call_result_171924)
    # Getting the type of 'c' (line 719)
    c_171926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'c')
    # Getting the type of 'j' (line 719)
    j_171927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 18), 'j')
    int_171928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 22), 'int')
    # Applying the binary operator '-' (line 719)
    result_sub_171929 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 18), '-', j_171927, int_171928)
    
    # Storing an element on a container (line 719)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 16), c_171926, (result_sub_171929, result_iadd_171925))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Subscript (line 720):
    
    # Assigning a UnaryOp to a Subscript (line 720):
    
    
    # Obtaining the type of the subscript
    int_171930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 24), 'int')
    # Getting the type of 'c' (line 720)
    c_171931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 720)
    getitem___171932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 22), c_171931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 720)
    subscript_call_result_171933 = invoke(stypy.reporting.localization.Localization(__file__, 720, 22), getitem___171932, int_171930)
    
    # Applying the 'usub' unary operator (line 720)
    result___neg___171934 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 21), 'usub', subscript_call_result_171933)
    
    # Getting the type of 'der' (line 720)
    der_171935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'der')
    int_171936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 16), 'int')
    # Storing an element on a container (line 720)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 12), der_171935, (int_171936, result___neg___171934))
    
    # Assigning a Name to a Name (line 721):
    
    # Assigning a Name to a Name (line 721):
    # Getting the type of 'der' (line 721)
    der_171937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 16), 'der')
    # Assigning a type to the variable 'c' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'c', der_171937)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 710)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 722):
    
    # Assigning a Call to a Name (line 722):
    
    # Call to rollaxis(...): (line 722)
    # Processing the call arguments (line 722)
    # Getting the type of 'c' (line 722)
    c_171940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 20), 'c', False)
    int_171941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 23), 'int')
    # Getting the type of 'iaxis' (line 722)
    iaxis_171942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 26), 'iaxis', False)
    int_171943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 34), 'int')
    # Applying the binary operator '+' (line 722)
    result_add_171944 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 26), '+', iaxis_171942, int_171943)
    
    # Processing the call keyword arguments (line 722)
    kwargs_171945 = {}
    # Getting the type of 'np' (line 722)
    np_171938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 722)
    rollaxis_171939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 8), np_171938, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 722)
    rollaxis_call_result_171946 = invoke(stypy.reporting.localization.Localization(__file__, 722, 8), rollaxis_171939, *[c_171940, int_171941, result_add_171944], **kwargs_171945)
    
    # Assigning a type to the variable 'c' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'c', rollaxis_call_result_171946)
    # Getting the type of 'c' (line 723)
    c_171947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'stypy_return_type', c_171947)
    
    # ################# End of 'lagder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagder' in the type store
    # Getting the type of 'stypy_return_type' (line 634)
    stypy_return_type_171948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171948)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagder'
    return stypy_return_type_171948

# Assigning a type to the variable 'lagder' (line 634)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 0), 'lagder', lagder)

@norecursion
def lagint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_171949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 16), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 726)
    list_171950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 726)
    
    int_171951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 30), 'int')
    int_171952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 37), 'int')
    int_171953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 45), 'int')
    defaults = [int_171949, list_171950, int_171951, int_171952, int_171953]
    # Create a new context for function 'lagint'
    module_type_store = module_type_store.open_function_context('lagint', 726, 0, False)
    
    # Passed parameters checking function
    lagint.stypy_localization = localization
    lagint.stypy_type_of_self = None
    lagint.stypy_type_store = module_type_store
    lagint.stypy_function_name = 'lagint'
    lagint.stypy_param_names_list = ['c', 'm', 'k', 'lbnd', 'scl', 'axis']
    lagint.stypy_varargs_param_name = None
    lagint.stypy_kwargs_param_name = None
    lagint.stypy_call_defaults = defaults
    lagint.stypy_call_varargs = varargs
    lagint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagint', ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagint', localization, ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagint(...)' code ##################

    str_171954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, (-1)), 'str', '\n    Integrate a Laguerre series.\n\n    Returns the Laguerre series coefficients `c` integrated `m` times from\n    `lbnd` along `axis`. At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.)  The argument `c` is an array of\n    coefficients from low to high degree along each axis, e.g., [1,2,3]\n    represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]\n    represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +\n    2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.\n\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Laguerre series coefficients. If `c` is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at\n        ``lbnd`` is the first value in the list, the value of the second\n        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the\n        default), all constants are set to zero.  If ``m == 1``, a single\n        scalar can be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        Laguerre series coefficients of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or\n        ``np.isscalar(scl) == False``.\n\n    See Also\n    --------\n    lagder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.\n    Why is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`.  Then\n    .. math::`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a` - perhaps not what one would have first thought.\n\n    Also note that, in general, the result of integrating a C-series needs\n    to be "reprojected" onto the C-series basis set.  Thus, typically,\n    the result of this function is "unintuitive," albeit correct; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagint\n    >>> lagint([1,2,3])\n    array([ 1.,  1.,  1., -3.])\n    >>> lagint([1,2,3], m=2)\n    array([ 1.,  0.,  0., -4.,  3.])\n    >>> lagint([1,2,3], k=1)\n    array([ 2.,  1.,  1., -3.])\n    >>> lagint([1,2,3], lbnd=-1)\n    array([ 11.5,   1. ,   1. ,  -3. ])\n    >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)\n    array([ 11.16666667,  -5.        ,  -3.        ,   2.        ])\n\n    ')
    
    # Assigning a Call to a Name (line 810):
    
    # Assigning a Call to a Name (line 810):
    
    # Call to array(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'c' (line 810)
    c_171957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 17), 'c', False)
    # Processing the call keyword arguments (line 810)
    int_171958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 26), 'int')
    keyword_171959 = int_171958
    int_171960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 34), 'int')
    keyword_171961 = int_171960
    kwargs_171962 = {'copy': keyword_171961, 'ndmin': keyword_171959}
    # Getting the type of 'np' (line 810)
    np_171955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 810)
    array_171956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), np_171955, 'array')
    # Calling array(args, kwargs) (line 810)
    array_call_result_171963 = invoke(stypy.reporting.localization.Localization(__file__, 810, 8), array_171956, *[c_171957], **kwargs_171962)
    
    # Assigning a type to the variable 'c' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'c', array_call_result_171963)
    
    
    # Getting the type of 'c' (line 811)
    c_171964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 811)
    dtype_171965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 7), c_171964, 'dtype')
    # Obtaining the member 'char' of a type (line 811)
    char_171966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 7), dtype_171965, 'char')
    str_171967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 811)
    result_contains_171968 = python_operator(stypy.reporting.localization.Localization(__file__, 811, 7), 'in', char_171966, str_171967)
    
    # Testing the type of an if condition (line 811)
    if_condition_171969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 811, 4), result_contains_171968)
    # Assigning a type to the variable 'if_condition_171969' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'if_condition_171969', if_condition_171969)
    # SSA begins for if statement (line 811)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 812):
    
    # Assigning a Call to a Name (line 812):
    
    # Call to astype(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'np' (line 812)
    np_171972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 812)
    double_171973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 21), np_171972, 'double')
    # Processing the call keyword arguments (line 812)
    kwargs_171974 = {}
    # Getting the type of 'c' (line 812)
    c_171970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 812)
    astype_171971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 12), c_171970, 'astype')
    # Calling astype(args, kwargs) (line 812)
    astype_call_result_171975 = invoke(stypy.reporting.localization.Localization(__file__, 812, 12), astype_171971, *[double_171973], **kwargs_171974)
    
    # Assigning a type to the variable 'c' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'c', astype_call_result_171975)
    # SSA join for if statement (line 811)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to iterable(...): (line 813)
    # Processing the call arguments (line 813)
    # Getting the type of 'k' (line 813)
    k_171978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 23), 'k', False)
    # Processing the call keyword arguments (line 813)
    kwargs_171979 = {}
    # Getting the type of 'np' (line 813)
    np_171976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 11), 'np', False)
    # Obtaining the member 'iterable' of a type (line 813)
    iterable_171977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 11), np_171976, 'iterable')
    # Calling iterable(args, kwargs) (line 813)
    iterable_call_result_171980 = invoke(stypy.reporting.localization.Localization(__file__, 813, 11), iterable_171977, *[k_171978], **kwargs_171979)
    
    # Applying the 'not' unary operator (line 813)
    result_not__171981 = python_operator(stypy.reporting.localization.Localization(__file__, 813, 7), 'not', iterable_call_result_171980)
    
    # Testing the type of an if condition (line 813)
    if_condition_171982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 813, 4), result_not__171981)
    # Assigning a type to the variable 'if_condition_171982' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'if_condition_171982', if_condition_171982)
    # SSA begins for if statement (line 813)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 814):
    
    # Assigning a List to a Name (line 814):
    
    # Obtaining an instance of the builtin type 'list' (line 814)
    list_171983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 814)
    # Adding element type (line 814)
    # Getting the type of 'k' (line 814)
    k_171984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 12), list_171983, k_171984)
    
    # Assigning a type to the variable 'k' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'k', list_171983)
    # SSA join for if statement (line 813)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 815):
    
    # Assigning a Subscript to a Name (line 815):
    
    # Obtaining the type of the subscript
    int_171985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 815)
    list_171990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 815)
    # Adding element type (line 815)
    # Getting the type of 'm' (line 815)
    m_171991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_171990, m_171991)
    # Adding element type (line 815)
    # Getting the type of 'axis' (line 815)
    axis_171992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_171990, axis_171992)
    
    comprehension_171993 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_171990)
    # Assigning a type to the variable 't' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 't', comprehension_171993)
    
    # Call to int(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 't' (line 815)
    t_171987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 't', False)
    # Processing the call keyword arguments (line 815)
    kwargs_171988 = {}
    # Getting the type of 'int' (line 815)
    int_171986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 'int', False)
    # Calling int(args, kwargs) (line 815)
    int_call_result_171989 = invoke(stypy.reporting.localization.Localization(__file__, 815, 18), int_171986, *[t_171987], **kwargs_171988)
    
    list_171994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_171994, int_call_result_171989)
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___171995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 4), list_171994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_171996 = invoke(stypy.reporting.localization.Localization(__file__, 815, 4), getitem___171995, int_171985)
    
    # Assigning a type to the variable 'tuple_var_assignment_170789' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_170789', subscript_call_result_171996)
    
    # Assigning a Subscript to a Name (line 815):
    
    # Obtaining the type of the subscript
    int_171997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 815)
    list_172002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 815)
    # Adding element type (line 815)
    # Getting the type of 'm' (line 815)
    m_172003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_172002, m_172003)
    # Adding element type (line 815)
    # Getting the type of 'axis' (line 815)
    axis_172004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_172002, axis_172004)
    
    comprehension_172005 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_172002)
    # Assigning a type to the variable 't' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 't', comprehension_172005)
    
    # Call to int(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 't' (line 815)
    t_171999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 't', False)
    # Processing the call keyword arguments (line 815)
    kwargs_172000 = {}
    # Getting the type of 'int' (line 815)
    int_171998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 'int', False)
    # Calling int(args, kwargs) (line 815)
    int_call_result_172001 = invoke(stypy.reporting.localization.Localization(__file__, 815, 18), int_171998, *[t_171999], **kwargs_172000)
    
    list_172006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_172006, int_call_result_172001)
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___172007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 4), list_172006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_172008 = invoke(stypy.reporting.localization.Localization(__file__, 815, 4), getitem___172007, int_171997)
    
    # Assigning a type to the variable 'tuple_var_assignment_170790' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_170790', subscript_call_result_172008)
    
    # Assigning a Name to a Name (line 815):
    # Getting the type of 'tuple_var_assignment_170789' (line 815)
    tuple_var_assignment_170789_172009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_170789')
    # Assigning a type to the variable 'cnt' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'cnt', tuple_var_assignment_170789_172009)
    
    # Assigning a Name to a Name (line 815):
    # Getting the type of 'tuple_var_assignment_170790' (line 815)
    tuple_var_assignment_170790_172010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_170790')
    # Assigning a type to the variable 'iaxis' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 9), 'iaxis', tuple_var_assignment_170790_172010)
    
    
    # Getting the type of 'cnt' (line 817)
    cnt_172011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 7), 'cnt')
    # Getting the type of 'm' (line 817)
    m_172012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 14), 'm')
    # Applying the binary operator '!=' (line 817)
    result_ne_172013 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 7), '!=', cnt_172011, m_172012)
    
    # Testing the type of an if condition (line 817)
    if_condition_172014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 4), result_ne_172013)
    # Assigning a type to the variable 'if_condition_172014' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'if_condition_172014', if_condition_172014)
    # SSA begins for if statement (line 817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 818)
    # Processing the call arguments (line 818)
    str_172016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 25), 'str', 'The order of integration must be integer')
    # Processing the call keyword arguments (line 818)
    kwargs_172017 = {}
    # Getting the type of 'ValueError' (line 818)
    ValueError_172015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 818)
    ValueError_call_result_172018 = invoke(stypy.reporting.localization.Localization(__file__, 818, 14), ValueError_172015, *[str_172016], **kwargs_172017)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 818, 8), ValueError_call_result_172018, 'raise parameter', BaseException)
    # SSA join for if statement (line 817)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 819)
    cnt_172019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 7), 'cnt')
    int_172020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 13), 'int')
    # Applying the binary operator '<' (line 819)
    result_lt_172021 = python_operator(stypy.reporting.localization.Localization(__file__, 819, 7), '<', cnt_172019, int_172020)
    
    # Testing the type of an if condition (line 819)
    if_condition_172022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 819, 4), result_lt_172021)
    # Assigning a type to the variable 'if_condition_172022' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'if_condition_172022', if_condition_172022)
    # SSA begins for if statement (line 819)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 820)
    # Processing the call arguments (line 820)
    str_172024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 25), 'str', 'The order of integration must be non-negative')
    # Processing the call keyword arguments (line 820)
    kwargs_172025 = {}
    # Getting the type of 'ValueError' (line 820)
    ValueError_172023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 820)
    ValueError_call_result_172026 = invoke(stypy.reporting.localization.Localization(__file__, 820, 14), ValueError_172023, *[str_172024], **kwargs_172025)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 820, 8), ValueError_call_result_172026, 'raise parameter', BaseException)
    # SSA join for if statement (line 819)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 821)
    # Processing the call arguments (line 821)
    # Getting the type of 'k' (line 821)
    k_172028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 11), 'k', False)
    # Processing the call keyword arguments (line 821)
    kwargs_172029 = {}
    # Getting the type of 'len' (line 821)
    len_172027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 7), 'len', False)
    # Calling len(args, kwargs) (line 821)
    len_call_result_172030 = invoke(stypy.reporting.localization.Localization(__file__, 821, 7), len_172027, *[k_172028], **kwargs_172029)
    
    # Getting the type of 'cnt' (line 821)
    cnt_172031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'cnt')
    # Applying the binary operator '>' (line 821)
    result_gt_172032 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 7), '>', len_call_result_172030, cnt_172031)
    
    # Testing the type of an if condition (line 821)
    if_condition_172033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 4), result_gt_172032)
    # Assigning a type to the variable 'if_condition_172033' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'if_condition_172033', if_condition_172033)
    # SSA begins for if statement (line 821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 822)
    # Processing the call arguments (line 822)
    str_172035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 25), 'str', 'Too many integration constants')
    # Processing the call keyword arguments (line 822)
    kwargs_172036 = {}
    # Getting the type of 'ValueError' (line 822)
    ValueError_172034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 822)
    ValueError_call_result_172037 = invoke(stypy.reporting.localization.Localization(__file__, 822, 14), ValueError_172034, *[str_172035], **kwargs_172036)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 822, 8), ValueError_call_result_172037, 'raise parameter', BaseException)
    # SSA join for if statement (line 821)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 823)
    iaxis_172038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 7), 'iaxis')
    # Getting the type of 'axis' (line 823)
    axis_172039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 16), 'axis')
    # Applying the binary operator '!=' (line 823)
    result_ne_172040 = python_operator(stypy.reporting.localization.Localization(__file__, 823, 7), '!=', iaxis_172038, axis_172039)
    
    # Testing the type of an if condition (line 823)
    if_condition_172041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 823, 4), result_ne_172040)
    # Assigning a type to the variable 'if_condition_172041' (line 823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 4), 'if_condition_172041', if_condition_172041)
    # SSA begins for if statement (line 823)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 824)
    # Processing the call arguments (line 824)
    str_172043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 824)
    kwargs_172044 = {}
    # Getting the type of 'ValueError' (line 824)
    ValueError_172042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 824)
    ValueError_call_result_172045 = invoke(stypy.reporting.localization.Localization(__file__, 824, 14), ValueError_172042, *[str_172043], **kwargs_172044)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 824, 8), ValueError_call_result_172045, 'raise parameter', BaseException)
    # SSA join for if statement (line 823)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 825)
    c_172046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 825)
    ndim_172047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 12), c_172046, 'ndim')
    # Applying the 'usub' unary operator (line 825)
    result___neg___172048 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), 'usub', ndim_172047)
    
    # Getting the type of 'iaxis' (line 825)
    iaxis_172049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 22), 'iaxis')
    # Applying the binary operator '<=' (line 825)
    result_le_172050 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '<=', result___neg___172048, iaxis_172049)
    # Getting the type of 'c' (line 825)
    c_172051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 825)
    ndim_172052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 30), c_172051, 'ndim')
    # Applying the binary operator '<' (line 825)
    result_lt_172053 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '<', iaxis_172049, ndim_172052)
    # Applying the binary operator '&' (line 825)
    result_and__172054 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '&', result_le_172050, result_lt_172053)
    
    # Applying the 'not' unary operator (line 825)
    result_not__172055 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 7), 'not', result_and__172054)
    
    # Testing the type of an if condition (line 825)
    if_condition_172056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 825, 4), result_not__172055)
    # Assigning a type to the variable 'if_condition_172056' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'if_condition_172056', if_condition_172056)
    # SSA begins for if statement (line 825)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 826)
    # Processing the call arguments (line 826)
    str_172058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 826)
    kwargs_172059 = {}
    # Getting the type of 'ValueError' (line 826)
    ValueError_172057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 826)
    ValueError_call_result_172060 = invoke(stypy.reporting.localization.Localization(__file__, 826, 14), ValueError_172057, *[str_172058], **kwargs_172059)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 826, 8), ValueError_call_result_172060, 'raise parameter', BaseException)
    # SSA join for if statement (line 825)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 827)
    iaxis_172061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 7), 'iaxis')
    int_172062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 15), 'int')
    # Applying the binary operator '<' (line 827)
    result_lt_172063 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 7), '<', iaxis_172061, int_172062)
    
    # Testing the type of an if condition (line 827)
    if_condition_172064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 4), result_lt_172063)
    # Assigning a type to the variable 'if_condition_172064' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'if_condition_172064', if_condition_172064)
    # SSA begins for if statement (line 827)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 828)
    iaxis_172065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'iaxis')
    # Getting the type of 'c' (line 828)
    c_172066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 828)
    ndim_172067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 17), c_172066, 'ndim')
    # Applying the binary operator '+=' (line 828)
    result_iadd_172068 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 8), '+=', iaxis_172065, ndim_172067)
    # Assigning a type to the variable 'iaxis' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'iaxis', result_iadd_172068)
    
    # SSA join for if statement (line 827)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 830)
    cnt_172069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 7), 'cnt')
    int_172070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 14), 'int')
    # Applying the binary operator '==' (line 830)
    result_eq_172071 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 7), '==', cnt_172069, int_172070)
    
    # Testing the type of an if condition (line 830)
    if_condition_172072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 830, 4), result_eq_172071)
    # Assigning a type to the variable 'if_condition_172072' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'if_condition_172072', if_condition_172072)
    # SSA begins for if statement (line 830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 831)
    c_172073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'stypy_return_type', c_172073)
    # SSA join for if statement (line 830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 833):
    
    # Assigning a Call to a Name (line 833):
    
    # Call to rollaxis(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'c' (line 833)
    c_172076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 20), 'c', False)
    # Getting the type of 'iaxis' (line 833)
    iaxis_172077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 833)
    kwargs_172078 = {}
    # Getting the type of 'np' (line 833)
    np_172074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 833)
    rollaxis_172075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 8), np_172074, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 833)
    rollaxis_call_result_172079 = invoke(stypy.reporting.localization.Localization(__file__, 833, 8), rollaxis_172075, *[c_172076, iaxis_172077], **kwargs_172078)
    
    # Assigning a type to the variable 'c' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'c', rollaxis_call_result_172079)
    
    # Assigning a BinOp to a Name (line 834):
    
    # Assigning a BinOp to a Name (line 834):
    
    # Call to list(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'k' (line 834)
    k_172081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 13), 'k', False)
    # Processing the call keyword arguments (line 834)
    kwargs_172082 = {}
    # Getting the type of 'list' (line 834)
    list_172080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'list', False)
    # Calling list(args, kwargs) (line 834)
    list_call_result_172083 = invoke(stypy.reporting.localization.Localization(__file__, 834, 8), list_172080, *[k_172081], **kwargs_172082)
    
    
    # Obtaining an instance of the builtin type 'list' (line 834)
    list_172084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 834)
    # Adding element type (line 834)
    int_172085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 18), list_172084, int_172085)
    
    # Getting the type of 'cnt' (line 834)
    cnt_172086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 23), 'cnt')
    
    # Call to len(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'k' (line 834)
    k_172088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 33), 'k', False)
    # Processing the call keyword arguments (line 834)
    kwargs_172089 = {}
    # Getting the type of 'len' (line 834)
    len_172087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 29), 'len', False)
    # Calling len(args, kwargs) (line 834)
    len_call_result_172090 = invoke(stypy.reporting.localization.Localization(__file__, 834, 29), len_172087, *[k_172088], **kwargs_172089)
    
    # Applying the binary operator '-' (line 834)
    result_sub_172091 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 23), '-', cnt_172086, len_call_result_172090)
    
    # Applying the binary operator '*' (line 834)
    result_mul_172092 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 18), '*', list_172084, result_sub_172091)
    
    # Applying the binary operator '+' (line 834)
    result_add_172093 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 8), '+', list_call_result_172083, result_mul_172092)
    
    # Assigning a type to the variable 'k' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'k', result_add_172093)
    
    
    # Call to range(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'cnt' (line 835)
    cnt_172095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 19), 'cnt', False)
    # Processing the call keyword arguments (line 835)
    kwargs_172096 = {}
    # Getting the type of 'range' (line 835)
    range_172094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 13), 'range', False)
    # Calling range(args, kwargs) (line 835)
    range_call_result_172097 = invoke(stypy.reporting.localization.Localization(__file__, 835, 13), range_172094, *[cnt_172095], **kwargs_172096)
    
    # Testing the type of a for loop iterable (line 835)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 835, 4), range_call_result_172097)
    # Getting the type of the for loop variable (line 835)
    for_loop_var_172098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 835, 4), range_call_result_172097)
    # Assigning a type to the variable 'i' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 4), 'i', for_loop_var_172098)
    # SSA begins for a for statement (line 835)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 836):
    
    # Assigning a Call to a Name (line 836):
    
    # Call to len(...): (line 836)
    # Processing the call arguments (line 836)
    # Getting the type of 'c' (line 836)
    c_172100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'c', False)
    # Processing the call keyword arguments (line 836)
    kwargs_172101 = {}
    # Getting the type of 'len' (line 836)
    len_172099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 12), 'len', False)
    # Calling len(args, kwargs) (line 836)
    len_call_result_172102 = invoke(stypy.reporting.localization.Localization(__file__, 836, 12), len_172099, *[c_172100], **kwargs_172101)
    
    # Assigning a type to the variable 'n' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'n', len_call_result_172102)
    
    # Getting the type of 'c' (line 837)
    c_172103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'c')
    # Getting the type of 'scl' (line 837)
    scl_172104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 13), 'scl')
    # Applying the binary operator '*=' (line 837)
    result_imul_172105 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 8), '*=', c_172103, scl_172104)
    # Assigning a type to the variable 'c' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'c', result_imul_172105)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 838)
    n_172106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 11), 'n')
    int_172107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 16), 'int')
    # Applying the binary operator '==' (line 838)
    result_eq_172108 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 11), '==', n_172106, int_172107)
    
    
    # Call to all(...): (line 838)
    # Processing the call arguments (line 838)
    
    
    # Obtaining the type of the subscript
    int_172111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 31), 'int')
    # Getting the type of 'c' (line 838)
    c_172112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 838)
    getitem___172113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 29), c_172112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 838)
    subscript_call_result_172114 = invoke(stypy.reporting.localization.Localization(__file__, 838, 29), getitem___172113, int_172111)
    
    int_172115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 37), 'int')
    # Applying the binary operator '==' (line 838)
    result_eq_172116 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 29), '==', subscript_call_result_172114, int_172115)
    
    # Processing the call keyword arguments (line 838)
    kwargs_172117 = {}
    # Getting the type of 'np' (line 838)
    np_172109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 22), 'np', False)
    # Obtaining the member 'all' of a type (line 838)
    all_172110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 22), np_172109, 'all')
    # Calling all(args, kwargs) (line 838)
    all_call_result_172118 = invoke(stypy.reporting.localization.Localization(__file__, 838, 22), all_172110, *[result_eq_172116], **kwargs_172117)
    
    # Applying the binary operator 'and' (line 838)
    result_and_keyword_172119 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 11), 'and', result_eq_172108, all_call_result_172118)
    
    # Testing the type of an if condition (line 838)
    if_condition_172120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 838, 8), result_and_keyword_172119)
    # Assigning a type to the variable 'if_condition_172120' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'if_condition_172120', if_condition_172120)
    # SSA begins for if statement (line 838)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 839)
    c_172121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'c')
    
    # Obtaining the type of the subscript
    int_172122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 14), 'int')
    # Getting the type of 'c' (line 839)
    c_172123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___172124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 12), c_172123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_172125 = invoke(stypy.reporting.localization.Localization(__file__, 839, 12), getitem___172124, int_172122)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 839)
    i_172126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 22), 'i')
    # Getting the type of 'k' (line 839)
    k_172127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 20), 'k')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___172128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 20), k_172127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_172129 = invoke(stypy.reporting.localization.Localization(__file__, 839, 20), getitem___172128, i_172126)
    
    # Applying the binary operator '+=' (line 839)
    result_iadd_172130 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 12), '+=', subscript_call_result_172125, subscript_call_result_172129)
    # Getting the type of 'c' (line 839)
    c_172131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'c')
    int_172132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 14), 'int')
    # Storing an element on a container (line 839)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 12), c_172131, (int_172132, result_iadd_172130))
    
    # SSA branch for the else part of an if statement (line 838)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 841):
    
    # Assigning a Call to a Name (line 841):
    
    # Call to empty(...): (line 841)
    # Processing the call arguments (line 841)
    
    # Obtaining an instance of the builtin type 'tuple' (line 841)
    tuple_172135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 841)
    # Adding element type (line 841)
    # Getting the type of 'n' (line 841)
    n_172136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 28), 'n', False)
    int_172137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 32), 'int')
    # Applying the binary operator '+' (line 841)
    result_add_172138 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 28), '+', n_172136, int_172137)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 28), tuple_172135, result_add_172138)
    
    
    # Obtaining the type of the subscript
    int_172139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 46), 'int')
    slice_172140 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 841, 38), int_172139, None, None)
    # Getting the type of 'c' (line 841)
    c_172141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 841)
    shape_172142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 38), c_172141, 'shape')
    # Obtaining the member '__getitem__' of a type (line 841)
    getitem___172143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 38), shape_172142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 841)
    subscript_call_result_172144 = invoke(stypy.reporting.localization.Localization(__file__, 841, 38), getitem___172143, slice_172140)
    
    # Applying the binary operator '+' (line 841)
    result_add_172145 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 27), '+', tuple_172135, subscript_call_result_172144)
    
    # Processing the call keyword arguments (line 841)
    # Getting the type of 'c' (line 841)
    c_172146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 57), 'c', False)
    # Obtaining the member 'dtype' of a type (line 841)
    dtype_172147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 57), c_172146, 'dtype')
    keyword_172148 = dtype_172147
    kwargs_172149 = {'dtype': keyword_172148}
    # Getting the type of 'np' (line 841)
    np_172133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 841)
    empty_172134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 18), np_172133, 'empty')
    # Calling empty(args, kwargs) (line 841)
    empty_call_result_172150 = invoke(stypy.reporting.localization.Localization(__file__, 841, 18), empty_172134, *[result_add_172145], **kwargs_172149)
    
    # Assigning a type to the variable 'tmp' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'tmp', empty_call_result_172150)
    
    # Assigning a Subscript to a Subscript (line 842):
    
    # Assigning a Subscript to a Subscript (line 842):
    
    # Obtaining the type of the subscript
    int_172151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 23), 'int')
    # Getting the type of 'c' (line 842)
    c_172152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 842)
    getitem___172153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 21), c_172152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 842)
    subscript_call_result_172154 = invoke(stypy.reporting.localization.Localization(__file__, 842, 21), getitem___172153, int_172151)
    
    # Getting the type of 'tmp' (line 842)
    tmp_172155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'tmp')
    int_172156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 16), 'int')
    # Storing an element on a container (line 842)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 12), tmp_172155, (int_172156, subscript_call_result_172154))
    
    # Assigning a UnaryOp to a Subscript (line 843):
    
    # Assigning a UnaryOp to a Subscript (line 843):
    
    
    # Obtaining the type of the subscript
    int_172157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 24), 'int')
    # Getting the type of 'c' (line 843)
    c_172158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 843)
    getitem___172159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 22), c_172158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 843)
    subscript_call_result_172160 = invoke(stypy.reporting.localization.Localization(__file__, 843, 22), getitem___172159, int_172157)
    
    # Applying the 'usub' unary operator (line 843)
    result___neg___172161 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 21), 'usub', subscript_call_result_172160)
    
    # Getting the type of 'tmp' (line 843)
    tmp_172162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'tmp')
    int_172163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 16), 'int')
    # Storing an element on a container (line 843)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 843, 12), tmp_172162, (int_172163, result___neg___172161))
    
    
    # Call to range(...): (line 844)
    # Processing the call arguments (line 844)
    int_172165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 27), 'int')
    # Getting the type of 'n' (line 844)
    n_172166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 30), 'n', False)
    # Processing the call keyword arguments (line 844)
    kwargs_172167 = {}
    # Getting the type of 'range' (line 844)
    range_172164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 21), 'range', False)
    # Calling range(args, kwargs) (line 844)
    range_call_result_172168 = invoke(stypy.reporting.localization.Localization(__file__, 844, 21), range_172164, *[int_172165, n_172166], **kwargs_172167)
    
    # Testing the type of a for loop iterable (line 844)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 844, 12), range_call_result_172168)
    # Getting the type of the for loop variable (line 844)
    for_loop_var_172169 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 844, 12), range_call_result_172168)
    # Assigning a type to the variable 'j' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'j', for_loop_var_172169)
    # SSA begins for a for statement (line 844)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'tmp' (line 845)
    tmp_172170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 16), 'tmp')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 845)
    j_172171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 20), 'j')
    # Getting the type of 'tmp' (line 845)
    tmp_172172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 16), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___172173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 16), tmp_172172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_172174 = invoke(stypy.reporting.localization.Localization(__file__, 845, 16), getitem___172173, j_172171)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 845)
    j_172175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 28), 'j')
    # Getting the type of 'c' (line 845)
    c_172176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 26), 'c')
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___172177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 26), c_172176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_172178 = invoke(stypy.reporting.localization.Localization(__file__, 845, 26), getitem___172177, j_172175)
    
    # Applying the binary operator '+=' (line 845)
    result_iadd_172179 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 16), '+=', subscript_call_result_172174, subscript_call_result_172178)
    # Getting the type of 'tmp' (line 845)
    tmp_172180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 16), 'tmp')
    # Getting the type of 'j' (line 845)
    j_172181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 20), 'j')
    # Storing an element on a container (line 845)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 16), tmp_172180, (j_172181, result_iadd_172179))
    
    
    # Assigning a UnaryOp to a Subscript (line 846):
    
    # Assigning a UnaryOp to a Subscript (line 846):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 846)
    j_172182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 32), 'j')
    # Getting the type of 'c' (line 846)
    c_172183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 30), 'c')
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___172184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 30), c_172183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_172185 = invoke(stypy.reporting.localization.Localization(__file__, 846, 30), getitem___172184, j_172182)
    
    # Applying the 'usub' unary operator (line 846)
    result___neg___172186 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 29), 'usub', subscript_call_result_172185)
    
    # Getting the type of 'tmp' (line 846)
    tmp_172187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 16), 'tmp')
    # Getting the type of 'j' (line 846)
    j_172188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 20), 'j')
    int_172189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 24), 'int')
    # Applying the binary operator '+' (line 846)
    result_add_172190 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 20), '+', j_172188, int_172189)
    
    # Storing an element on a container (line 846)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 16), tmp_172187, (result_add_172190, result___neg___172186))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tmp' (line 847)
    tmp_172191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'tmp')
    
    # Obtaining the type of the subscript
    int_172192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 16), 'int')
    # Getting the type of 'tmp' (line 847)
    tmp_172193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 847)
    getitem___172194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 12), tmp_172193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 847)
    subscript_call_result_172195 = invoke(stypy.reporting.localization.Localization(__file__, 847, 12), getitem___172194, int_172192)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 847)
    i_172196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 24), 'i')
    # Getting the type of 'k' (line 847)
    k_172197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 22), 'k')
    # Obtaining the member '__getitem__' of a type (line 847)
    getitem___172198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 22), k_172197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 847)
    subscript_call_result_172199 = invoke(stypy.reporting.localization.Localization(__file__, 847, 22), getitem___172198, i_172196)
    
    
    # Call to lagval(...): (line 847)
    # Processing the call arguments (line 847)
    # Getting the type of 'lbnd' (line 847)
    lbnd_172201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 36), 'lbnd', False)
    # Getting the type of 'tmp' (line 847)
    tmp_172202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 42), 'tmp', False)
    # Processing the call keyword arguments (line 847)
    kwargs_172203 = {}
    # Getting the type of 'lagval' (line 847)
    lagval_172200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 29), 'lagval', False)
    # Calling lagval(args, kwargs) (line 847)
    lagval_call_result_172204 = invoke(stypy.reporting.localization.Localization(__file__, 847, 29), lagval_172200, *[lbnd_172201, tmp_172202], **kwargs_172203)
    
    # Applying the binary operator '-' (line 847)
    result_sub_172205 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 22), '-', subscript_call_result_172199, lagval_call_result_172204)
    
    # Applying the binary operator '+=' (line 847)
    result_iadd_172206 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 12), '+=', subscript_call_result_172195, result_sub_172205)
    # Getting the type of 'tmp' (line 847)
    tmp_172207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'tmp')
    int_172208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 16), 'int')
    # Storing an element on a container (line 847)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 847, 12), tmp_172207, (int_172208, result_iadd_172206))
    
    
    # Assigning a Name to a Name (line 848):
    
    # Assigning a Name to a Name (line 848):
    # Getting the type of 'tmp' (line 848)
    tmp_172209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 16), 'tmp')
    # Assigning a type to the variable 'c' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 12), 'c', tmp_172209)
    # SSA join for if statement (line 838)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 849):
    
    # Assigning a Call to a Name (line 849):
    
    # Call to rollaxis(...): (line 849)
    # Processing the call arguments (line 849)
    # Getting the type of 'c' (line 849)
    c_172212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 20), 'c', False)
    int_172213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 23), 'int')
    # Getting the type of 'iaxis' (line 849)
    iaxis_172214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 26), 'iaxis', False)
    int_172215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 34), 'int')
    # Applying the binary operator '+' (line 849)
    result_add_172216 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 26), '+', iaxis_172214, int_172215)
    
    # Processing the call keyword arguments (line 849)
    kwargs_172217 = {}
    # Getting the type of 'np' (line 849)
    np_172210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 849)
    rollaxis_172211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 8), np_172210, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 849)
    rollaxis_call_result_172218 = invoke(stypy.reporting.localization.Localization(__file__, 849, 8), rollaxis_172211, *[c_172212, int_172213, result_add_172216], **kwargs_172217)
    
    # Assigning a type to the variable 'c' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 4), 'c', rollaxis_call_result_172218)
    # Getting the type of 'c' (line 850)
    c_172219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 850)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 4), 'stypy_return_type', c_172219)
    
    # ################# End of 'lagint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagint' in the type store
    # Getting the type of 'stypy_return_type' (line 726)
    stypy_return_type_172220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172220)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagint'
    return stypy_return_type_172220

# Assigning a type to the variable 'lagint' (line 726)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'lagint', lagint)

@norecursion
def lagval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 853)
    True_172221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 24), 'True')
    defaults = [True_172221]
    # Create a new context for function 'lagval'
    module_type_store = module_type_store.open_function_context('lagval', 853, 0, False)
    
    # Passed parameters checking function
    lagval.stypy_localization = localization
    lagval.stypy_type_of_self = None
    lagval.stypy_type_store = module_type_store
    lagval.stypy_function_name = 'lagval'
    lagval.stypy_param_names_list = ['x', 'c', 'tensor']
    lagval.stypy_varargs_param_name = None
    lagval.stypy_kwargs_param_name = None
    lagval.stypy_call_defaults = defaults
    lagval.stypy_call_varargs = varargs
    lagval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagval', ['x', 'c', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagval', localization, ['x', 'c', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagval(...)' code ##################

    str_172222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, (-1)), 'str', '\n    Evaluate a Laguerre series at points x.\n\n    If `c` is of length `n + 1`, this function returns the value:\n\n    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, algebra_like\n        The shape of the return value is described above.\n\n    See Also\n    --------\n    lagval2d, laggrid2d, lagval3d, laggrid3d\n\n    Notes\n    -----\n    The evaluation uses Clenshaw recursion, aka synthetic division.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagval\n    >>> coef = [1,2,3]\n    >>> lagval(1, coef)\n    -0.5\n    >>> lagval([[1,2],[3,4]], coef)\n    array([[-0.5, -4. ],\n           [-4.5, -2. ]])\n\n    ')
    
    # Assigning a Call to a Name (line 922):
    
    # Assigning a Call to a Name (line 922):
    
    # Call to array(...): (line 922)
    # Processing the call arguments (line 922)
    # Getting the type of 'c' (line 922)
    c_172225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 17), 'c', False)
    # Processing the call keyword arguments (line 922)
    int_172226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 26), 'int')
    keyword_172227 = int_172226
    int_172228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 34), 'int')
    keyword_172229 = int_172228
    kwargs_172230 = {'copy': keyword_172229, 'ndmin': keyword_172227}
    # Getting the type of 'np' (line 922)
    np_172223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 922)
    array_172224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 8), np_172223, 'array')
    # Calling array(args, kwargs) (line 922)
    array_call_result_172231 = invoke(stypy.reporting.localization.Localization(__file__, 922, 8), array_172224, *[c_172225], **kwargs_172230)
    
    # Assigning a type to the variable 'c' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'c', array_call_result_172231)
    
    
    # Getting the type of 'c' (line 923)
    c_172232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 923)
    dtype_172233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 7), c_172232, 'dtype')
    # Obtaining the member 'char' of a type (line 923)
    char_172234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 7), dtype_172233, 'char')
    str_172235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 923)
    result_contains_172236 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 7), 'in', char_172234, str_172235)
    
    # Testing the type of an if condition (line 923)
    if_condition_172237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 923, 4), result_contains_172236)
    # Assigning a type to the variable 'if_condition_172237' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'if_condition_172237', if_condition_172237)
    # SSA begins for if statement (line 923)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 924):
    
    # Assigning a Call to a Name (line 924):
    
    # Call to astype(...): (line 924)
    # Processing the call arguments (line 924)
    # Getting the type of 'np' (line 924)
    np_172240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 924)
    double_172241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 21), np_172240, 'double')
    # Processing the call keyword arguments (line 924)
    kwargs_172242 = {}
    # Getting the type of 'c' (line 924)
    c_172238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 924)
    astype_172239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), c_172238, 'astype')
    # Calling astype(args, kwargs) (line 924)
    astype_call_result_172243 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), astype_172239, *[double_172241], **kwargs_172242)
    
    # Assigning a type to the variable 'c' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'c', astype_call_result_172243)
    # SSA join for if statement (line 923)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'x' (line 925)
    x_172245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 18), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 925)
    tuple_172246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 925)
    # Adding element type (line 925)
    # Getting the type of 'tuple' (line 925)
    tuple_172247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 22), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 925, 22), tuple_172246, tuple_172247)
    # Adding element type (line 925)
    # Getting the type of 'list' (line 925)
    list_172248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 925, 22), tuple_172246, list_172248)
    
    # Processing the call keyword arguments (line 925)
    kwargs_172249 = {}
    # Getting the type of 'isinstance' (line 925)
    isinstance_172244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 925)
    isinstance_call_result_172250 = invoke(stypy.reporting.localization.Localization(__file__, 925, 7), isinstance_172244, *[x_172245, tuple_172246], **kwargs_172249)
    
    # Testing the type of an if condition (line 925)
    if_condition_172251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 925, 4), isinstance_call_result_172250)
    # Assigning a type to the variable 'if_condition_172251' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 4), 'if_condition_172251', if_condition_172251)
    # SSA begins for if statement (line 925)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 926):
    
    # Assigning a Call to a Name (line 926):
    
    # Call to asarray(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'x' (line 926)
    x_172254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 23), 'x', False)
    # Processing the call keyword arguments (line 926)
    kwargs_172255 = {}
    # Getting the type of 'np' (line 926)
    np_172252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 926)
    asarray_172253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 12), np_172252, 'asarray')
    # Calling asarray(args, kwargs) (line 926)
    asarray_call_result_172256 = invoke(stypy.reporting.localization.Localization(__file__, 926, 12), asarray_172253, *[x_172254], **kwargs_172255)
    
    # Assigning a type to the variable 'x' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'x', asarray_call_result_172256)
    # SSA join for if statement (line 925)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'x' (line 927)
    x_172258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 18), 'x', False)
    # Getting the type of 'np' (line 927)
    np_172259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 927)
    ndarray_172260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 21), np_172259, 'ndarray')
    # Processing the call keyword arguments (line 927)
    kwargs_172261 = {}
    # Getting the type of 'isinstance' (line 927)
    isinstance_172257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 927)
    isinstance_call_result_172262 = invoke(stypy.reporting.localization.Localization(__file__, 927, 7), isinstance_172257, *[x_172258, ndarray_172260], **kwargs_172261)
    
    # Getting the type of 'tensor' (line 927)
    tensor_172263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 37), 'tensor')
    # Applying the binary operator 'and' (line 927)
    result_and_keyword_172264 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 7), 'and', isinstance_call_result_172262, tensor_172263)
    
    # Testing the type of an if condition (line 927)
    if_condition_172265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 927, 4), result_and_keyword_172264)
    # Assigning a type to the variable 'if_condition_172265' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 4), 'if_condition_172265', if_condition_172265)
    # SSA begins for if statement (line 927)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 928):
    
    # Assigning a Call to a Name (line 928):
    
    # Call to reshape(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 'c' (line 928)
    c_172268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 22), 'c', False)
    # Obtaining the member 'shape' of a type (line 928)
    shape_172269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 22), c_172268, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 928)
    tuple_172270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 928)
    # Adding element type (line 928)
    int_172271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 928, 33), tuple_172270, int_172271)
    
    # Getting the type of 'x' (line 928)
    x_172272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 37), 'x', False)
    # Obtaining the member 'ndim' of a type (line 928)
    ndim_172273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 37), x_172272, 'ndim')
    # Applying the binary operator '*' (line 928)
    result_mul_172274 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 32), '*', tuple_172270, ndim_172273)
    
    # Applying the binary operator '+' (line 928)
    result_add_172275 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 22), '+', shape_172269, result_mul_172274)
    
    # Processing the call keyword arguments (line 928)
    kwargs_172276 = {}
    # Getting the type of 'c' (line 928)
    c_172266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 928)
    reshape_172267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 12), c_172266, 'reshape')
    # Calling reshape(args, kwargs) (line 928)
    reshape_call_result_172277 = invoke(stypy.reporting.localization.Localization(__file__, 928, 12), reshape_172267, *[result_add_172275], **kwargs_172276)
    
    # Assigning a type to the variable 'c' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), 'c', reshape_call_result_172277)
    # SSA join for if statement (line 927)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 930)
    # Processing the call arguments (line 930)
    # Getting the type of 'c' (line 930)
    c_172279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 11), 'c', False)
    # Processing the call keyword arguments (line 930)
    kwargs_172280 = {}
    # Getting the type of 'len' (line 930)
    len_172278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 7), 'len', False)
    # Calling len(args, kwargs) (line 930)
    len_call_result_172281 = invoke(stypy.reporting.localization.Localization(__file__, 930, 7), len_172278, *[c_172279], **kwargs_172280)
    
    int_172282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 17), 'int')
    # Applying the binary operator '==' (line 930)
    result_eq_172283 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 7), '==', len_call_result_172281, int_172282)
    
    # Testing the type of an if condition (line 930)
    if_condition_172284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 930, 4), result_eq_172283)
    # Assigning a type to the variable 'if_condition_172284' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'if_condition_172284', if_condition_172284)
    # SSA begins for if statement (line 930)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 931):
    
    # Assigning a Subscript to a Name (line 931):
    
    # Obtaining the type of the subscript
    int_172285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 15), 'int')
    # Getting the type of 'c' (line 931)
    c_172286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 931)
    getitem___172287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 13), c_172286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 931)
    subscript_call_result_172288 = invoke(stypy.reporting.localization.Localization(__file__, 931, 13), getitem___172287, int_172285)
    
    # Assigning a type to the variable 'c0' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'c0', subscript_call_result_172288)
    
    # Assigning a Num to a Name (line 932):
    
    # Assigning a Num to a Name (line 932):
    int_172289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 13), 'int')
    # Assigning a type to the variable 'c1' (line 932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'c1', int_172289)
    # SSA branch for the else part of an if statement (line 930)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 933)
    # Processing the call arguments (line 933)
    # Getting the type of 'c' (line 933)
    c_172291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 13), 'c', False)
    # Processing the call keyword arguments (line 933)
    kwargs_172292 = {}
    # Getting the type of 'len' (line 933)
    len_172290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 9), 'len', False)
    # Calling len(args, kwargs) (line 933)
    len_call_result_172293 = invoke(stypy.reporting.localization.Localization(__file__, 933, 9), len_172290, *[c_172291], **kwargs_172292)
    
    int_172294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 19), 'int')
    # Applying the binary operator '==' (line 933)
    result_eq_172295 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 9), '==', len_call_result_172293, int_172294)
    
    # Testing the type of an if condition (line 933)
    if_condition_172296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 933, 9), result_eq_172295)
    # Assigning a type to the variable 'if_condition_172296' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 9), 'if_condition_172296', if_condition_172296)
    # SSA begins for if statement (line 933)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 934):
    
    # Assigning a Subscript to a Name (line 934):
    
    # Obtaining the type of the subscript
    int_172297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 15), 'int')
    # Getting the type of 'c' (line 934)
    c_172298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 934)
    getitem___172299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 13), c_172298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 934)
    subscript_call_result_172300 = invoke(stypy.reporting.localization.Localization(__file__, 934, 13), getitem___172299, int_172297)
    
    # Assigning a type to the variable 'c0' (line 934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'c0', subscript_call_result_172300)
    
    # Assigning a Subscript to a Name (line 935):
    
    # Assigning a Subscript to a Name (line 935):
    
    # Obtaining the type of the subscript
    int_172301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 15), 'int')
    # Getting the type of 'c' (line 935)
    c_172302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 935)
    getitem___172303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 13), c_172302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 935)
    subscript_call_result_172304 = invoke(stypy.reporting.localization.Localization(__file__, 935, 13), getitem___172303, int_172301)
    
    # Assigning a type to the variable 'c1' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'c1', subscript_call_result_172304)
    # SSA branch for the else part of an if statement (line 933)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 937):
    
    # Assigning a Call to a Name (line 937):
    
    # Call to len(...): (line 937)
    # Processing the call arguments (line 937)
    # Getting the type of 'c' (line 937)
    c_172306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 17), 'c', False)
    # Processing the call keyword arguments (line 937)
    kwargs_172307 = {}
    # Getting the type of 'len' (line 937)
    len_172305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 13), 'len', False)
    # Calling len(args, kwargs) (line 937)
    len_call_result_172308 = invoke(stypy.reporting.localization.Localization(__file__, 937, 13), len_172305, *[c_172306], **kwargs_172307)
    
    # Assigning a type to the variable 'nd' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 8), 'nd', len_call_result_172308)
    
    # Assigning a Subscript to a Name (line 938):
    
    # Assigning a Subscript to a Name (line 938):
    
    # Obtaining the type of the subscript
    int_172309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 15), 'int')
    # Getting the type of 'c' (line 938)
    c_172310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 938)
    getitem___172311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 13), c_172310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 938)
    subscript_call_result_172312 = invoke(stypy.reporting.localization.Localization(__file__, 938, 13), getitem___172311, int_172309)
    
    # Assigning a type to the variable 'c0' (line 938)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 938, 8), 'c0', subscript_call_result_172312)
    
    # Assigning a Subscript to a Name (line 939):
    
    # Assigning a Subscript to a Name (line 939):
    
    # Obtaining the type of the subscript
    int_172313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 15), 'int')
    # Getting the type of 'c' (line 939)
    c_172314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 939)
    getitem___172315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 13), c_172314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 939)
    subscript_call_result_172316 = invoke(stypy.reporting.localization.Localization(__file__, 939, 13), getitem___172315, int_172313)
    
    # Assigning a type to the variable 'c1' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'c1', subscript_call_result_172316)
    
    
    # Call to range(...): (line 940)
    # Processing the call arguments (line 940)
    int_172318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 23), 'int')
    
    # Call to len(...): (line 940)
    # Processing the call arguments (line 940)
    # Getting the type of 'c' (line 940)
    c_172320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 30), 'c', False)
    # Processing the call keyword arguments (line 940)
    kwargs_172321 = {}
    # Getting the type of 'len' (line 940)
    len_172319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 26), 'len', False)
    # Calling len(args, kwargs) (line 940)
    len_call_result_172322 = invoke(stypy.reporting.localization.Localization(__file__, 940, 26), len_172319, *[c_172320], **kwargs_172321)
    
    int_172323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 35), 'int')
    # Applying the binary operator '+' (line 940)
    result_add_172324 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 26), '+', len_call_result_172322, int_172323)
    
    # Processing the call keyword arguments (line 940)
    kwargs_172325 = {}
    # Getting the type of 'range' (line 940)
    range_172317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 17), 'range', False)
    # Calling range(args, kwargs) (line 940)
    range_call_result_172326 = invoke(stypy.reporting.localization.Localization(__file__, 940, 17), range_172317, *[int_172318, result_add_172324], **kwargs_172325)
    
    # Testing the type of a for loop iterable (line 940)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 940, 8), range_call_result_172326)
    # Getting the type of the for loop variable (line 940)
    for_loop_var_172327 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 940, 8), range_call_result_172326)
    # Assigning a type to the variable 'i' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'i', for_loop_var_172327)
    # SSA begins for a for statement (line 940)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 941):
    
    # Assigning a Name to a Name (line 941):
    # Getting the type of 'c0' (line 941)
    c0_172328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 12), 'tmp', c0_172328)
    
    # Assigning a BinOp to a Name (line 942):
    
    # Assigning a BinOp to a Name (line 942):
    # Getting the type of 'nd' (line 942)
    nd_172329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 17), 'nd')
    int_172330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 22), 'int')
    # Applying the binary operator '-' (line 942)
    result_sub_172331 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 17), '-', nd_172329, int_172330)
    
    # Assigning a type to the variable 'nd' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 12), 'nd', result_sub_172331)
    
    # Assigning a BinOp to a Name (line 943):
    
    # Assigning a BinOp to a Name (line 943):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 943)
    i_172332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 20), 'i')
    # Applying the 'usub' unary operator (line 943)
    result___neg___172333 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 19), 'usub', i_172332)
    
    # Getting the type of 'c' (line 943)
    c_172334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 17), 'c')
    # Obtaining the member '__getitem__' of a type (line 943)
    getitem___172335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 17), c_172334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 943)
    subscript_call_result_172336 = invoke(stypy.reporting.localization.Localization(__file__, 943, 17), getitem___172335, result___neg___172333)
    
    # Getting the type of 'c1' (line 943)
    c1_172337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 26), 'c1')
    # Getting the type of 'nd' (line 943)
    nd_172338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 30), 'nd')
    int_172339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 35), 'int')
    # Applying the binary operator '-' (line 943)
    result_sub_172340 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 30), '-', nd_172338, int_172339)
    
    # Applying the binary operator '*' (line 943)
    result_mul_172341 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 26), '*', c1_172337, result_sub_172340)
    
    # Getting the type of 'nd' (line 943)
    nd_172342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 39), 'nd')
    # Applying the binary operator 'div' (line 943)
    result_div_172343 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 25), 'div', result_mul_172341, nd_172342)
    
    # Applying the binary operator '-' (line 943)
    result_sub_172344 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 17), '-', subscript_call_result_172336, result_div_172343)
    
    # Assigning a type to the variable 'c0' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 12), 'c0', result_sub_172344)
    
    # Assigning a BinOp to a Name (line 944):
    
    # Assigning a BinOp to a Name (line 944):
    # Getting the type of 'tmp' (line 944)
    tmp_172345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 17), 'tmp')
    # Getting the type of 'c1' (line 944)
    c1_172346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 24), 'c1')
    int_172347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 29), 'int')
    # Getting the type of 'nd' (line 944)
    nd_172348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 31), 'nd')
    # Applying the binary operator '*' (line 944)
    result_mul_172349 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 29), '*', int_172347, nd_172348)
    
    int_172350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 36), 'int')
    # Applying the binary operator '-' (line 944)
    result_sub_172351 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 29), '-', result_mul_172349, int_172350)
    
    # Getting the type of 'x' (line 944)
    x_172352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 41), 'x')
    # Applying the binary operator '-' (line 944)
    result_sub_172353 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 28), '-', result_sub_172351, x_172352)
    
    # Applying the binary operator '*' (line 944)
    result_mul_172354 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 24), '*', c1_172346, result_sub_172353)
    
    # Getting the type of 'nd' (line 944)
    nd_172355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 45), 'nd')
    # Applying the binary operator 'div' (line 944)
    result_div_172356 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 23), 'div', result_mul_172354, nd_172355)
    
    # Applying the binary operator '+' (line 944)
    result_add_172357 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 17), '+', tmp_172345, result_div_172356)
    
    # Assigning a type to the variable 'c1' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 12), 'c1', result_add_172357)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 933)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 930)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 945)
    c0_172358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 11), 'c0')
    # Getting the type of 'c1' (line 945)
    c1_172359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 16), 'c1')
    int_172360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 20), 'int')
    # Getting the type of 'x' (line 945)
    x_172361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 24), 'x')
    # Applying the binary operator '-' (line 945)
    result_sub_172362 = python_operator(stypy.reporting.localization.Localization(__file__, 945, 20), '-', int_172360, x_172361)
    
    # Applying the binary operator '*' (line 945)
    result_mul_172363 = python_operator(stypy.reporting.localization.Localization(__file__, 945, 16), '*', c1_172359, result_sub_172362)
    
    # Applying the binary operator '+' (line 945)
    result_add_172364 = python_operator(stypy.reporting.localization.Localization(__file__, 945, 11), '+', c0_172358, result_mul_172363)
    
    # Assigning a type to the variable 'stypy_return_type' (line 945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 4), 'stypy_return_type', result_add_172364)
    
    # ################# End of 'lagval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagval' in the type store
    # Getting the type of 'stypy_return_type' (line 853)
    stypy_return_type_172365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172365)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagval'
    return stypy_return_type_172365

# Assigning a type to the variable 'lagval' (line 853)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 0), 'lagval', lagval)

@norecursion
def lagval2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagval2d'
    module_type_store = module_type_store.open_function_context('lagval2d', 948, 0, False)
    
    # Passed parameters checking function
    lagval2d.stypy_localization = localization
    lagval2d.stypy_type_of_self = None
    lagval2d.stypy_type_store = module_type_store
    lagval2d.stypy_function_name = 'lagval2d'
    lagval2d.stypy_param_names_list = ['x', 'y', 'c']
    lagval2d.stypy_varargs_param_name = None
    lagval2d.stypy_kwargs_param_name = None
    lagval2d.stypy_call_defaults = defaults
    lagval2d.stypy_call_varargs = varargs
    lagval2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagval2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagval2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagval2d(...)' code ##################

    str_172366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, (-1)), 'str', "\n    Evaluate a 2-D Laguerre series at points (x, y).\n\n    This function returns the values:\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` is a 1-D array a one is implicitly appended to its shape to make\n    it 2-D. The shape of the result will be c.shape[2:] + x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and if it isn't an ndarray it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in ``c[i,j]``. If `c` has\n        dimension greater than two the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points formed with\n        pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    lagval, laggrid2d, lagval3d, laggrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 994)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 995):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 995)
    # Processing the call arguments (line 995)
    
    # Obtaining an instance of the builtin type 'tuple' (line 995)
    tuple_172369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 995)
    # Adding element type (line 995)
    # Getting the type of 'x' (line 995)
    x_172370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 25), tuple_172369, x_172370)
    # Adding element type (line 995)
    # Getting the type of 'y' (line 995)
    y_172371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 28), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 25), tuple_172369, y_172371)
    
    # Processing the call keyword arguments (line 995)
    int_172372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 37), 'int')
    keyword_172373 = int_172372
    kwargs_172374 = {'copy': keyword_172373}
    # Getting the type of 'np' (line 995)
    np_172367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 995)
    array_172368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 15), np_172367, 'array')
    # Calling array(args, kwargs) (line 995)
    array_call_result_172375 = invoke(stypy.reporting.localization.Localization(__file__, 995, 15), array_172368, *[tuple_172369], **kwargs_172374)
    
    # Assigning a type to the variable 'call_assignment_170791' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170791', array_call_result_172375)
    
    # Assigning a Call to a Name (line 995):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_172378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 8), 'int')
    # Processing the call keyword arguments
    kwargs_172379 = {}
    # Getting the type of 'call_assignment_170791' (line 995)
    call_assignment_170791_172376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170791', False)
    # Obtaining the member '__getitem__' of a type (line 995)
    getitem___172377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 8), call_assignment_170791_172376, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_172380 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___172377, *[int_172378], **kwargs_172379)
    
    # Assigning a type to the variable 'call_assignment_170792' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170792', getitem___call_result_172380)
    
    # Assigning a Name to a Name (line 995):
    # Getting the type of 'call_assignment_170792' (line 995)
    call_assignment_170792_172381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170792')
    # Assigning a type to the variable 'x' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'x', call_assignment_170792_172381)
    
    # Assigning a Call to a Name (line 995):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_172384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 8), 'int')
    # Processing the call keyword arguments
    kwargs_172385 = {}
    # Getting the type of 'call_assignment_170791' (line 995)
    call_assignment_170791_172382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170791', False)
    # Obtaining the member '__getitem__' of a type (line 995)
    getitem___172383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 8), call_assignment_170791_172382, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_172386 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___172383, *[int_172384], **kwargs_172385)
    
    # Assigning a type to the variable 'call_assignment_170793' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170793', getitem___call_result_172386)
    
    # Assigning a Name to a Name (line 995):
    # Getting the type of 'call_assignment_170793' (line 995)
    call_assignment_170793_172387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_170793')
    # Assigning a type to the variable 'y' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 11), 'y', call_assignment_170793_172387)
    # SSA branch for the except part of a try statement (line 994)
    # SSA branch for the except '<any exception>' branch of a try statement (line 994)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 997)
    # Processing the call arguments (line 997)
    str_172389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 25), 'str', 'x, y are incompatible')
    # Processing the call keyword arguments (line 997)
    kwargs_172390 = {}
    # Getting the type of 'ValueError' (line 997)
    ValueError_172388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 997)
    ValueError_call_result_172391 = invoke(stypy.reporting.localization.Localization(__file__, 997, 14), ValueError_172388, *[str_172389], **kwargs_172390)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 997, 8), ValueError_call_result_172391, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 994)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 999):
    
    # Assigning a Call to a Name (line 999):
    
    # Call to lagval(...): (line 999)
    # Processing the call arguments (line 999)
    # Getting the type of 'x' (line 999)
    x_172393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 15), 'x', False)
    # Getting the type of 'c' (line 999)
    c_172394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 18), 'c', False)
    # Processing the call keyword arguments (line 999)
    kwargs_172395 = {}
    # Getting the type of 'lagval' (line 999)
    lagval_172392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 999)
    lagval_call_result_172396 = invoke(stypy.reporting.localization.Localization(__file__, 999, 8), lagval_172392, *[x_172393, c_172394], **kwargs_172395)
    
    # Assigning a type to the variable 'c' (line 999)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'c', lagval_call_result_172396)
    
    # Assigning a Call to a Name (line 1000):
    
    # Assigning a Call to a Name (line 1000):
    
    # Call to lagval(...): (line 1000)
    # Processing the call arguments (line 1000)
    # Getting the type of 'y' (line 1000)
    y_172398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 15), 'y', False)
    # Getting the type of 'c' (line 1000)
    c_172399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 18), 'c', False)
    # Processing the call keyword arguments (line 1000)
    # Getting the type of 'False' (line 1000)
    False_172400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 28), 'False', False)
    keyword_172401 = False_172400
    kwargs_172402 = {'tensor': keyword_172401}
    # Getting the type of 'lagval' (line 1000)
    lagval_172397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1000)
    lagval_call_result_172403 = invoke(stypy.reporting.localization.Localization(__file__, 1000, 8), lagval_172397, *[y_172398, c_172399], **kwargs_172402)
    
    # Assigning a type to the variable 'c' (line 1000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'c', lagval_call_result_172403)
    # Getting the type of 'c' (line 1001)
    c_172404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'stypy_return_type', c_172404)
    
    # ################# End of 'lagval2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagval2d' in the type store
    # Getting the type of 'stypy_return_type' (line 948)
    stypy_return_type_172405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172405)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagval2d'
    return stypy_return_type_172405

# Assigning a type to the variable 'lagval2d' (line 948)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 0), 'lagval2d', lagval2d)

@norecursion
def laggrid2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'laggrid2d'
    module_type_store = module_type_store.open_function_context('laggrid2d', 1004, 0, False)
    
    # Passed parameters checking function
    laggrid2d.stypy_localization = localization
    laggrid2d.stypy_type_of_self = None
    laggrid2d.stypy_type_store = module_type_store
    laggrid2d.stypy_function_name = 'laggrid2d'
    laggrid2d.stypy_param_names_list = ['x', 'y', 'c']
    laggrid2d.stypy_varargs_param_name = None
    laggrid2d.stypy_kwargs_param_name = None
    laggrid2d.stypy_call_defaults = defaults
    laggrid2d.stypy_call_varargs = varargs
    laggrid2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'laggrid2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'laggrid2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'laggrid2d(...)' code ##################

    str_172406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, (-1)), 'str', "\n    Evaluate a 2-D Laguerre series on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape + y.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional Chebyshev series at points in the\n        Cartesian product of `x` and `y`.\n\n    See Also\n    --------\n    lagval, lagval2d, lagval3d, laggrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1054):
    
    # Assigning a Call to a Name (line 1054):
    
    # Call to lagval(...): (line 1054)
    # Processing the call arguments (line 1054)
    # Getting the type of 'x' (line 1054)
    x_172408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 15), 'x', False)
    # Getting the type of 'c' (line 1054)
    c_172409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 18), 'c', False)
    # Processing the call keyword arguments (line 1054)
    kwargs_172410 = {}
    # Getting the type of 'lagval' (line 1054)
    lagval_172407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1054)
    lagval_call_result_172411 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 8), lagval_172407, *[x_172408, c_172409], **kwargs_172410)
    
    # Assigning a type to the variable 'c' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 4), 'c', lagval_call_result_172411)
    
    # Assigning a Call to a Name (line 1055):
    
    # Assigning a Call to a Name (line 1055):
    
    # Call to lagval(...): (line 1055)
    # Processing the call arguments (line 1055)
    # Getting the type of 'y' (line 1055)
    y_172413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 15), 'y', False)
    # Getting the type of 'c' (line 1055)
    c_172414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 18), 'c', False)
    # Processing the call keyword arguments (line 1055)
    kwargs_172415 = {}
    # Getting the type of 'lagval' (line 1055)
    lagval_172412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1055)
    lagval_call_result_172416 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 8), lagval_172412, *[y_172413, c_172414], **kwargs_172415)
    
    # Assigning a type to the variable 'c' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 4), 'c', lagval_call_result_172416)
    # Getting the type of 'c' (line 1056)
    c_172417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1056)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 4), 'stypy_return_type', c_172417)
    
    # ################# End of 'laggrid2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'laggrid2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1004)
    stypy_return_type_172418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172418)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'laggrid2d'
    return stypy_return_type_172418

# Assigning a type to the variable 'laggrid2d' (line 1004)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 0), 'laggrid2d', laggrid2d)

@norecursion
def lagval3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagval3d'
    module_type_store = module_type_store.open_function_context('lagval3d', 1059, 0, False)
    
    # Passed parameters checking function
    lagval3d.stypy_localization = localization
    lagval3d.stypy_type_of_self = None
    lagval3d.stypy_type_store = module_type_store
    lagval3d.stypy_function_name = 'lagval3d'
    lagval3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    lagval3d.stypy_varargs_param_name = None
    lagval3d.stypy_kwargs_param_name = None
    lagval3d.stypy_call_defaults = defaults
    lagval3d.stypy_call_varargs = varargs
    lagval3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagval3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagval3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagval3d(...)' code ##################

    str_172419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, (-1)), 'str', "\n    Evaluate a 3-D Laguerre series at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimension polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    lagval, lagval2d, laggrid2d, laggrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1108):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1108)
    # Processing the call arguments (line 1108)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1108)
    tuple_172422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1108)
    # Adding element type (line 1108)
    # Getting the type of 'x' (line 1108)
    x_172423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1108, 28), tuple_172422, x_172423)
    # Adding element type (line 1108)
    # Getting the type of 'y' (line 1108)
    y_172424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1108, 28), tuple_172422, y_172424)
    # Adding element type (line 1108)
    # Getting the type of 'z' (line 1108)
    z_172425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 34), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1108, 28), tuple_172422, z_172425)
    
    # Processing the call keyword arguments (line 1108)
    int_172426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 43), 'int')
    keyword_172427 = int_172426
    kwargs_172428 = {'copy': keyword_172427}
    # Getting the type of 'np' (line 1108)
    np_172420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 1108)
    array_172421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 18), np_172420, 'array')
    # Calling array(args, kwargs) (line 1108)
    array_call_result_172429 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 18), array_172421, *[tuple_172422], **kwargs_172428)
    
    # Assigning a type to the variable 'call_assignment_170794' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170794', array_call_result_172429)
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_172432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 8), 'int')
    # Processing the call keyword arguments
    kwargs_172433 = {}
    # Getting the type of 'call_assignment_170794' (line 1108)
    call_assignment_170794_172430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170794', False)
    # Obtaining the member '__getitem__' of a type (line 1108)
    getitem___172431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 8), call_assignment_170794_172430, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_172434 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___172431, *[int_172432], **kwargs_172433)
    
    # Assigning a type to the variable 'call_assignment_170795' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170795', getitem___call_result_172434)
    
    # Assigning a Name to a Name (line 1108):
    # Getting the type of 'call_assignment_170795' (line 1108)
    call_assignment_170795_172435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170795')
    # Assigning a type to the variable 'x' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'x', call_assignment_170795_172435)
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_172438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 8), 'int')
    # Processing the call keyword arguments
    kwargs_172439 = {}
    # Getting the type of 'call_assignment_170794' (line 1108)
    call_assignment_170794_172436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170794', False)
    # Obtaining the member '__getitem__' of a type (line 1108)
    getitem___172437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 8), call_assignment_170794_172436, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_172440 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___172437, *[int_172438], **kwargs_172439)
    
    # Assigning a type to the variable 'call_assignment_170796' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170796', getitem___call_result_172440)
    
    # Assigning a Name to a Name (line 1108):
    # Getting the type of 'call_assignment_170796' (line 1108)
    call_assignment_170796_172441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170796')
    # Assigning a type to the variable 'y' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 11), 'y', call_assignment_170796_172441)
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_172444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 8), 'int')
    # Processing the call keyword arguments
    kwargs_172445 = {}
    # Getting the type of 'call_assignment_170794' (line 1108)
    call_assignment_170794_172442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170794', False)
    # Obtaining the member '__getitem__' of a type (line 1108)
    getitem___172443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 8), call_assignment_170794_172442, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_172446 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___172443, *[int_172444], **kwargs_172445)
    
    # Assigning a type to the variable 'call_assignment_170797' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170797', getitem___call_result_172446)
    
    # Assigning a Name to a Name (line 1108):
    # Getting the type of 'call_assignment_170797' (line 1108)
    call_assignment_170797_172447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_170797')
    # Assigning a type to the variable 'z' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 14), 'z', call_assignment_170797_172447)
    # SSA branch for the except part of a try statement (line 1107)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1107)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1110)
    # Processing the call arguments (line 1110)
    str_172449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 25), 'str', 'x, y, z are incompatible')
    # Processing the call keyword arguments (line 1110)
    kwargs_172450 = {}
    # Getting the type of 'ValueError' (line 1110)
    ValueError_172448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1110)
    ValueError_call_result_172451 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 14), ValueError_172448, *[str_172449], **kwargs_172450)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1110, 8), ValueError_call_result_172451, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1112):
    
    # Assigning a Call to a Name (line 1112):
    
    # Call to lagval(...): (line 1112)
    # Processing the call arguments (line 1112)
    # Getting the type of 'x' (line 1112)
    x_172453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 15), 'x', False)
    # Getting the type of 'c' (line 1112)
    c_172454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 18), 'c', False)
    # Processing the call keyword arguments (line 1112)
    kwargs_172455 = {}
    # Getting the type of 'lagval' (line 1112)
    lagval_172452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1112)
    lagval_call_result_172456 = invoke(stypy.reporting.localization.Localization(__file__, 1112, 8), lagval_172452, *[x_172453, c_172454], **kwargs_172455)
    
    # Assigning a type to the variable 'c' (line 1112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 4), 'c', lagval_call_result_172456)
    
    # Assigning a Call to a Name (line 1113):
    
    # Assigning a Call to a Name (line 1113):
    
    # Call to lagval(...): (line 1113)
    # Processing the call arguments (line 1113)
    # Getting the type of 'y' (line 1113)
    y_172458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 15), 'y', False)
    # Getting the type of 'c' (line 1113)
    c_172459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 18), 'c', False)
    # Processing the call keyword arguments (line 1113)
    # Getting the type of 'False' (line 1113)
    False_172460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 28), 'False', False)
    keyword_172461 = False_172460
    kwargs_172462 = {'tensor': keyword_172461}
    # Getting the type of 'lagval' (line 1113)
    lagval_172457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1113)
    lagval_call_result_172463 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 8), lagval_172457, *[y_172458, c_172459], **kwargs_172462)
    
    # Assigning a type to the variable 'c' (line 1113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 4), 'c', lagval_call_result_172463)
    
    # Assigning a Call to a Name (line 1114):
    
    # Assigning a Call to a Name (line 1114):
    
    # Call to lagval(...): (line 1114)
    # Processing the call arguments (line 1114)
    # Getting the type of 'z' (line 1114)
    z_172465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 15), 'z', False)
    # Getting the type of 'c' (line 1114)
    c_172466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 18), 'c', False)
    # Processing the call keyword arguments (line 1114)
    # Getting the type of 'False' (line 1114)
    False_172467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 28), 'False', False)
    keyword_172468 = False_172467
    kwargs_172469 = {'tensor': keyword_172468}
    # Getting the type of 'lagval' (line 1114)
    lagval_172464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1114)
    lagval_call_result_172470 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 8), lagval_172464, *[z_172465, c_172466], **kwargs_172469)
    
    # Assigning a type to the variable 'c' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'c', lagval_call_result_172470)
    # Getting the type of 'c' (line 1115)
    c_172471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 4), 'stypy_return_type', c_172471)
    
    # ################# End of 'lagval3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagval3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1059)
    stypy_return_type_172472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagval3d'
    return stypy_return_type_172472

# Assigning a type to the variable 'lagval3d' (line 1059)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 0), 'lagval3d', lagval3d)

@norecursion
def laggrid3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'laggrid3d'
    module_type_store = module_type_store.open_function_context('laggrid3d', 1118, 0, False)
    
    # Passed parameters checking function
    laggrid3d.stypy_localization = localization
    laggrid3d.stypy_type_of_self = None
    laggrid3d.stypy_type_store = module_type_store
    laggrid3d.stypy_function_name = 'laggrid3d'
    laggrid3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    laggrid3d.stypy_varargs_param_name = None
    laggrid3d.stypy_kwargs_param_name = None
    laggrid3d.stypy_call_defaults = defaults
    laggrid3d.stypy_call_varargs = varargs
    laggrid3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'laggrid3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'laggrid3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'laggrid3d(...)' code ##################

    str_172473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1170, (-1)), 'str', "\n    Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    lagval, lagval2d, laggrid2d, lagval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1171):
    
    # Assigning a Call to a Name (line 1171):
    
    # Call to lagval(...): (line 1171)
    # Processing the call arguments (line 1171)
    # Getting the type of 'x' (line 1171)
    x_172475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 15), 'x', False)
    # Getting the type of 'c' (line 1171)
    c_172476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 18), 'c', False)
    # Processing the call keyword arguments (line 1171)
    kwargs_172477 = {}
    # Getting the type of 'lagval' (line 1171)
    lagval_172474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1171)
    lagval_call_result_172478 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 8), lagval_172474, *[x_172475, c_172476], **kwargs_172477)
    
    # Assigning a type to the variable 'c' (line 1171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 4), 'c', lagval_call_result_172478)
    
    # Assigning a Call to a Name (line 1172):
    
    # Assigning a Call to a Name (line 1172):
    
    # Call to lagval(...): (line 1172)
    # Processing the call arguments (line 1172)
    # Getting the type of 'y' (line 1172)
    y_172480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 15), 'y', False)
    # Getting the type of 'c' (line 1172)
    c_172481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 18), 'c', False)
    # Processing the call keyword arguments (line 1172)
    kwargs_172482 = {}
    # Getting the type of 'lagval' (line 1172)
    lagval_172479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1172)
    lagval_call_result_172483 = invoke(stypy.reporting.localization.Localization(__file__, 1172, 8), lagval_172479, *[y_172480, c_172481], **kwargs_172482)
    
    # Assigning a type to the variable 'c' (line 1172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 4), 'c', lagval_call_result_172483)
    
    # Assigning a Call to a Name (line 1173):
    
    # Assigning a Call to a Name (line 1173):
    
    # Call to lagval(...): (line 1173)
    # Processing the call arguments (line 1173)
    # Getting the type of 'z' (line 1173)
    z_172485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 15), 'z', False)
    # Getting the type of 'c' (line 1173)
    c_172486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 18), 'c', False)
    # Processing the call keyword arguments (line 1173)
    kwargs_172487 = {}
    # Getting the type of 'lagval' (line 1173)
    lagval_172484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 8), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1173)
    lagval_call_result_172488 = invoke(stypy.reporting.localization.Localization(__file__, 1173, 8), lagval_172484, *[z_172485, c_172486], **kwargs_172487)
    
    # Assigning a type to the variable 'c' (line 1173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 4), 'c', lagval_call_result_172488)
    # Getting the type of 'c' (line 1174)
    c_172489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 4), 'stypy_return_type', c_172489)
    
    # ################# End of 'laggrid3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'laggrid3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1118)
    stypy_return_type_172490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172490)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'laggrid3d'
    return stypy_return_type_172490

# Assigning a type to the variable 'laggrid3d' (line 1118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 0), 'laggrid3d', laggrid3d)

@norecursion
def lagvander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagvander'
    module_type_store = module_type_store.open_function_context('lagvander', 1177, 0, False)
    
    # Passed parameters checking function
    lagvander.stypy_localization = localization
    lagvander.stypy_type_of_self = None
    lagvander.stypy_type_store = module_type_store
    lagvander.stypy_function_name = 'lagvander'
    lagvander.stypy_param_names_list = ['x', 'deg']
    lagvander.stypy_varargs_param_name = None
    lagvander.stypy_kwargs_param_name = None
    lagvander.stypy_call_defaults = defaults
    lagvander.stypy_call_varargs = varargs
    lagvander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagvander', ['x', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagvander', localization, ['x', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagvander(...)' code ##################

    str_172491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1220, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degree.\n\n    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points\n    `x`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = L_i(x)\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the degree of the Laguerre polynomial.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and\n    ``lagval(x, c)`` are the same up to roundoff. This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of Laguerre series of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray\n        The pseudo-Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where The last index is the degree of the\n        corresponding Laguerre polynomial.  The dtype will be the same as\n        the converted `x`.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagvander\n    >>> x = np.array([0, 1, 2])\n    >>> lagvander(x, 3)\n    array([[ 1.        ,  1.        ,  1.        ,  1.        ],\n           [ 1.        ,  0.        , -0.5       , -0.66666667],\n           [ 1.        , -1.        , -1.        , -0.33333333]])\n\n    ')
    
    # Assigning a Call to a Name (line 1221):
    
    # Assigning a Call to a Name (line 1221):
    
    # Call to int(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'deg' (line 1221)
    deg_172493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 15), 'deg', False)
    # Processing the call keyword arguments (line 1221)
    kwargs_172494 = {}
    # Getting the type of 'int' (line 1221)
    int_172492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 11), 'int', False)
    # Calling int(args, kwargs) (line 1221)
    int_call_result_172495 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 11), int_172492, *[deg_172493], **kwargs_172494)
    
    # Assigning a type to the variable 'ideg' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'ideg', int_call_result_172495)
    
    
    # Getting the type of 'ideg' (line 1222)
    ideg_172496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 7), 'ideg')
    # Getting the type of 'deg' (line 1222)
    deg_172497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 15), 'deg')
    # Applying the binary operator '!=' (line 1222)
    result_ne_172498 = python_operator(stypy.reporting.localization.Localization(__file__, 1222, 7), '!=', ideg_172496, deg_172497)
    
    # Testing the type of an if condition (line 1222)
    if_condition_172499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1222, 4), result_ne_172498)
    # Assigning a type to the variable 'if_condition_172499' (line 1222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 4), 'if_condition_172499', if_condition_172499)
    # SSA begins for if statement (line 1222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1223)
    # Processing the call arguments (line 1223)
    str_172501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 25), 'str', 'deg must be integer')
    # Processing the call keyword arguments (line 1223)
    kwargs_172502 = {}
    # Getting the type of 'ValueError' (line 1223)
    ValueError_172500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1223)
    ValueError_call_result_172503 = invoke(stypy.reporting.localization.Localization(__file__, 1223, 14), ValueError_172500, *[str_172501], **kwargs_172502)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1223, 8), ValueError_call_result_172503, 'raise parameter', BaseException)
    # SSA join for if statement (line 1222)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ideg' (line 1224)
    ideg_172504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 7), 'ideg')
    int_172505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 14), 'int')
    # Applying the binary operator '<' (line 1224)
    result_lt_172506 = python_operator(stypy.reporting.localization.Localization(__file__, 1224, 7), '<', ideg_172504, int_172505)
    
    # Testing the type of an if condition (line 1224)
    if_condition_172507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1224, 4), result_lt_172506)
    # Assigning a type to the variable 'if_condition_172507' (line 1224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1224, 4), 'if_condition_172507', if_condition_172507)
    # SSA begins for if statement (line 1224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1225)
    # Processing the call arguments (line 1225)
    str_172509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 25), 'str', 'deg must be non-negative')
    # Processing the call keyword arguments (line 1225)
    kwargs_172510 = {}
    # Getting the type of 'ValueError' (line 1225)
    ValueError_172508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1225)
    ValueError_call_result_172511 = invoke(stypy.reporting.localization.Localization(__file__, 1225, 14), ValueError_172508, *[str_172509], **kwargs_172510)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1225, 8), ValueError_call_result_172511, 'raise parameter', BaseException)
    # SSA join for if statement (line 1224)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1227):
    
    # Assigning a BinOp to a Name (line 1227):
    
    # Call to array(...): (line 1227)
    # Processing the call arguments (line 1227)
    # Getting the type of 'x' (line 1227)
    x_172514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 17), 'x', False)
    # Processing the call keyword arguments (line 1227)
    int_172515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 25), 'int')
    keyword_172516 = int_172515
    int_172517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 34), 'int')
    keyword_172518 = int_172517
    kwargs_172519 = {'copy': keyword_172516, 'ndmin': keyword_172518}
    # Getting the type of 'np' (line 1227)
    np_172512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1227)
    array_172513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 8), np_172512, 'array')
    # Calling array(args, kwargs) (line 1227)
    array_call_result_172520 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 8), array_172513, *[x_172514], **kwargs_172519)
    
    float_172521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 39), 'float')
    # Applying the binary operator '+' (line 1227)
    result_add_172522 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 8), '+', array_call_result_172520, float_172521)
    
    # Assigning a type to the variable 'x' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'x', result_add_172522)
    
    # Assigning a BinOp to a Name (line 1228):
    
    # Assigning a BinOp to a Name (line 1228):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1228)
    tuple_172523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1228, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1228)
    # Adding element type (line 1228)
    # Getting the type of 'ideg' (line 1228)
    ideg_172524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 12), 'ideg')
    int_172525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1228, 19), 'int')
    # Applying the binary operator '+' (line 1228)
    result_add_172526 = python_operator(stypy.reporting.localization.Localization(__file__, 1228, 12), '+', ideg_172524, int_172525)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1228, 12), tuple_172523, result_add_172526)
    
    # Getting the type of 'x' (line 1228)
    x_172527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 25), 'x')
    # Obtaining the member 'shape' of a type (line 1228)
    shape_172528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1228, 25), x_172527, 'shape')
    # Applying the binary operator '+' (line 1228)
    result_add_172529 = python_operator(stypy.reporting.localization.Localization(__file__, 1228, 11), '+', tuple_172523, shape_172528)
    
    # Assigning a type to the variable 'dims' (line 1228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 4), 'dims', result_add_172529)
    
    # Assigning a Attribute to a Name (line 1229):
    
    # Assigning a Attribute to a Name (line 1229):
    # Getting the type of 'x' (line 1229)
    x_172530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 1229)
    dtype_172531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1229, 11), x_172530, 'dtype')
    # Assigning a type to the variable 'dtyp' (line 1229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 4), 'dtyp', dtype_172531)
    
    # Assigning a Call to a Name (line 1230):
    
    # Assigning a Call to a Name (line 1230):
    
    # Call to empty(...): (line 1230)
    # Processing the call arguments (line 1230)
    # Getting the type of 'dims' (line 1230)
    dims_172534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 17), 'dims', False)
    # Processing the call keyword arguments (line 1230)
    # Getting the type of 'dtyp' (line 1230)
    dtyp_172535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 29), 'dtyp', False)
    keyword_172536 = dtyp_172535
    kwargs_172537 = {'dtype': keyword_172536}
    # Getting the type of 'np' (line 1230)
    np_172532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1230)
    empty_172533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1230, 8), np_172532, 'empty')
    # Calling empty(args, kwargs) (line 1230)
    empty_call_result_172538 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 8), empty_172533, *[dims_172534], **kwargs_172537)
    
    # Assigning a type to the variable 'v' (line 1230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 4), 'v', empty_call_result_172538)
    
    # Assigning a BinOp to a Subscript (line 1231):
    
    # Assigning a BinOp to a Subscript (line 1231):
    # Getting the type of 'x' (line 1231)
    x_172539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 11), 'x')
    int_172540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 13), 'int')
    # Applying the binary operator '*' (line 1231)
    result_mul_172541 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 11), '*', x_172539, int_172540)
    
    int_172542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 17), 'int')
    # Applying the binary operator '+' (line 1231)
    result_add_172543 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 11), '+', result_mul_172541, int_172542)
    
    # Getting the type of 'v' (line 1231)
    v_172544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 4), 'v')
    int_172545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 6), 'int')
    # Storing an element on a container (line 1231)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1231, 4), v_172544, (int_172545, result_add_172543))
    
    
    # Getting the type of 'ideg' (line 1232)
    ideg_172546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 7), 'ideg')
    int_172547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 14), 'int')
    # Applying the binary operator '>' (line 1232)
    result_gt_172548 = python_operator(stypy.reporting.localization.Localization(__file__, 1232, 7), '>', ideg_172546, int_172547)
    
    # Testing the type of an if condition (line 1232)
    if_condition_172549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1232, 4), result_gt_172548)
    # Assigning a type to the variable 'if_condition_172549' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 4), 'if_condition_172549', if_condition_172549)
    # SSA begins for if statement (line 1232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 1233):
    
    # Assigning a BinOp to a Subscript (line 1233):
    int_172550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 15), 'int')
    # Getting the type of 'x' (line 1233)
    x_172551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 19), 'x')
    # Applying the binary operator '-' (line 1233)
    result_sub_172552 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 15), '-', int_172550, x_172551)
    
    # Getting the type of 'v' (line 1233)
    v_172553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 8), 'v')
    int_172554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 10), 'int')
    # Storing an element on a container (line 1233)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1233, 8), v_172553, (int_172554, result_sub_172552))
    
    
    # Call to range(...): (line 1234)
    # Processing the call arguments (line 1234)
    int_172556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 23), 'int')
    # Getting the type of 'ideg' (line 1234)
    ideg_172557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 26), 'ideg', False)
    int_172558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 33), 'int')
    # Applying the binary operator '+' (line 1234)
    result_add_172559 = python_operator(stypy.reporting.localization.Localization(__file__, 1234, 26), '+', ideg_172557, int_172558)
    
    # Processing the call keyword arguments (line 1234)
    kwargs_172560 = {}
    # Getting the type of 'range' (line 1234)
    range_172555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 17), 'range', False)
    # Calling range(args, kwargs) (line 1234)
    range_call_result_172561 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 17), range_172555, *[int_172556, result_add_172559], **kwargs_172560)
    
    # Testing the type of a for loop iterable (line 1234)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1234, 8), range_call_result_172561)
    # Getting the type of the for loop variable (line 1234)
    for_loop_var_172562 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1234, 8), range_call_result_172561)
    # Assigning a type to the variable 'i' (line 1234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 8), 'i', for_loop_var_172562)
    # SSA begins for a for statement (line 1234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 1235):
    
    # Assigning a BinOp to a Subscript (line 1235):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1235)
    i_172563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 22), 'i')
    int_172564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 24), 'int')
    # Applying the binary operator '-' (line 1235)
    result_sub_172565 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 22), '-', i_172563, int_172564)
    
    # Getting the type of 'v' (line 1235)
    v_172566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 20), 'v')
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___172567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 20), v_172566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_172568 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 20), getitem___172567, result_sub_172565)
    
    int_172569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 28), 'int')
    # Getting the type of 'i' (line 1235)
    i_172570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 30), 'i')
    # Applying the binary operator '*' (line 1235)
    result_mul_172571 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 28), '*', int_172569, i_172570)
    
    int_172572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 34), 'int')
    # Applying the binary operator '-' (line 1235)
    result_sub_172573 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 28), '-', result_mul_172571, int_172572)
    
    # Getting the type of 'x' (line 1235)
    x_172574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 38), 'x')
    # Applying the binary operator '-' (line 1235)
    result_sub_172575 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 36), '-', result_sub_172573, x_172574)
    
    # Applying the binary operator '*' (line 1235)
    result_mul_172576 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 20), '*', subscript_call_result_172568, result_sub_172575)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1235)
    i_172577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 45), 'i')
    int_172578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 47), 'int')
    # Applying the binary operator '-' (line 1235)
    result_sub_172579 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 45), '-', i_172577, int_172578)
    
    # Getting the type of 'v' (line 1235)
    v_172580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 43), 'v')
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___172581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 43), v_172580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_172582 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 43), getitem___172581, result_sub_172579)
    
    # Getting the type of 'i' (line 1235)
    i_172583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 51), 'i')
    int_172584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 55), 'int')
    # Applying the binary operator '-' (line 1235)
    result_sub_172585 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 51), '-', i_172583, int_172584)
    
    # Applying the binary operator '*' (line 1235)
    result_mul_172586 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 43), '*', subscript_call_result_172582, result_sub_172585)
    
    # Applying the binary operator '-' (line 1235)
    result_sub_172587 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 20), '-', result_mul_172576, result_mul_172586)
    
    # Getting the type of 'i' (line 1235)
    i_172588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 59), 'i')
    # Applying the binary operator 'div' (line 1235)
    result_div_172589 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 19), 'div', result_sub_172587, i_172588)
    
    # Getting the type of 'v' (line 1235)
    v_172590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 12), 'v')
    # Getting the type of 'i' (line 1235)
    i_172591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 14), 'i')
    # Storing an element on a container (line 1235)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 12), v_172590, (i_172591, result_div_172589))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1232)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rollaxis(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'v' (line 1236)
    v_172594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 23), 'v', False)
    int_172595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 26), 'int')
    # Getting the type of 'v' (line 1236)
    v_172596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 29), 'v', False)
    # Obtaining the member 'ndim' of a type (line 1236)
    ndim_172597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 29), v_172596, 'ndim')
    # Processing the call keyword arguments (line 1236)
    kwargs_172598 = {}
    # Getting the type of 'np' (line 1236)
    np_172592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 11), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1236)
    rollaxis_172593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 11), np_172592, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1236)
    rollaxis_call_result_172599 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 11), rollaxis_172593, *[v_172594, int_172595, ndim_172597], **kwargs_172598)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'stypy_return_type', rollaxis_call_result_172599)
    
    # ################# End of 'lagvander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagvander' in the type store
    # Getting the type of 'stypy_return_type' (line 1177)
    stypy_return_type_172600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagvander'
    return stypy_return_type_172600

# Assigning a type to the variable 'lagvander' (line 1177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 0), 'lagvander', lagvander)

@norecursion
def lagvander2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagvander2d'
    module_type_store = module_type_store.open_function_context('lagvander2d', 1239, 0, False)
    
    # Passed parameters checking function
    lagvander2d.stypy_localization = localization
    lagvander2d.stypy_type_of_self = None
    lagvander2d.stypy_type_store = module_type_store
    lagvander2d.stypy_function_name = 'lagvander2d'
    lagvander2d.stypy_param_names_list = ['x', 'y', 'deg']
    lagvander2d.stypy_varargs_param_name = None
    lagvander2d.stypy_kwargs_param_name = None
    lagvander2d.stypy_call_defaults = defaults
    lagvander2d.stypy_call_varargs = varargs
    lagvander2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagvander2d', ['x', 'y', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagvander2d', localization, ['x', 'y', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagvander2d(...)' code ##################

    str_172601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., deg[1]*i + j] = L_i(x) * L_j(y),\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the degrees of\n    the Laguerre polynomials.\n\n    If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D Laguerre\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to\n        1-D arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    lagvander, lagvander3d. lagval2d, lagval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1289):
    
    # Assigning a ListComp to a Name (line 1289):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1289)
    deg_172606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 28), 'deg')
    comprehension_172607 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1289, 12), deg_172606)
    # Assigning a type to the variable 'd' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 12), 'd', comprehension_172607)
    
    # Call to int(...): (line 1289)
    # Processing the call arguments (line 1289)
    # Getting the type of 'd' (line 1289)
    d_172603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 16), 'd', False)
    # Processing the call keyword arguments (line 1289)
    kwargs_172604 = {}
    # Getting the type of 'int' (line 1289)
    int_172602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 12), 'int', False)
    # Calling int(args, kwargs) (line 1289)
    int_call_result_172605 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 12), int_172602, *[d_172603], **kwargs_172604)
    
    list_172608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1289, 12), list_172608, int_call_result_172605)
    # Assigning a type to the variable 'ideg' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'ideg', list_172608)
    
    # Assigning a ListComp to a Name (line 1290):
    
    # Assigning a ListComp to a Name (line 1290):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1290)
    # Processing the call arguments (line 1290)
    # Getting the type of 'ideg' (line 1290)
    ideg_172617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1290)
    deg_172618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 59), 'deg', False)
    # Processing the call keyword arguments (line 1290)
    kwargs_172619 = {}
    # Getting the type of 'zip' (line 1290)
    zip_172616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1290)
    zip_call_result_172620 = invoke(stypy.reporting.localization.Localization(__file__, 1290, 49), zip_172616, *[ideg_172617, deg_172618], **kwargs_172619)
    
    comprehension_172621 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1290, 16), zip_call_result_172620)
    # Assigning a type to the variable 'id' (line 1290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1290, 16), comprehension_172621))
    # Assigning a type to the variable 'd' (line 1290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1290, 16), comprehension_172621))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1290)
    id_172609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 16), 'id')
    # Getting the type of 'd' (line 1290)
    d_172610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 22), 'd')
    # Applying the binary operator '==' (line 1290)
    result_eq_172611 = python_operator(stypy.reporting.localization.Localization(__file__, 1290, 16), '==', id_172609, d_172610)
    
    
    # Getting the type of 'id' (line 1290)
    id_172612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 28), 'id')
    int_172613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1290, 34), 'int')
    # Applying the binary operator '>=' (line 1290)
    result_ge_172614 = python_operator(stypy.reporting.localization.Localization(__file__, 1290, 28), '>=', id_172612, int_172613)
    
    # Applying the binary operator 'and' (line 1290)
    result_and_keyword_172615 = python_operator(stypy.reporting.localization.Localization(__file__, 1290, 16), 'and', result_eq_172611, result_ge_172614)
    
    list_172622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1290, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1290, 16), list_172622, result_and_keyword_172615)
    # Assigning a type to the variable 'is_valid' (line 1290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 4), 'is_valid', list_172622)
    
    
    # Getting the type of 'is_valid' (line 1291)
    is_valid_172623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1291)
    list_172624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1291)
    # Adding element type (line 1291)
    int_172625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 19), list_172624, int_172625)
    # Adding element type (line 1291)
    int_172626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 19), list_172624, int_172626)
    
    # Applying the binary operator '!=' (line 1291)
    result_ne_172627 = python_operator(stypy.reporting.localization.Localization(__file__, 1291, 7), '!=', is_valid_172623, list_172624)
    
    # Testing the type of an if condition (line 1291)
    if_condition_172628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1291, 4), result_ne_172627)
    # Assigning a type to the variable 'if_condition_172628' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'if_condition_172628', if_condition_172628)
    # SSA begins for if statement (line 1291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1292)
    # Processing the call arguments (line 1292)
    str_172630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1292)
    kwargs_172631 = {}
    # Getting the type of 'ValueError' (line 1292)
    ValueError_172629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1292)
    ValueError_call_result_172632 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 14), ValueError_172629, *[str_172630], **kwargs_172631)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1292, 8), ValueError_call_result_172632, 'raise parameter', BaseException)
    # SSA join for if statement (line 1291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1293):
    
    # Assigning a Subscript to a Name (line 1293):
    
    # Obtaining the type of the subscript
    int_172633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1293, 4), 'int')
    # Getting the type of 'ideg' (line 1293)
    ideg_172634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1293)
    getitem___172635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1293, 4), ideg_172634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1293)
    subscript_call_result_172636 = invoke(stypy.reporting.localization.Localization(__file__, 1293, 4), getitem___172635, int_172633)
    
    # Assigning a type to the variable 'tuple_var_assignment_170798' (line 1293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 4), 'tuple_var_assignment_170798', subscript_call_result_172636)
    
    # Assigning a Subscript to a Name (line 1293):
    
    # Obtaining the type of the subscript
    int_172637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1293, 4), 'int')
    # Getting the type of 'ideg' (line 1293)
    ideg_172638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1293)
    getitem___172639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1293, 4), ideg_172638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1293)
    subscript_call_result_172640 = invoke(stypy.reporting.localization.Localization(__file__, 1293, 4), getitem___172639, int_172637)
    
    # Assigning a type to the variable 'tuple_var_assignment_170799' (line 1293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 4), 'tuple_var_assignment_170799', subscript_call_result_172640)
    
    # Assigning a Name to a Name (line 1293):
    # Getting the type of 'tuple_var_assignment_170798' (line 1293)
    tuple_var_assignment_170798_172641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 4), 'tuple_var_assignment_170798')
    # Assigning a type to the variable 'degx' (line 1293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 4), 'degx', tuple_var_assignment_170798_172641)
    
    # Assigning a Name to a Name (line 1293):
    # Getting the type of 'tuple_var_assignment_170799' (line 1293)
    tuple_var_assignment_170799_172642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 4), 'tuple_var_assignment_170799')
    # Assigning a type to the variable 'degy' (line 1293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 10), 'degy', tuple_var_assignment_170799_172642)
    
    # Assigning a BinOp to a Tuple (line 1294):
    
    # Assigning a Subscript to a Name (line 1294):
    
    # Obtaining the type of the subscript
    int_172643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 4), 'int')
    
    # Call to array(...): (line 1294)
    # Processing the call arguments (line 1294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_172646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    # Getting the type of 'x' (line 1294)
    x_172647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 21), tuple_172646, x_172647)
    # Adding element type (line 1294)
    # Getting the type of 'y' (line 1294)
    y_172648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 21), tuple_172646, y_172648)
    
    # Processing the call keyword arguments (line 1294)
    int_172649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 33), 'int')
    keyword_172650 = int_172649
    kwargs_172651 = {'copy': keyword_172650}
    # Getting the type of 'np' (line 1294)
    np_172644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1294)
    array_172645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 11), np_172644, 'array')
    # Calling array(args, kwargs) (line 1294)
    array_call_result_172652 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 11), array_172645, *[tuple_172646], **kwargs_172651)
    
    float_172653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 38), 'float')
    # Applying the binary operator '+' (line 1294)
    result_add_172654 = python_operator(stypy.reporting.localization.Localization(__file__, 1294, 11), '+', array_call_result_172652, float_172653)
    
    # Obtaining the member '__getitem__' of a type (line 1294)
    getitem___172655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 4), result_add_172654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1294)
    subscript_call_result_172656 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 4), getitem___172655, int_172643)
    
    # Assigning a type to the variable 'tuple_var_assignment_170800' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_170800', subscript_call_result_172656)
    
    # Assigning a Subscript to a Name (line 1294):
    
    # Obtaining the type of the subscript
    int_172657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 4), 'int')
    
    # Call to array(...): (line 1294)
    # Processing the call arguments (line 1294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_172660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    # Getting the type of 'x' (line 1294)
    x_172661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 21), tuple_172660, x_172661)
    # Adding element type (line 1294)
    # Getting the type of 'y' (line 1294)
    y_172662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 21), tuple_172660, y_172662)
    
    # Processing the call keyword arguments (line 1294)
    int_172663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 33), 'int')
    keyword_172664 = int_172663
    kwargs_172665 = {'copy': keyword_172664}
    # Getting the type of 'np' (line 1294)
    np_172658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1294)
    array_172659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 11), np_172658, 'array')
    # Calling array(args, kwargs) (line 1294)
    array_call_result_172666 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 11), array_172659, *[tuple_172660], **kwargs_172665)
    
    float_172667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 38), 'float')
    # Applying the binary operator '+' (line 1294)
    result_add_172668 = python_operator(stypy.reporting.localization.Localization(__file__, 1294, 11), '+', array_call_result_172666, float_172667)
    
    # Obtaining the member '__getitem__' of a type (line 1294)
    getitem___172669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 4), result_add_172668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1294)
    subscript_call_result_172670 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 4), getitem___172669, int_172657)
    
    # Assigning a type to the variable 'tuple_var_assignment_170801' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_170801', subscript_call_result_172670)
    
    # Assigning a Name to a Name (line 1294):
    # Getting the type of 'tuple_var_assignment_170800' (line 1294)
    tuple_var_assignment_170800_172671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_170800')
    # Assigning a type to the variable 'x' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'x', tuple_var_assignment_170800_172671)
    
    # Assigning a Name to a Name (line 1294):
    # Getting the type of 'tuple_var_assignment_170801' (line 1294)
    tuple_var_assignment_170801_172672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_170801')
    # Assigning a type to the variable 'y' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 7), 'y', tuple_var_assignment_170801_172672)
    
    # Assigning a Call to a Name (line 1296):
    
    # Assigning a Call to a Name (line 1296):
    
    # Call to lagvander(...): (line 1296)
    # Processing the call arguments (line 1296)
    # Getting the type of 'x' (line 1296)
    x_172674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 19), 'x', False)
    # Getting the type of 'degx' (line 1296)
    degx_172675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 22), 'degx', False)
    # Processing the call keyword arguments (line 1296)
    kwargs_172676 = {}
    # Getting the type of 'lagvander' (line 1296)
    lagvander_172673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 9), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1296)
    lagvander_call_result_172677 = invoke(stypy.reporting.localization.Localization(__file__, 1296, 9), lagvander_172673, *[x_172674, degx_172675], **kwargs_172676)
    
    # Assigning a type to the variable 'vx' (line 1296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1296, 4), 'vx', lagvander_call_result_172677)
    
    # Assigning a Call to a Name (line 1297):
    
    # Assigning a Call to a Name (line 1297):
    
    # Call to lagvander(...): (line 1297)
    # Processing the call arguments (line 1297)
    # Getting the type of 'y' (line 1297)
    y_172679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 19), 'y', False)
    # Getting the type of 'degy' (line 1297)
    degy_172680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 22), 'degy', False)
    # Processing the call keyword arguments (line 1297)
    kwargs_172681 = {}
    # Getting the type of 'lagvander' (line 1297)
    lagvander_172678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 9), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1297)
    lagvander_call_result_172682 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 9), lagvander_172678, *[y_172679, degy_172680], **kwargs_172681)
    
    # Assigning a type to the variable 'vy' (line 1297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1297, 4), 'vy', lagvander_call_result_172682)
    
    # Assigning a BinOp to a Name (line 1298):
    
    # Assigning a BinOp to a Name (line 1298):
    
    # Obtaining the type of the subscript
    Ellipsis_172683 = Ellipsis
    # Getting the type of 'None' (line 1298)
    None_172684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 16), 'None')
    # Getting the type of 'vx' (line 1298)
    vx_172685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1298)
    getitem___172686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 8), vx_172685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1298)
    subscript_call_result_172687 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 8), getitem___172686, (Ellipsis_172683, None_172684))
    
    
    # Obtaining the type of the subscript
    Ellipsis_172688 = Ellipsis
    # Getting the type of 'None' (line 1298)
    None_172689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 30), 'None')
    slice_172690 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1298, 22), None, None, None)
    # Getting the type of 'vy' (line 1298)
    vy_172691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 22), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1298)
    getitem___172692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 22), vy_172691, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1298)
    subscript_call_result_172693 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 22), getitem___172692, (Ellipsis_172688, None_172689, slice_172690))
    
    # Applying the binary operator '*' (line 1298)
    result_mul_172694 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 8), '*', subscript_call_result_172687, subscript_call_result_172693)
    
    # Assigning a type to the variable 'v' (line 1298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1298, 4), 'v', result_mul_172694)
    
    # Call to reshape(...): (line 1299)
    # Processing the call arguments (line 1299)
    
    # Obtaining the type of the subscript
    int_172697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1299, 30), 'int')
    slice_172698 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1299, 21), None, int_172697, None)
    # Getting the type of 'v' (line 1299)
    v_172699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1299)
    shape_172700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1299, 21), v_172699, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1299)
    getitem___172701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1299, 21), shape_172700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1299)
    subscript_call_result_172702 = invoke(stypy.reporting.localization.Localization(__file__, 1299, 21), getitem___172701, slice_172698)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1299)
    tuple_172703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1299, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1299)
    # Adding element type (line 1299)
    int_172704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1299, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1299, 37), tuple_172703, int_172704)
    
    # Applying the binary operator '+' (line 1299)
    result_add_172705 = python_operator(stypy.reporting.localization.Localization(__file__, 1299, 21), '+', subscript_call_result_172702, tuple_172703)
    
    # Processing the call keyword arguments (line 1299)
    kwargs_172706 = {}
    # Getting the type of 'v' (line 1299)
    v_172695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1299)
    reshape_172696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1299, 11), v_172695, 'reshape')
    # Calling reshape(args, kwargs) (line 1299)
    reshape_call_result_172707 = invoke(stypy.reporting.localization.Localization(__file__, 1299, 11), reshape_172696, *[result_add_172705], **kwargs_172706)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1299, 4), 'stypy_return_type', reshape_call_result_172707)
    
    # ################# End of 'lagvander2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagvander2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1239)
    stypy_return_type_172708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagvander2d'
    return stypy_return_type_172708

# Assigning a type to the variable 'lagvander2d' (line 1239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 0), 'lagvander2d', lagvander2d)

@norecursion
def lagvander3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagvander3d'
    module_type_store = module_type_store.open_function_context('lagvander3d', 1302, 0, False)
    
    # Passed parameters checking function
    lagvander3d.stypy_localization = localization
    lagvander3d.stypy_type_of_self = None
    lagvander3d.stypy_type_store = module_type_store
    lagvander3d.stypy_function_name = 'lagvander3d'
    lagvander3d.stypy_param_names_list = ['x', 'y', 'z', 'deg']
    lagvander3d.stypy_varargs_param_name = None
    lagvander3d.stypy_kwargs_param_name = None
    lagvander3d.stypy_call_defaults = defaults
    lagvander3d.stypy_call_varargs = varargs
    lagvander3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagvander3d', ['x', 'y', 'z', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagvander3d', localization, ['x', 'y', 'z', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagvander3d(...)' code ##################

    str_172709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1352, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the degrees of the Laguerre polynomials.\n\n    If ``V = lagvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D Laguerre\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    lagvander, lagvander3d. lagval2d, lagval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1353):
    
    # Assigning a ListComp to a Name (line 1353):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1353)
    deg_172714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 28), 'deg')
    comprehension_172715 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1353, 12), deg_172714)
    # Assigning a type to the variable 'd' (line 1353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1353, 12), 'd', comprehension_172715)
    
    # Call to int(...): (line 1353)
    # Processing the call arguments (line 1353)
    # Getting the type of 'd' (line 1353)
    d_172711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 16), 'd', False)
    # Processing the call keyword arguments (line 1353)
    kwargs_172712 = {}
    # Getting the type of 'int' (line 1353)
    int_172710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 12), 'int', False)
    # Calling int(args, kwargs) (line 1353)
    int_call_result_172713 = invoke(stypy.reporting.localization.Localization(__file__, 1353, 12), int_172710, *[d_172711], **kwargs_172712)
    
    list_172716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1353, 12), list_172716, int_call_result_172713)
    # Assigning a type to the variable 'ideg' (line 1353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1353, 4), 'ideg', list_172716)
    
    # Assigning a ListComp to a Name (line 1354):
    
    # Assigning a ListComp to a Name (line 1354):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1354)
    # Processing the call arguments (line 1354)
    # Getting the type of 'ideg' (line 1354)
    ideg_172725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1354)
    deg_172726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 59), 'deg', False)
    # Processing the call keyword arguments (line 1354)
    kwargs_172727 = {}
    # Getting the type of 'zip' (line 1354)
    zip_172724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1354)
    zip_call_result_172728 = invoke(stypy.reporting.localization.Localization(__file__, 1354, 49), zip_172724, *[ideg_172725, deg_172726], **kwargs_172727)
    
    comprehension_172729 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 16), zip_call_result_172728)
    # Assigning a type to the variable 'id' (line 1354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1354, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 16), comprehension_172729))
    # Assigning a type to the variable 'd' (line 1354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1354, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 16), comprehension_172729))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1354)
    id_172717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 16), 'id')
    # Getting the type of 'd' (line 1354)
    d_172718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 22), 'd')
    # Applying the binary operator '==' (line 1354)
    result_eq_172719 = python_operator(stypy.reporting.localization.Localization(__file__, 1354, 16), '==', id_172717, d_172718)
    
    
    # Getting the type of 'id' (line 1354)
    id_172720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 28), 'id')
    int_172721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 34), 'int')
    # Applying the binary operator '>=' (line 1354)
    result_ge_172722 = python_operator(stypy.reporting.localization.Localization(__file__, 1354, 28), '>=', id_172720, int_172721)
    
    # Applying the binary operator 'and' (line 1354)
    result_and_keyword_172723 = python_operator(stypy.reporting.localization.Localization(__file__, 1354, 16), 'and', result_eq_172719, result_ge_172722)
    
    list_172730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 16), list_172730, result_and_keyword_172723)
    # Assigning a type to the variable 'is_valid' (line 1354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1354, 4), 'is_valid', list_172730)
    
    
    # Getting the type of 'is_valid' (line 1355)
    is_valid_172731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1355)
    list_172732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1355)
    # Adding element type (line 1355)
    int_172733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 19), list_172732, int_172733)
    # Adding element type (line 1355)
    int_172734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 19), list_172732, int_172734)
    # Adding element type (line 1355)
    int_172735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 19), list_172732, int_172735)
    
    # Applying the binary operator '!=' (line 1355)
    result_ne_172736 = python_operator(stypy.reporting.localization.Localization(__file__, 1355, 7), '!=', is_valid_172731, list_172732)
    
    # Testing the type of an if condition (line 1355)
    if_condition_172737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1355, 4), result_ne_172736)
    # Assigning a type to the variable 'if_condition_172737' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'if_condition_172737', if_condition_172737)
    # SSA begins for if statement (line 1355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1356)
    # Processing the call arguments (line 1356)
    str_172739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1356)
    kwargs_172740 = {}
    # Getting the type of 'ValueError' (line 1356)
    ValueError_172738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1356)
    ValueError_call_result_172741 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 14), ValueError_172738, *[str_172739], **kwargs_172740)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1356, 8), ValueError_call_result_172741, 'raise parameter', BaseException)
    # SSA join for if statement (line 1355)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1357):
    
    # Assigning a Subscript to a Name (line 1357):
    
    # Obtaining the type of the subscript
    int_172742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1357, 4), 'int')
    # Getting the type of 'ideg' (line 1357)
    ideg_172743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1357)
    getitem___172744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1357, 4), ideg_172743, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1357)
    subscript_call_result_172745 = invoke(stypy.reporting.localization.Localization(__file__, 1357, 4), getitem___172744, int_172742)
    
    # Assigning a type to the variable 'tuple_var_assignment_170802' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'tuple_var_assignment_170802', subscript_call_result_172745)
    
    # Assigning a Subscript to a Name (line 1357):
    
    # Obtaining the type of the subscript
    int_172746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1357, 4), 'int')
    # Getting the type of 'ideg' (line 1357)
    ideg_172747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1357)
    getitem___172748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1357, 4), ideg_172747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1357)
    subscript_call_result_172749 = invoke(stypy.reporting.localization.Localization(__file__, 1357, 4), getitem___172748, int_172746)
    
    # Assigning a type to the variable 'tuple_var_assignment_170803' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'tuple_var_assignment_170803', subscript_call_result_172749)
    
    # Assigning a Subscript to a Name (line 1357):
    
    # Obtaining the type of the subscript
    int_172750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1357, 4), 'int')
    # Getting the type of 'ideg' (line 1357)
    ideg_172751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1357)
    getitem___172752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1357, 4), ideg_172751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1357)
    subscript_call_result_172753 = invoke(stypy.reporting.localization.Localization(__file__, 1357, 4), getitem___172752, int_172750)
    
    # Assigning a type to the variable 'tuple_var_assignment_170804' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'tuple_var_assignment_170804', subscript_call_result_172753)
    
    # Assigning a Name to a Name (line 1357):
    # Getting the type of 'tuple_var_assignment_170802' (line 1357)
    tuple_var_assignment_170802_172754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'tuple_var_assignment_170802')
    # Assigning a type to the variable 'degx' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'degx', tuple_var_assignment_170802_172754)
    
    # Assigning a Name to a Name (line 1357):
    # Getting the type of 'tuple_var_assignment_170803' (line 1357)
    tuple_var_assignment_170803_172755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'tuple_var_assignment_170803')
    # Assigning a type to the variable 'degy' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 10), 'degy', tuple_var_assignment_170803_172755)
    
    # Assigning a Name to a Name (line 1357):
    # Getting the type of 'tuple_var_assignment_170804' (line 1357)
    tuple_var_assignment_170804_172756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'tuple_var_assignment_170804')
    # Assigning a type to the variable 'degz' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 16), 'degz', tuple_var_assignment_170804_172756)
    
    # Assigning a BinOp to a Tuple (line 1358):
    
    # Assigning a Subscript to a Name (line 1358):
    
    # Obtaining the type of the subscript
    int_172757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 4), 'int')
    
    # Call to array(...): (line 1358)
    # Processing the call arguments (line 1358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1358)
    tuple_172760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1358)
    # Adding element type (line 1358)
    # Getting the type of 'x' (line 1358)
    x_172761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172760, x_172761)
    # Adding element type (line 1358)
    # Getting the type of 'y' (line 1358)
    y_172762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172760, y_172762)
    # Adding element type (line 1358)
    # Getting the type of 'z' (line 1358)
    z_172763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172760, z_172763)
    
    # Processing the call keyword arguments (line 1358)
    int_172764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 39), 'int')
    keyword_172765 = int_172764
    kwargs_172766 = {'copy': keyword_172765}
    # Getting the type of 'np' (line 1358)
    np_172758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1358)
    array_172759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 14), np_172758, 'array')
    # Calling array(args, kwargs) (line 1358)
    array_call_result_172767 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 14), array_172759, *[tuple_172760], **kwargs_172766)
    
    float_172768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 44), 'float')
    # Applying the binary operator '+' (line 1358)
    result_add_172769 = python_operator(stypy.reporting.localization.Localization(__file__, 1358, 14), '+', array_call_result_172767, float_172768)
    
    # Obtaining the member '__getitem__' of a type (line 1358)
    getitem___172770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 4), result_add_172769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1358)
    subscript_call_result_172771 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 4), getitem___172770, int_172757)
    
    # Assigning a type to the variable 'tuple_var_assignment_170805' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_170805', subscript_call_result_172771)
    
    # Assigning a Subscript to a Name (line 1358):
    
    # Obtaining the type of the subscript
    int_172772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 4), 'int')
    
    # Call to array(...): (line 1358)
    # Processing the call arguments (line 1358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1358)
    tuple_172775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1358)
    # Adding element type (line 1358)
    # Getting the type of 'x' (line 1358)
    x_172776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172775, x_172776)
    # Adding element type (line 1358)
    # Getting the type of 'y' (line 1358)
    y_172777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172775, y_172777)
    # Adding element type (line 1358)
    # Getting the type of 'z' (line 1358)
    z_172778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172775, z_172778)
    
    # Processing the call keyword arguments (line 1358)
    int_172779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 39), 'int')
    keyword_172780 = int_172779
    kwargs_172781 = {'copy': keyword_172780}
    # Getting the type of 'np' (line 1358)
    np_172773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1358)
    array_172774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 14), np_172773, 'array')
    # Calling array(args, kwargs) (line 1358)
    array_call_result_172782 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 14), array_172774, *[tuple_172775], **kwargs_172781)
    
    float_172783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 44), 'float')
    # Applying the binary operator '+' (line 1358)
    result_add_172784 = python_operator(stypy.reporting.localization.Localization(__file__, 1358, 14), '+', array_call_result_172782, float_172783)
    
    # Obtaining the member '__getitem__' of a type (line 1358)
    getitem___172785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 4), result_add_172784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1358)
    subscript_call_result_172786 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 4), getitem___172785, int_172772)
    
    # Assigning a type to the variable 'tuple_var_assignment_170806' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_170806', subscript_call_result_172786)
    
    # Assigning a Subscript to a Name (line 1358):
    
    # Obtaining the type of the subscript
    int_172787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 4), 'int')
    
    # Call to array(...): (line 1358)
    # Processing the call arguments (line 1358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1358)
    tuple_172790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1358)
    # Adding element type (line 1358)
    # Getting the type of 'x' (line 1358)
    x_172791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172790, x_172791)
    # Adding element type (line 1358)
    # Getting the type of 'y' (line 1358)
    y_172792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172790, y_172792)
    # Adding element type (line 1358)
    # Getting the type of 'z' (line 1358)
    z_172793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 24), tuple_172790, z_172793)
    
    # Processing the call keyword arguments (line 1358)
    int_172794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 39), 'int')
    keyword_172795 = int_172794
    kwargs_172796 = {'copy': keyword_172795}
    # Getting the type of 'np' (line 1358)
    np_172788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1358)
    array_172789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 14), np_172788, 'array')
    # Calling array(args, kwargs) (line 1358)
    array_call_result_172797 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 14), array_172789, *[tuple_172790], **kwargs_172796)
    
    float_172798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 44), 'float')
    # Applying the binary operator '+' (line 1358)
    result_add_172799 = python_operator(stypy.reporting.localization.Localization(__file__, 1358, 14), '+', array_call_result_172797, float_172798)
    
    # Obtaining the member '__getitem__' of a type (line 1358)
    getitem___172800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 4), result_add_172799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1358)
    subscript_call_result_172801 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 4), getitem___172800, int_172787)
    
    # Assigning a type to the variable 'tuple_var_assignment_170807' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_170807', subscript_call_result_172801)
    
    # Assigning a Name to a Name (line 1358):
    # Getting the type of 'tuple_var_assignment_170805' (line 1358)
    tuple_var_assignment_170805_172802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_170805')
    # Assigning a type to the variable 'x' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'x', tuple_var_assignment_170805_172802)
    
    # Assigning a Name to a Name (line 1358):
    # Getting the type of 'tuple_var_assignment_170806' (line 1358)
    tuple_var_assignment_170806_172803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_170806')
    # Assigning a type to the variable 'y' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 7), 'y', tuple_var_assignment_170806_172803)
    
    # Assigning a Name to a Name (line 1358):
    # Getting the type of 'tuple_var_assignment_170807' (line 1358)
    tuple_var_assignment_170807_172804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_170807')
    # Assigning a type to the variable 'z' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 10), 'z', tuple_var_assignment_170807_172804)
    
    # Assigning a Call to a Name (line 1360):
    
    # Assigning a Call to a Name (line 1360):
    
    # Call to lagvander(...): (line 1360)
    # Processing the call arguments (line 1360)
    # Getting the type of 'x' (line 1360)
    x_172806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 19), 'x', False)
    # Getting the type of 'degx' (line 1360)
    degx_172807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 22), 'degx', False)
    # Processing the call keyword arguments (line 1360)
    kwargs_172808 = {}
    # Getting the type of 'lagvander' (line 1360)
    lagvander_172805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 9), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1360)
    lagvander_call_result_172809 = invoke(stypy.reporting.localization.Localization(__file__, 1360, 9), lagvander_172805, *[x_172806, degx_172807], **kwargs_172808)
    
    # Assigning a type to the variable 'vx' (line 1360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1360, 4), 'vx', lagvander_call_result_172809)
    
    # Assigning a Call to a Name (line 1361):
    
    # Assigning a Call to a Name (line 1361):
    
    # Call to lagvander(...): (line 1361)
    # Processing the call arguments (line 1361)
    # Getting the type of 'y' (line 1361)
    y_172811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 19), 'y', False)
    # Getting the type of 'degy' (line 1361)
    degy_172812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 22), 'degy', False)
    # Processing the call keyword arguments (line 1361)
    kwargs_172813 = {}
    # Getting the type of 'lagvander' (line 1361)
    lagvander_172810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 9), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1361)
    lagvander_call_result_172814 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 9), lagvander_172810, *[y_172811, degy_172812], **kwargs_172813)
    
    # Assigning a type to the variable 'vy' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 4), 'vy', lagvander_call_result_172814)
    
    # Assigning a Call to a Name (line 1362):
    
    # Assigning a Call to a Name (line 1362):
    
    # Call to lagvander(...): (line 1362)
    # Processing the call arguments (line 1362)
    # Getting the type of 'z' (line 1362)
    z_172816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 19), 'z', False)
    # Getting the type of 'degz' (line 1362)
    degz_172817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 22), 'degz', False)
    # Processing the call keyword arguments (line 1362)
    kwargs_172818 = {}
    # Getting the type of 'lagvander' (line 1362)
    lagvander_172815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 9), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1362)
    lagvander_call_result_172819 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 9), lagvander_172815, *[z_172816, degz_172817], **kwargs_172818)
    
    # Assigning a type to the variable 'vz' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'vz', lagvander_call_result_172819)
    
    # Assigning a BinOp to a Name (line 1363):
    
    # Assigning a BinOp to a Name (line 1363):
    
    # Obtaining the type of the subscript
    Ellipsis_172820 = Ellipsis
    # Getting the type of 'None' (line 1363)
    None_172821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 16), 'None')
    # Getting the type of 'None' (line 1363)
    None_172822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 22), 'None')
    # Getting the type of 'vx' (line 1363)
    vx_172823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1363)
    getitem___172824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 8), vx_172823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1363)
    subscript_call_result_172825 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 8), getitem___172824, (Ellipsis_172820, None_172821, None_172822))
    
    
    # Obtaining the type of the subscript
    Ellipsis_172826 = Ellipsis
    # Getting the type of 'None' (line 1363)
    None_172827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 36), 'None')
    slice_172828 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1363, 28), None, None, None)
    # Getting the type of 'None' (line 1363)
    None_172829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 44), 'None')
    # Getting the type of 'vy' (line 1363)
    vy_172830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 28), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1363)
    getitem___172831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 28), vy_172830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1363)
    subscript_call_result_172832 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 28), getitem___172831, (Ellipsis_172826, None_172827, slice_172828, None_172829))
    
    # Applying the binary operator '*' (line 1363)
    result_mul_172833 = python_operator(stypy.reporting.localization.Localization(__file__, 1363, 8), '*', subscript_call_result_172825, subscript_call_result_172832)
    
    
    # Obtaining the type of the subscript
    Ellipsis_172834 = Ellipsis
    # Getting the type of 'None' (line 1363)
    None_172835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 58), 'None')
    # Getting the type of 'None' (line 1363)
    None_172836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 64), 'None')
    slice_172837 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1363, 50), None, None, None)
    # Getting the type of 'vz' (line 1363)
    vz_172838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 50), 'vz')
    # Obtaining the member '__getitem__' of a type (line 1363)
    getitem___172839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 50), vz_172838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1363)
    subscript_call_result_172840 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 50), getitem___172839, (Ellipsis_172834, None_172835, None_172836, slice_172837))
    
    # Applying the binary operator '*' (line 1363)
    result_mul_172841 = python_operator(stypy.reporting.localization.Localization(__file__, 1363, 49), '*', result_mul_172833, subscript_call_result_172840)
    
    # Assigning a type to the variable 'v' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'v', result_mul_172841)
    
    # Call to reshape(...): (line 1364)
    # Processing the call arguments (line 1364)
    
    # Obtaining the type of the subscript
    int_172844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 30), 'int')
    slice_172845 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1364, 21), None, int_172844, None)
    # Getting the type of 'v' (line 1364)
    v_172846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1364)
    shape_172847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 21), v_172846, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1364)
    getitem___172848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 21), shape_172847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1364)
    subscript_call_result_172849 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 21), getitem___172848, slice_172845)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1364)
    tuple_172850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1364)
    # Adding element type (line 1364)
    int_172851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 37), tuple_172850, int_172851)
    
    # Applying the binary operator '+' (line 1364)
    result_add_172852 = python_operator(stypy.reporting.localization.Localization(__file__, 1364, 21), '+', subscript_call_result_172849, tuple_172850)
    
    # Processing the call keyword arguments (line 1364)
    kwargs_172853 = {}
    # Getting the type of 'v' (line 1364)
    v_172842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1364)
    reshape_172843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 11), v_172842, 'reshape')
    # Calling reshape(args, kwargs) (line 1364)
    reshape_call_result_172854 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 11), reshape_172843, *[result_add_172852], **kwargs_172853)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 4), 'stypy_return_type', reshape_call_result_172854)
    
    # ################# End of 'lagvander3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagvander3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1302)
    stypy_return_type_172855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172855)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagvander3d'
    return stypy_return_type_172855

# Assigning a type to the variable 'lagvander3d' (line 1302)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 0), 'lagvander3d', lagvander3d)

@norecursion
def lagfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1367)
    None_172856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 28), 'None')
    # Getting the type of 'False' (line 1367)
    False_172857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 39), 'False')
    # Getting the type of 'None' (line 1367)
    None_172858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 48), 'None')
    defaults = [None_172856, False_172857, None_172858]
    # Create a new context for function 'lagfit'
    module_type_store = module_type_store.open_function_context('lagfit', 1367, 0, False)
    
    # Passed parameters checking function
    lagfit.stypy_localization = localization
    lagfit.stypy_type_of_self = None
    lagfit.stypy_type_store = module_type_store
    lagfit.stypy_function_name = 'lagfit'
    lagfit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w']
    lagfit.stypy_varargs_param_name = None
    lagfit.stypy_kwargs_param_name = None
    lagfit.stypy_call_defaults = defaults
    lagfit.stypy_call_varargs = varargs
    lagfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagfit', ['x', 'y', 'deg', 'rcond', 'full', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagfit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagfit(...)' code ##################

    str_172859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, (-1)), 'str', '\n    Least squares fit of Laguerre series to data.\n\n    Return the coefficients of a Laguerre series of degree `deg` that is the\n    least squares fit to the data values `y` given at points `x`. If `y` is\n    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple\n    fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),\n\n    where `n` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For Numpy versions >= 1.11 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the contribution of each point\n        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n        weights are chosen so that the errors of the products ``w[i]*y[i]``\n        all have the same variance.  The default value is None.\n\n    Returns\n    -------\n    coef : ndarray, shape (M,) or (M, K)\n        Laguerre coefficients ordered from low to high. If `y` was 2-D,\n        the coefficients for the data in column k  of `y` are in column\n        `k`.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if `full` = True\n\n        resid -- sum of squared residuals of the least squares fit\n        rank -- the numerical rank of the scaled Vandermonde matrix\n        sv -- singular values of the scaled Vandermonde matrix\n        rcond -- value of `rcond`.\n\n        For more details, see `linalg.lstsq`.\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if `full` = False.  The\n        warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', RankWarning)\n\n    See Also\n    --------\n    chebfit, legfit, polyfit, hermfit, hermefit\n    lagval : Evaluates a Laguerre series.\n    lagvander : pseudo Vandermonde matrix of Laguerre series.\n    lagweight : Laguerre weight function.\n    linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the Laguerre series `p` that\n    minimizes the sum of the weighted squared errors\n\n    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where the :math:`w_j` are the weights. This problem is solved by\n    setting up as the (typically) overdetermined matrix equation\n\n    .. math:: V(x) * c = w * y,\n\n    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the\n    coefficients to be solved for, `w` are the weights, and `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of `V`.\n\n    If some of the singular values of `V` are so small that they are\n    neglected, then a `RankWarning` will be issued. This means that the\n    coefficient values may be poorly determined. Using a lower order fit\n    will usually get rid of the warning.  The `rcond` parameter can also be\n    set to a value smaller than its default, but the resulting fit may be\n    spurious and have large contributions from roundoff error.\n\n    Fits using Laguerre series are probably most useful when the data can\n    be approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the Laguerre\n    weight. In that case the weight ``sqrt(w(x[i])`` should be used\n    together with data values ``y[i]/sqrt(w(x[i])``. The weight function is\n    available as `lagweight`.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           http://en.wikipedia.org/wiki/Curve_fitting\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagfit, lagval\n    >>> x = np.linspace(0, 10)\n    >>> err = np.random.randn(len(x))/10\n    >>> y = lagval(x, [1, 2, 3]) + err\n    >>> lagfit(x, y, 2)\n    array([ 0.96971004,  2.00193749,  3.00288744])\n\n    ')
    
    # Assigning a BinOp to a Name (line 1491):
    
    # Assigning a BinOp to a Name (line 1491):
    
    # Call to asarray(...): (line 1491)
    # Processing the call arguments (line 1491)
    # Getting the type of 'x' (line 1491)
    x_172862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 19), 'x', False)
    # Processing the call keyword arguments (line 1491)
    kwargs_172863 = {}
    # Getting the type of 'np' (line 1491)
    np_172860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1491)
    asarray_172861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1491, 8), np_172860, 'asarray')
    # Calling asarray(args, kwargs) (line 1491)
    asarray_call_result_172864 = invoke(stypy.reporting.localization.Localization(__file__, 1491, 8), asarray_172861, *[x_172862], **kwargs_172863)
    
    float_172865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1491, 24), 'float')
    # Applying the binary operator '+' (line 1491)
    result_add_172866 = python_operator(stypy.reporting.localization.Localization(__file__, 1491, 8), '+', asarray_call_result_172864, float_172865)
    
    # Assigning a type to the variable 'x' (line 1491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1491, 4), 'x', result_add_172866)
    
    # Assigning a BinOp to a Name (line 1492):
    
    # Assigning a BinOp to a Name (line 1492):
    
    # Call to asarray(...): (line 1492)
    # Processing the call arguments (line 1492)
    # Getting the type of 'y' (line 1492)
    y_172869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 19), 'y', False)
    # Processing the call keyword arguments (line 1492)
    kwargs_172870 = {}
    # Getting the type of 'np' (line 1492)
    np_172867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1492)
    asarray_172868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1492, 8), np_172867, 'asarray')
    # Calling asarray(args, kwargs) (line 1492)
    asarray_call_result_172871 = invoke(stypy.reporting.localization.Localization(__file__, 1492, 8), asarray_172868, *[y_172869], **kwargs_172870)
    
    float_172872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1492, 24), 'float')
    # Applying the binary operator '+' (line 1492)
    result_add_172873 = python_operator(stypy.reporting.localization.Localization(__file__, 1492, 8), '+', asarray_call_result_172871, float_172872)
    
    # Assigning a type to the variable 'y' (line 1492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1492, 4), 'y', result_add_172873)
    
    # Assigning a Call to a Name (line 1493):
    
    # Assigning a Call to a Name (line 1493):
    
    # Call to asarray(...): (line 1493)
    # Processing the call arguments (line 1493)
    # Getting the type of 'deg' (line 1493)
    deg_172876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 21), 'deg', False)
    # Processing the call keyword arguments (line 1493)
    kwargs_172877 = {}
    # Getting the type of 'np' (line 1493)
    np_172874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1493)
    asarray_172875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1493, 10), np_172874, 'asarray')
    # Calling asarray(args, kwargs) (line 1493)
    asarray_call_result_172878 = invoke(stypy.reporting.localization.Localization(__file__, 1493, 10), asarray_172875, *[deg_172876], **kwargs_172877)
    
    # Assigning a type to the variable 'deg' (line 1493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1493, 4), 'deg', asarray_call_result_172878)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deg' (line 1496)
    deg_172879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1496)
    ndim_172880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1496, 7), deg_172879, 'ndim')
    int_172881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, 18), 'int')
    # Applying the binary operator '>' (line 1496)
    result_gt_172882 = python_operator(stypy.reporting.localization.Localization(__file__, 1496, 7), '>', ndim_172880, int_172881)
    
    
    # Getting the type of 'deg' (line 1496)
    deg_172883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 23), 'deg')
    # Obtaining the member 'dtype' of a type (line 1496)
    dtype_172884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1496, 23), deg_172883, 'dtype')
    # Obtaining the member 'kind' of a type (line 1496)
    kind_172885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1496, 23), dtype_172884, 'kind')
    str_172886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, 45), 'str', 'iu')
    # Applying the binary operator 'notin' (line 1496)
    result_contains_172887 = python_operator(stypy.reporting.localization.Localization(__file__, 1496, 23), 'notin', kind_172885, str_172886)
    
    # Applying the binary operator 'or' (line 1496)
    result_or_keyword_172888 = python_operator(stypy.reporting.localization.Localization(__file__, 1496, 7), 'or', result_gt_172882, result_contains_172887)
    
    # Getting the type of 'deg' (line 1496)
    deg_172889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 53), 'deg')
    # Obtaining the member 'size' of a type (line 1496)
    size_172890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1496, 53), deg_172889, 'size')
    int_172891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, 65), 'int')
    # Applying the binary operator '==' (line 1496)
    result_eq_172892 = python_operator(stypy.reporting.localization.Localization(__file__, 1496, 53), '==', size_172890, int_172891)
    
    # Applying the binary operator 'or' (line 1496)
    result_or_keyword_172893 = python_operator(stypy.reporting.localization.Localization(__file__, 1496, 7), 'or', result_or_keyword_172888, result_eq_172892)
    
    # Testing the type of an if condition (line 1496)
    if_condition_172894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1496, 4), result_or_keyword_172893)
    # Assigning a type to the variable 'if_condition_172894' (line 1496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1496, 4), 'if_condition_172894', if_condition_172894)
    # SSA begins for if statement (line 1496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1497)
    # Processing the call arguments (line 1497)
    str_172896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1497, 24), 'str', 'deg must be an int or non-empty 1-D array of int')
    # Processing the call keyword arguments (line 1497)
    kwargs_172897 = {}
    # Getting the type of 'TypeError' (line 1497)
    TypeError_172895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1497)
    TypeError_call_result_172898 = invoke(stypy.reporting.localization.Localization(__file__, 1497, 14), TypeError_172895, *[str_172896], **kwargs_172897)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1497, 8), TypeError_call_result_172898, 'raise parameter', BaseException)
    # SSA join for if statement (line 1496)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to min(...): (line 1498)
    # Processing the call keyword arguments (line 1498)
    kwargs_172901 = {}
    # Getting the type of 'deg' (line 1498)
    deg_172899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1498, 7), 'deg', False)
    # Obtaining the member 'min' of a type (line 1498)
    min_172900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1498, 7), deg_172899, 'min')
    # Calling min(args, kwargs) (line 1498)
    min_call_result_172902 = invoke(stypy.reporting.localization.Localization(__file__, 1498, 7), min_172900, *[], **kwargs_172901)
    
    int_172903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1498, 19), 'int')
    # Applying the binary operator '<' (line 1498)
    result_lt_172904 = python_operator(stypy.reporting.localization.Localization(__file__, 1498, 7), '<', min_call_result_172902, int_172903)
    
    # Testing the type of an if condition (line 1498)
    if_condition_172905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1498, 4), result_lt_172904)
    # Assigning a type to the variable 'if_condition_172905' (line 1498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1498, 4), 'if_condition_172905', if_condition_172905)
    # SSA begins for if statement (line 1498)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1499)
    # Processing the call arguments (line 1499)
    str_172907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1499, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 1499)
    kwargs_172908 = {}
    # Getting the type of 'ValueError' (line 1499)
    ValueError_172906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1499)
    ValueError_call_result_172909 = invoke(stypy.reporting.localization.Localization(__file__, 1499, 14), ValueError_172906, *[str_172907], **kwargs_172908)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1499, 8), ValueError_call_result_172909, 'raise parameter', BaseException)
    # SSA join for if statement (line 1498)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1500)
    x_172910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1500, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1500)
    ndim_172911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1500, 7), x_172910, 'ndim')
    int_172912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1500, 17), 'int')
    # Applying the binary operator '!=' (line 1500)
    result_ne_172913 = python_operator(stypy.reporting.localization.Localization(__file__, 1500, 7), '!=', ndim_172911, int_172912)
    
    # Testing the type of an if condition (line 1500)
    if_condition_172914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1500, 4), result_ne_172913)
    # Assigning a type to the variable 'if_condition_172914' (line 1500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1500, 4), 'if_condition_172914', if_condition_172914)
    # SSA begins for if statement (line 1500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1501)
    # Processing the call arguments (line 1501)
    str_172916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1501, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 1501)
    kwargs_172917 = {}
    # Getting the type of 'TypeError' (line 1501)
    TypeError_172915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1501)
    TypeError_call_result_172918 = invoke(stypy.reporting.localization.Localization(__file__, 1501, 14), TypeError_172915, *[str_172916], **kwargs_172917)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1501, 8), TypeError_call_result_172918, 'raise parameter', BaseException)
    # SSA join for if statement (line 1500)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1502)
    x_172919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 7), 'x')
    # Obtaining the member 'size' of a type (line 1502)
    size_172920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1502, 7), x_172919, 'size')
    int_172921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 17), 'int')
    # Applying the binary operator '==' (line 1502)
    result_eq_172922 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 7), '==', size_172920, int_172921)
    
    # Testing the type of an if condition (line 1502)
    if_condition_172923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1502, 4), result_eq_172922)
    # Assigning a type to the variable 'if_condition_172923' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 4), 'if_condition_172923', if_condition_172923)
    # SSA begins for if statement (line 1502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1503)
    # Processing the call arguments (line 1503)
    str_172925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1503, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 1503)
    kwargs_172926 = {}
    # Getting the type of 'TypeError' (line 1503)
    TypeError_172924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1503)
    TypeError_call_result_172927 = invoke(stypy.reporting.localization.Localization(__file__, 1503, 14), TypeError_172924, *[str_172925], **kwargs_172926)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1503, 8), TypeError_call_result_172927, 'raise parameter', BaseException)
    # SSA join for if statement (line 1502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 1504)
    y_172928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1504)
    ndim_172929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1504, 7), y_172928, 'ndim')
    int_172930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1504, 16), 'int')
    # Applying the binary operator '<' (line 1504)
    result_lt_172931 = python_operator(stypy.reporting.localization.Localization(__file__, 1504, 7), '<', ndim_172929, int_172930)
    
    
    # Getting the type of 'y' (line 1504)
    y_172932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 1504)
    ndim_172933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1504, 21), y_172932, 'ndim')
    int_172934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1504, 30), 'int')
    # Applying the binary operator '>' (line 1504)
    result_gt_172935 = python_operator(stypy.reporting.localization.Localization(__file__, 1504, 21), '>', ndim_172933, int_172934)
    
    # Applying the binary operator 'or' (line 1504)
    result_or_keyword_172936 = python_operator(stypy.reporting.localization.Localization(__file__, 1504, 7), 'or', result_lt_172931, result_gt_172935)
    
    # Testing the type of an if condition (line 1504)
    if_condition_172937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1504, 4), result_or_keyword_172936)
    # Assigning a type to the variable 'if_condition_172937' (line 1504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1504, 4), 'if_condition_172937', if_condition_172937)
    # SSA begins for if statement (line 1504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1505)
    # Processing the call arguments (line 1505)
    str_172939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 1505)
    kwargs_172940 = {}
    # Getting the type of 'TypeError' (line 1505)
    TypeError_172938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1505)
    TypeError_call_result_172941 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 14), TypeError_172938, *[str_172939], **kwargs_172940)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1505, 8), TypeError_call_result_172941, 'raise parameter', BaseException)
    # SSA join for if statement (line 1504)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1506)
    # Processing the call arguments (line 1506)
    # Getting the type of 'x' (line 1506)
    x_172943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 11), 'x', False)
    # Processing the call keyword arguments (line 1506)
    kwargs_172944 = {}
    # Getting the type of 'len' (line 1506)
    len_172942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 7), 'len', False)
    # Calling len(args, kwargs) (line 1506)
    len_call_result_172945 = invoke(stypy.reporting.localization.Localization(__file__, 1506, 7), len_172942, *[x_172943], **kwargs_172944)
    
    
    # Call to len(...): (line 1506)
    # Processing the call arguments (line 1506)
    # Getting the type of 'y' (line 1506)
    y_172947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 21), 'y', False)
    # Processing the call keyword arguments (line 1506)
    kwargs_172948 = {}
    # Getting the type of 'len' (line 1506)
    len_172946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 17), 'len', False)
    # Calling len(args, kwargs) (line 1506)
    len_call_result_172949 = invoke(stypy.reporting.localization.Localization(__file__, 1506, 17), len_172946, *[y_172947], **kwargs_172948)
    
    # Applying the binary operator '!=' (line 1506)
    result_ne_172950 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 7), '!=', len_call_result_172945, len_call_result_172949)
    
    # Testing the type of an if condition (line 1506)
    if_condition_172951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1506, 4), result_ne_172950)
    # Assigning a type to the variable 'if_condition_172951' (line 1506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1506, 4), 'if_condition_172951', if_condition_172951)
    # SSA begins for if statement (line 1506)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1507)
    # Processing the call arguments (line 1507)
    str_172953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1507, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 1507)
    kwargs_172954 = {}
    # Getting the type of 'TypeError' (line 1507)
    TypeError_172952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1507)
    TypeError_call_result_172955 = invoke(stypy.reporting.localization.Localization(__file__, 1507, 14), TypeError_172952, *[str_172953], **kwargs_172954)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1507, 8), TypeError_call_result_172955, 'raise parameter', BaseException)
    # SSA join for if statement (line 1506)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'deg' (line 1509)
    deg_172956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1509, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1509)
    ndim_172957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1509, 7), deg_172956, 'ndim')
    int_172958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1509, 19), 'int')
    # Applying the binary operator '==' (line 1509)
    result_eq_172959 = python_operator(stypy.reporting.localization.Localization(__file__, 1509, 7), '==', ndim_172957, int_172958)
    
    # Testing the type of an if condition (line 1509)
    if_condition_172960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1509, 4), result_eq_172959)
    # Assigning a type to the variable 'if_condition_172960' (line 1509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1509, 4), 'if_condition_172960', if_condition_172960)
    # SSA begins for if statement (line 1509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1510):
    
    # Assigning a Name to a Name (line 1510):
    # Getting the type of 'deg' (line 1510)
    deg_172961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 15), 'deg')
    # Assigning a type to the variable 'lmax' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 8), 'lmax', deg_172961)
    
    # Assigning a BinOp to a Name (line 1511):
    
    # Assigning a BinOp to a Name (line 1511):
    # Getting the type of 'lmax' (line 1511)
    lmax_172962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 16), 'lmax')
    int_172963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1511, 23), 'int')
    # Applying the binary operator '+' (line 1511)
    result_add_172964 = python_operator(stypy.reporting.localization.Localization(__file__, 1511, 16), '+', lmax_172962, int_172963)
    
    # Assigning a type to the variable 'order' (line 1511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1511, 8), 'order', result_add_172964)
    
    # Assigning a Call to a Name (line 1512):
    
    # Assigning a Call to a Name (line 1512):
    
    # Call to lagvander(...): (line 1512)
    # Processing the call arguments (line 1512)
    # Getting the type of 'x' (line 1512)
    x_172966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 24), 'x', False)
    # Getting the type of 'lmax' (line 1512)
    lmax_172967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 27), 'lmax', False)
    # Processing the call keyword arguments (line 1512)
    kwargs_172968 = {}
    # Getting the type of 'lagvander' (line 1512)
    lagvander_172965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 14), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1512)
    lagvander_call_result_172969 = invoke(stypy.reporting.localization.Localization(__file__, 1512, 14), lagvander_172965, *[x_172966, lmax_172967], **kwargs_172968)
    
    # Assigning a type to the variable 'van' (line 1512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1512, 8), 'van', lagvander_call_result_172969)
    # SSA branch for the else part of an if statement (line 1509)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1514):
    
    # Assigning a Call to a Name (line 1514):
    
    # Call to sort(...): (line 1514)
    # Processing the call arguments (line 1514)
    # Getting the type of 'deg' (line 1514)
    deg_172972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1514, 22), 'deg', False)
    # Processing the call keyword arguments (line 1514)
    kwargs_172973 = {}
    # Getting the type of 'np' (line 1514)
    np_172970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1514, 14), 'np', False)
    # Obtaining the member 'sort' of a type (line 1514)
    sort_172971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1514, 14), np_172970, 'sort')
    # Calling sort(args, kwargs) (line 1514)
    sort_call_result_172974 = invoke(stypy.reporting.localization.Localization(__file__, 1514, 14), sort_172971, *[deg_172972], **kwargs_172973)
    
    # Assigning a type to the variable 'deg' (line 1514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1514, 8), 'deg', sort_call_result_172974)
    
    # Assigning a Subscript to a Name (line 1515):
    
    # Assigning a Subscript to a Name (line 1515):
    
    # Obtaining the type of the subscript
    int_172975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1515, 19), 'int')
    # Getting the type of 'deg' (line 1515)
    deg_172976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 15), 'deg')
    # Obtaining the member '__getitem__' of a type (line 1515)
    getitem___172977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1515, 15), deg_172976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1515)
    subscript_call_result_172978 = invoke(stypy.reporting.localization.Localization(__file__, 1515, 15), getitem___172977, int_172975)
    
    # Assigning a type to the variable 'lmax' (line 1515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1515, 8), 'lmax', subscript_call_result_172978)
    
    # Assigning a Call to a Name (line 1516):
    
    # Assigning a Call to a Name (line 1516):
    
    # Call to len(...): (line 1516)
    # Processing the call arguments (line 1516)
    # Getting the type of 'deg' (line 1516)
    deg_172980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1516, 20), 'deg', False)
    # Processing the call keyword arguments (line 1516)
    kwargs_172981 = {}
    # Getting the type of 'len' (line 1516)
    len_172979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1516, 16), 'len', False)
    # Calling len(args, kwargs) (line 1516)
    len_call_result_172982 = invoke(stypy.reporting.localization.Localization(__file__, 1516, 16), len_172979, *[deg_172980], **kwargs_172981)
    
    # Assigning a type to the variable 'order' (line 1516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1516, 8), 'order', len_call_result_172982)
    
    # Assigning a Subscript to a Name (line 1517):
    
    # Assigning a Subscript to a Name (line 1517):
    
    # Obtaining the type of the subscript
    slice_172983 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1517, 14), None, None, None)
    # Getting the type of 'deg' (line 1517)
    deg_172984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 36), 'deg')
    
    # Call to lagvander(...): (line 1517)
    # Processing the call arguments (line 1517)
    # Getting the type of 'x' (line 1517)
    x_172986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 24), 'x', False)
    # Getting the type of 'lmax' (line 1517)
    lmax_172987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 27), 'lmax', False)
    # Processing the call keyword arguments (line 1517)
    kwargs_172988 = {}
    # Getting the type of 'lagvander' (line 1517)
    lagvander_172985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 14), 'lagvander', False)
    # Calling lagvander(args, kwargs) (line 1517)
    lagvander_call_result_172989 = invoke(stypy.reporting.localization.Localization(__file__, 1517, 14), lagvander_172985, *[x_172986, lmax_172987], **kwargs_172988)
    
    # Obtaining the member '__getitem__' of a type (line 1517)
    getitem___172990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1517, 14), lagvander_call_result_172989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1517)
    subscript_call_result_172991 = invoke(stypy.reporting.localization.Localization(__file__, 1517, 14), getitem___172990, (slice_172983, deg_172984))
    
    # Assigning a type to the variable 'van' (line 1517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1517, 8), 'van', subscript_call_result_172991)
    # SSA join for if statement (line 1509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1520):
    
    # Assigning a Attribute to a Name (line 1520):
    # Getting the type of 'van' (line 1520)
    van_172992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 10), 'van')
    # Obtaining the member 'T' of a type (line 1520)
    T_172993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1520, 10), van_172992, 'T')
    # Assigning a type to the variable 'lhs' (line 1520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1520, 4), 'lhs', T_172993)
    
    # Assigning a Attribute to a Name (line 1521):
    
    # Assigning a Attribute to a Name (line 1521):
    # Getting the type of 'y' (line 1521)
    y_172994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 10), 'y')
    # Obtaining the member 'T' of a type (line 1521)
    T_172995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1521, 10), y_172994, 'T')
    # Assigning a type to the variable 'rhs' (line 1521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1521, 4), 'rhs', T_172995)
    
    # Type idiom detected: calculating its left and rigth part (line 1522)
    # Getting the type of 'w' (line 1522)
    w_172996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'w')
    # Getting the type of 'None' (line 1522)
    None_172997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 16), 'None')
    
    (may_be_172998, more_types_in_union_172999) = may_not_be_none(w_172996, None_172997)

    if may_be_172998:

        if more_types_in_union_172999:
            # Runtime conditional SSA (line 1522)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1523):
        
        # Assigning a BinOp to a Name (line 1523):
        
        # Call to asarray(...): (line 1523)
        # Processing the call arguments (line 1523)
        # Getting the type of 'w' (line 1523)
        w_173002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 23), 'w', False)
        # Processing the call keyword arguments (line 1523)
        kwargs_173003 = {}
        # Getting the type of 'np' (line 1523)
        np_173000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1523)
        asarray_173001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1523, 12), np_173000, 'asarray')
        # Calling asarray(args, kwargs) (line 1523)
        asarray_call_result_173004 = invoke(stypy.reporting.localization.Localization(__file__, 1523, 12), asarray_173001, *[w_173002], **kwargs_173003)
        
        float_173005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1523, 28), 'float')
        # Applying the binary operator '+' (line 1523)
        result_add_173006 = python_operator(stypy.reporting.localization.Localization(__file__, 1523, 12), '+', asarray_call_result_173004, float_173005)
        
        # Assigning a type to the variable 'w' (line 1523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1523, 8), 'w', result_add_173006)
        
        
        # Getting the type of 'w' (line 1524)
        w_173007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 1524)
        ndim_173008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 11), w_173007, 'ndim')
        int_173009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 21), 'int')
        # Applying the binary operator '!=' (line 1524)
        result_ne_173010 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 11), '!=', ndim_173008, int_173009)
        
        # Testing the type of an if condition (line 1524)
        if_condition_173011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1524, 8), result_ne_173010)
        # Assigning a type to the variable 'if_condition_173011' (line 1524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 8), 'if_condition_173011', if_condition_173011)
        # SSA begins for if statement (line 1524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1525)
        # Processing the call arguments (line 1525)
        str_173013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1525, 28), 'str', 'expected 1D vector for w')
        # Processing the call keyword arguments (line 1525)
        kwargs_173014 = {}
        # Getting the type of 'TypeError' (line 1525)
        TypeError_173012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1525, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1525)
        TypeError_call_result_173015 = invoke(stypy.reporting.localization.Localization(__file__, 1525, 18), TypeError_173012, *[str_173013], **kwargs_173014)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1525, 12), TypeError_call_result_173015, 'raise parameter', BaseException)
        # SSA join for if statement (line 1524)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 1526)
        # Processing the call arguments (line 1526)
        # Getting the type of 'x' (line 1526)
        x_173017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 15), 'x', False)
        # Processing the call keyword arguments (line 1526)
        kwargs_173018 = {}
        # Getting the type of 'len' (line 1526)
        len_173016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 11), 'len', False)
        # Calling len(args, kwargs) (line 1526)
        len_call_result_173019 = invoke(stypy.reporting.localization.Localization(__file__, 1526, 11), len_173016, *[x_173017], **kwargs_173018)
        
        
        # Call to len(...): (line 1526)
        # Processing the call arguments (line 1526)
        # Getting the type of 'w' (line 1526)
        w_173021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 25), 'w', False)
        # Processing the call keyword arguments (line 1526)
        kwargs_173022 = {}
        # Getting the type of 'len' (line 1526)
        len_173020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 21), 'len', False)
        # Calling len(args, kwargs) (line 1526)
        len_call_result_173023 = invoke(stypy.reporting.localization.Localization(__file__, 1526, 21), len_173020, *[w_173021], **kwargs_173022)
        
        # Applying the binary operator '!=' (line 1526)
        result_ne_173024 = python_operator(stypy.reporting.localization.Localization(__file__, 1526, 11), '!=', len_call_result_173019, len_call_result_173023)
        
        # Testing the type of an if condition (line 1526)
        if_condition_173025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1526, 8), result_ne_173024)
        # Assigning a type to the variable 'if_condition_173025' (line 1526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1526, 8), 'if_condition_173025', if_condition_173025)
        # SSA begins for if statement (line 1526)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1527)
        # Processing the call arguments (line 1527)
        str_173027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1527, 28), 'str', 'expected x and w to have same length')
        # Processing the call keyword arguments (line 1527)
        kwargs_173028 = {}
        # Getting the type of 'TypeError' (line 1527)
        TypeError_173026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1527)
        TypeError_call_result_173029 = invoke(stypy.reporting.localization.Localization(__file__, 1527, 18), TypeError_173026, *[str_173027], **kwargs_173028)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1527, 12), TypeError_call_result_173029, 'raise parameter', BaseException)
        # SSA join for if statement (line 1526)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1530):
        
        # Assigning a BinOp to a Name (line 1530):
        # Getting the type of 'lhs' (line 1530)
        lhs_173030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1530, 14), 'lhs')
        # Getting the type of 'w' (line 1530)
        w_173031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1530, 20), 'w')
        # Applying the binary operator '*' (line 1530)
        result_mul_173032 = python_operator(stypy.reporting.localization.Localization(__file__, 1530, 14), '*', lhs_173030, w_173031)
        
        # Assigning a type to the variable 'lhs' (line 1530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1530, 8), 'lhs', result_mul_173032)
        
        # Assigning a BinOp to a Name (line 1531):
        
        # Assigning a BinOp to a Name (line 1531):
        # Getting the type of 'rhs' (line 1531)
        rhs_173033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 14), 'rhs')
        # Getting the type of 'w' (line 1531)
        w_173034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 20), 'w')
        # Applying the binary operator '*' (line 1531)
        result_mul_173035 = python_operator(stypy.reporting.localization.Localization(__file__, 1531, 14), '*', rhs_173033, w_173034)
        
        # Assigning a type to the variable 'rhs' (line 1531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1531, 8), 'rhs', result_mul_173035)

        if more_types_in_union_172999:
            # SSA join for if statement (line 1522)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1534)
    # Getting the type of 'rcond' (line 1534)
    rcond_173036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 7), 'rcond')
    # Getting the type of 'None' (line 1534)
    None_173037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 16), 'None')
    
    (may_be_173038, more_types_in_union_173039) = may_be_none(rcond_173036, None_173037)

    if may_be_173038:

        if more_types_in_union_173039:
            # Runtime conditional SSA (line 1534)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1535):
        
        # Assigning a BinOp to a Name (line 1535):
        
        # Call to len(...): (line 1535)
        # Processing the call arguments (line 1535)
        # Getting the type of 'x' (line 1535)
        x_173041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 20), 'x', False)
        # Processing the call keyword arguments (line 1535)
        kwargs_173042 = {}
        # Getting the type of 'len' (line 1535)
        len_173040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 16), 'len', False)
        # Calling len(args, kwargs) (line 1535)
        len_call_result_173043 = invoke(stypy.reporting.localization.Localization(__file__, 1535, 16), len_173040, *[x_173041], **kwargs_173042)
        
        
        # Call to finfo(...): (line 1535)
        # Processing the call arguments (line 1535)
        # Getting the type of 'x' (line 1535)
        x_173046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 32), 'x', False)
        # Obtaining the member 'dtype' of a type (line 1535)
        dtype_173047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1535, 32), x_173046, 'dtype')
        # Processing the call keyword arguments (line 1535)
        kwargs_173048 = {}
        # Getting the type of 'np' (line 1535)
        np_173044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 23), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1535)
        finfo_173045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1535, 23), np_173044, 'finfo')
        # Calling finfo(args, kwargs) (line 1535)
        finfo_call_result_173049 = invoke(stypy.reporting.localization.Localization(__file__, 1535, 23), finfo_173045, *[dtype_173047], **kwargs_173048)
        
        # Obtaining the member 'eps' of a type (line 1535)
        eps_173050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1535, 23), finfo_call_result_173049, 'eps')
        # Applying the binary operator '*' (line 1535)
        result_mul_173051 = python_operator(stypy.reporting.localization.Localization(__file__, 1535, 16), '*', len_call_result_173043, eps_173050)
        
        # Assigning a type to the variable 'rcond' (line 1535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1535, 8), 'rcond', result_mul_173051)

        if more_types_in_union_173039:
            # SSA join for if statement (line 1534)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issubclass(...): (line 1538)
    # Processing the call arguments (line 1538)
    # Getting the type of 'lhs' (line 1538)
    lhs_173053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 18), 'lhs', False)
    # Obtaining the member 'dtype' of a type (line 1538)
    dtype_173054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1538, 18), lhs_173053, 'dtype')
    # Obtaining the member 'type' of a type (line 1538)
    type_173055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1538, 18), dtype_173054, 'type')
    # Getting the type of 'np' (line 1538)
    np_173056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 34), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1538)
    complexfloating_173057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1538, 34), np_173056, 'complexfloating')
    # Processing the call keyword arguments (line 1538)
    kwargs_173058 = {}
    # Getting the type of 'issubclass' (line 1538)
    issubclass_173052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1538)
    issubclass_call_result_173059 = invoke(stypy.reporting.localization.Localization(__file__, 1538, 7), issubclass_173052, *[type_173055, complexfloating_173057], **kwargs_173058)
    
    # Testing the type of an if condition (line 1538)
    if_condition_173060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1538, 4), issubclass_call_result_173059)
    # Assigning a type to the variable 'if_condition_173060' (line 1538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1538, 4), 'if_condition_173060', if_condition_173060)
    # SSA begins for if statement (line 1538)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1539):
    
    # Assigning a Call to a Name (line 1539):
    
    # Call to sqrt(...): (line 1539)
    # Processing the call arguments (line 1539)
    
    # Call to sum(...): (line 1539)
    # Processing the call arguments (line 1539)
    int_173077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1539, 70), 'int')
    # Processing the call keyword arguments (line 1539)
    kwargs_173078 = {}
    
    # Call to square(...): (line 1539)
    # Processing the call arguments (line 1539)
    # Getting the type of 'lhs' (line 1539)
    lhs_173065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 33), 'lhs', False)
    # Obtaining the member 'real' of a type (line 1539)
    real_173066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 33), lhs_173065, 'real')
    # Processing the call keyword arguments (line 1539)
    kwargs_173067 = {}
    # Getting the type of 'np' (line 1539)
    np_173063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 1539)
    square_173064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 23), np_173063, 'square')
    # Calling square(args, kwargs) (line 1539)
    square_call_result_173068 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 23), square_173064, *[real_173066], **kwargs_173067)
    
    
    # Call to square(...): (line 1539)
    # Processing the call arguments (line 1539)
    # Getting the type of 'lhs' (line 1539)
    lhs_173071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 55), 'lhs', False)
    # Obtaining the member 'imag' of a type (line 1539)
    imag_173072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 55), lhs_173071, 'imag')
    # Processing the call keyword arguments (line 1539)
    kwargs_173073 = {}
    # Getting the type of 'np' (line 1539)
    np_173069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 45), 'np', False)
    # Obtaining the member 'square' of a type (line 1539)
    square_173070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 45), np_173069, 'square')
    # Calling square(args, kwargs) (line 1539)
    square_call_result_173074 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 45), square_173070, *[imag_173072], **kwargs_173073)
    
    # Applying the binary operator '+' (line 1539)
    result_add_173075 = python_operator(stypy.reporting.localization.Localization(__file__, 1539, 23), '+', square_call_result_173068, square_call_result_173074)
    
    # Obtaining the member 'sum' of a type (line 1539)
    sum_173076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 23), result_add_173075, 'sum')
    # Calling sum(args, kwargs) (line 1539)
    sum_call_result_173079 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 23), sum_173076, *[int_173077], **kwargs_173078)
    
    # Processing the call keyword arguments (line 1539)
    kwargs_173080 = {}
    # Getting the type of 'np' (line 1539)
    np_173061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1539)
    sqrt_173062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 14), np_173061, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1539)
    sqrt_call_result_173081 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 14), sqrt_173062, *[sum_call_result_173079], **kwargs_173080)
    
    # Assigning a type to the variable 'scl' (line 1539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1539, 8), 'scl', sqrt_call_result_173081)
    # SSA branch for the else part of an if statement (line 1538)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1541):
    
    # Assigning a Call to a Name (line 1541):
    
    # Call to sqrt(...): (line 1541)
    # Processing the call arguments (line 1541)
    
    # Call to sum(...): (line 1541)
    # Processing the call arguments (line 1541)
    int_173090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1541, 41), 'int')
    # Processing the call keyword arguments (line 1541)
    kwargs_173091 = {}
    
    # Call to square(...): (line 1541)
    # Processing the call arguments (line 1541)
    # Getting the type of 'lhs' (line 1541)
    lhs_173086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 32), 'lhs', False)
    # Processing the call keyword arguments (line 1541)
    kwargs_173087 = {}
    # Getting the type of 'np' (line 1541)
    np_173084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 22), 'np', False)
    # Obtaining the member 'square' of a type (line 1541)
    square_173085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1541, 22), np_173084, 'square')
    # Calling square(args, kwargs) (line 1541)
    square_call_result_173088 = invoke(stypy.reporting.localization.Localization(__file__, 1541, 22), square_173085, *[lhs_173086], **kwargs_173087)
    
    # Obtaining the member 'sum' of a type (line 1541)
    sum_173089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1541, 22), square_call_result_173088, 'sum')
    # Calling sum(args, kwargs) (line 1541)
    sum_call_result_173092 = invoke(stypy.reporting.localization.Localization(__file__, 1541, 22), sum_173089, *[int_173090], **kwargs_173091)
    
    # Processing the call keyword arguments (line 1541)
    kwargs_173093 = {}
    # Getting the type of 'np' (line 1541)
    np_173082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1541)
    sqrt_173083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1541, 14), np_173082, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1541)
    sqrt_call_result_173094 = invoke(stypy.reporting.localization.Localization(__file__, 1541, 14), sqrt_173083, *[sum_call_result_173092], **kwargs_173093)
    
    # Assigning a type to the variable 'scl' (line 1541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1541, 8), 'scl', sqrt_call_result_173094)
    # SSA join for if statement (line 1538)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 1542):
    
    # Assigning a Num to a Subscript (line 1542):
    int_173095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1542, 20), 'int')
    # Getting the type of 'scl' (line 1542)
    scl_173096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 4), 'scl')
    
    # Getting the type of 'scl' (line 1542)
    scl_173097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 8), 'scl')
    int_173098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1542, 15), 'int')
    # Applying the binary operator '==' (line 1542)
    result_eq_173099 = python_operator(stypy.reporting.localization.Localization(__file__, 1542, 8), '==', scl_173097, int_173098)
    
    # Storing an element on a container (line 1542)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1542, 4), scl_173096, (result_eq_173099, int_173095))
    
    # Assigning a Call to a Tuple (line 1545):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 1545)
    # Processing the call arguments (line 1545)
    # Getting the type of 'lhs' (line 1545)
    lhs_173102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 34), 'lhs', False)
    # Obtaining the member 'T' of a type (line 1545)
    T_173103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 34), lhs_173102, 'T')
    # Getting the type of 'scl' (line 1545)
    scl_173104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 40), 'scl', False)
    # Applying the binary operator 'div' (line 1545)
    result_div_173105 = python_operator(stypy.reporting.localization.Localization(__file__, 1545, 34), 'div', T_173103, scl_173104)
    
    # Getting the type of 'rhs' (line 1545)
    rhs_173106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 45), 'rhs', False)
    # Obtaining the member 'T' of a type (line 1545)
    T_173107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 45), rhs_173106, 'T')
    # Getting the type of 'rcond' (line 1545)
    rcond_173108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 52), 'rcond', False)
    # Processing the call keyword arguments (line 1545)
    kwargs_173109 = {}
    # Getting the type of 'la' (line 1545)
    la_173100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 25), 'la', False)
    # Obtaining the member 'lstsq' of a type (line 1545)
    lstsq_173101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 25), la_173100, 'lstsq')
    # Calling lstsq(args, kwargs) (line 1545)
    lstsq_call_result_173110 = invoke(stypy.reporting.localization.Localization(__file__, 1545, 25), lstsq_173101, *[result_div_173105, T_173107, rcond_173108], **kwargs_173109)
    
    # Assigning a type to the variable 'call_assignment_170808' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170808', lstsq_call_result_173110)
    
    # Assigning a Call to a Name (line 1545):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1545, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173114 = {}
    # Getting the type of 'call_assignment_170808' (line 1545)
    call_assignment_170808_173111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170808', False)
    # Obtaining the member '__getitem__' of a type (line 1545)
    getitem___173112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 4), call_assignment_170808_173111, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173115 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173112, *[int_173113], **kwargs_173114)
    
    # Assigning a type to the variable 'call_assignment_170809' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170809', getitem___call_result_173115)
    
    # Assigning a Name to a Name (line 1545):
    # Getting the type of 'call_assignment_170809' (line 1545)
    call_assignment_170809_173116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170809')
    # Assigning a type to the variable 'c' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'c', call_assignment_170809_173116)
    
    # Assigning a Call to a Name (line 1545):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1545, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173120 = {}
    # Getting the type of 'call_assignment_170808' (line 1545)
    call_assignment_170808_173117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170808', False)
    # Obtaining the member '__getitem__' of a type (line 1545)
    getitem___173118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 4), call_assignment_170808_173117, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173121 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173118, *[int_173119], **kwargs_173120)
    
    # Assigning a type to the variable 'call_assignment_170810' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170810', getitem___call_result_173121)
    
    # Assigning a Name to a Name (line 1545):
    # Getting the type of 'call_assignment_170810' (line 1545)
    call_assignment_170810_173122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170810')
    # Assigning a type to the variable 'resids' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 7), 'resids', call_assignment_170810_173122)
    
    # Assigning a Call to a Name (line 1545):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1545, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173126 = {}
    # Getting the type of 'call_assignment_170808' (line 1545)
    call_assignment_170808_173123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170808', False)
    # Obtaining the member '__getitem__' of a type (line 1545)
    getitem___173124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 4), call_assignment_170808_173123, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173127 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173124, *[int_173125], **kwargs_173126)
    
    # Assigning a type to the variable 'call_assignment_170811' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170811', getitem___call_result_173127)
    
    # Assigning a Name to a Name (line 1545):
    # Getting the type of 'call_assignment_170811' (line 1545)
    call_assignment_170811_173128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170811')
    # Assigning a type to the variable 'rank' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 15), 'rank', call_assignment_170811_173128)
    
    # Assigning a Call to a Name (line 1545):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1545, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173132 = {}
    # Getting the type of 'call_assignment_170808' (line 1545)
    call_assignment_170808_173129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170808', False)
    # Obtaining the member '__getitem__' of a type (line 1545)
    getitem___173130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 4), call_assignment_170808_173129, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173133 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173130, *[int_173131], **kwargs_173132)
    
    # Assigning a type to the variable 'call_assignment_170812' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170812', getitem___call_result_173133)
    
    # Assigning a Name to a Name (line 1545):
    # Getting the type of 'call_assignment_170812' (line 1545)
    call_assignment_170812_173134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'call_assignment_170812')
    # Assigning a type to the variable 's' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 21), 's', call_assignment_170812_173134)
    
    # Assigning a Attribute to a Name (line 1546):
    
    # Assigning a Attribute to a Name (line 1546):
    # Getting the type of 'c' (line 1546)
    c_173135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 9), 'c')
    # Obtaining the member 'T' of a type (line 1546)
    T_173136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 9), c_173135, 'T')
    # Getting the type of 'scl' (line 1546)
    scl_173137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 13), 'scl')
    # Applying the binary operator 'div' (line 1546)
    result_div_173138 = python_operator(stypy.reporting.localization.Localization(__file__, 1546, 9), 'div', T_173136, scl_173137)
    
    # Obtaining the member 'T' of a type (line 1546)
    T_173139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 9), result_div_173138, 'T')
    # Assigning a type to the variable 'c' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'c', T_173139)
    
    
    # Getting the type of 'deg' (line 1549)
    deg_173140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1549)
    ndim_173141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 7), deg_173140, 'ndim')
    int_173142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, 18), 'int')
    # Applying the binary operator '>' (line 1549)
    result_gt_173143 = python_operator(stypy.reporting.localization.Localization(__file__, 1549, 7), '>', ndim_173141, int_173142)
    
    # Testing the type of an if condition (line 1549)
    if_condition_173144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1549, 4), result_gt_173143)
    # Assigning a type to the variable 'if_condition_173144' (line 1549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1549, 4), 'if_condition_173144', if_condition_173144)
    # SSA begins for if statement (line 1549)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'c' (line 1550)
    c_173145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 11), 'c')
    # Obtaining the member 'ndim' of a type (line 1550)
    ndim_173146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 11), c_173145, 'ndim')
    int_173147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1550, 21), 'int')
    # Applying the binary operator '==' (line 1550)
    result_eq_173148 = python_operator(stypy.reporting.localization.Localization(__file__, 1550, 11), '==', ndim_173146, int_173147)
    
    # Testing the type of an if condition (line 1550)
    if_condition_173149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1550, 8), result_eq_173148)
    # Assigning a type to the variable 'if_condition_173149' (line 1550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1550, 8), 'if_condition_173149', if_condition_173149)
    # SSA begins for if statement (line 1550)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1551):
    
    # Assigning a Call to a Name (line 1551):
    
    # Call to zeros(...): (line 1551)
    # Processing the call arguments (line 1551)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1551)
    tuple_173152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1551)
    # Adding element type (line 1551)
    # Getting the type of 'lmax' (line 1551)
    lmax_173153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 27), 'lmax', False)
    int_173154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 32), 'int')
    # Applying the binary operator '+' (line 1551)
    result_add_173155 = python_operator(stypy.reporting.localization.Localization(__file__, 1551, 27), '+', lmax_173153, int_173154)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1551, 27), tuple_173152, result_add_173155)
    # Adding element type (line 1551)
    
    # Obtaining the type of the subscript
    int_173156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 43), 'int')
    # Getting the type of 'c' (line 1551)
    c_173157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 35), 'c', False)
    # Obtaining the member 'shape' of a type (line 1551)
    shape_173158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 35), c_173157, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1551)
    getitem___173159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 35), shape_173158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1551)
    subscript_call_result_173160 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 35), getitem___173159, int_173156)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1551, 27), tuple_173152, subscript_call_result_173160)
    
    # Processing the call keyword arguments (line 1551)
    # Getting the type of 'c' (line 1551)
    c_173161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 54), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1551)
    dtype_173162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 54), c_173161, 'dtype')
    keyword_173163 = dtype_173162
    kwargs_173164 = {'dtype': keyword_173163}
    # Getting the type of 'np' (line 1551)
    np_173150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1551)
    zeros_173151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 17), np_173150, 'zeros')
    # Calling zeros(args, kwargs) (line 1551)
    zeros_call_result_173165 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 17), zeros_173151, *[tuple_173152], **kwargs_173164)
    
    # Assigning a type to the variable 'cc' (line 1551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1551, 12), 'cc', zeros_call_result_173165)
    # SSA branch for the else part of an if statement (line 1550)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1553):
    
    # Assigning a Call to a Name (line 1553):
    
    # Call to zeros(...): (line 1553)
    # Processing the call arguments (line 1553)
    # Getting the type of 'lmax' (line 1553)
    lmax_173168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 26), 'lmax', False)
    int_173169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1553, 31), 'int')
    # Applying the binary operator '+' (line 1553)
    result_add_173170 = python_operator(stypy.reporting.localization.Localization(__file__, 1553, 26), '+', lmax_173168, int_173169)
    
    # Processing the call keyword arguments (line 1553)
    # Getting the type of 'c' (line 1553)
    c_173171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 40), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1553)
    dtype_173172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1553, 40), c_173171, 'dtype')
    keyword_173173 = dtype_173172
    kwargs_173174 = {'dtype': keyword_173173}
    # Getting the type of 'np' (line 1553)
    np_173166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1553)
    zeros_173167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1553, 17), np_173166, 'zeros')
    # Calling zeros(args, kwargs) (line 1553)
    zeros_call_result_173175 = invoke(stypy.reporting.localization.Localization(__file__, 1553, 17), zeros_173167, *[result_add_173170], **kwargs_173174)
    
    # Assigning a type to the variable 'cc' (line 1553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1553, 12), 'cc', zeros_call_result_173175)
    # SSA join for if statement (line 1550)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1554):
    
    # Assigning a Name to a Subscript (line 1554):
    # Getting the type of 'c' (line 1554)
    c_173176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 18), 'c')
    # Getting the type of 'cc' (line 1554)
    cc_173177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 8), 'cc')
    # Getting the type of 'deg' (line 1554)
    deg_173178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 11), 'deg')
    # Storing an element on a container (line 1554)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1554, 8), cc_173177, (deg_173178, c_173176))
    
    # Assigning a Name to a Name (line 1555):
    
    # Assigning a Name to a Name (line 1555):
    # Getting the type of 'cc' (line 1555)
    cc_173179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 12), 'cc')
    # Assigning a type to the variable 'c' (line 1555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1555, 8), 'c', cc_173179)
    # SSA join for if statement (line 1549)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1558)
    rank_173180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 7), 'rank')
    # Getting the type of 'order' (line 1558)
    order_173181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 15), 'order')
    # Applying the binary operator '!=' (line 1558)
    result_ne_173182 = python_operator(stypy.reporting.localization.Localization(__file__, 1558, 7), '!=', rank_173180, order_173181)
    
    
    # Getting the type of 'full' (line 1558)
    full_173183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 29), 'full')
    # Applying the 'not' unary operator (line 1558)
    result_not__173184 = python_operator(stypy.reporting.localization.Localization(__file__, 1558, 25), 'not', full_173183)
    
    # Applying the binary operator 'and' (line 1558)
    result_and_keyword_173185 = python_operator(stypy.reporting.localization.Localization(__file__, 1558, 7), 'and', result_ne_173182, result_not__173184)
    
    # Testing the type of an if condition (line 1558)
    if_condition_173186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1558, 4), result_and_keyword_173185)
    # Assigning a type to the variable 'if_condition_173186' (line 1558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1558, 4), 'if_condition_173186', if_condition_173186)
    # SSA begins for if statement (line 1558)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1559):
    
    # Assigning a Str to a Name (line 1559):
    str_173187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1559, 14), 'str', 'The fit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 1559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 8), 'msg', str_173187)
    
    # Call to warn(...): (line 1560)
    # Processing the call arguments (line 1560)
    # Getting the type of 'msg' (line 1560)
    msg_173190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1560, 22), 'msg', False)
    # Getting the type of 'pu' (line 1560)
    pu_173191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1560, 27), 'pu', False)
    # Obtaining the member 'RankWarning' of a type (line 1560)
    RankWarning_173192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1560, 27), pu_173191, 'RankWarning')
    # Processing the call keyword arguments (line 1560)
    kwargs_173193 = {}
    # Getting the type of 'warnings' (line 1560)
    warnings_173188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1560, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1560)
    warn_173189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1560, 8), warnings_173188, 'warn')
    # Calling warn(args, kwargs) (line 1560)
    warn_call_result_173194 = invoke(stypy.reporting.localization.Localization(__file__, 1560, 8), warn_173189, *[msg_173190, RankWarning_173192], **kwargs_173193)
    
    # SSA join for if statement (line 1558)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 1562)
    full_173195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1562, 7), 'full')
    # Testing the type of an if condition (line 1562)
    if_condition_173196 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1562, 4), full_173195)
    # Assigning a type to the variable 'if_condition_173196' (line 1562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1562, 4), 'if_condition_173196', if_condition_173196)
    # SSA begins for if statement (line 1562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1563)
    tuple_173197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1563, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1563)
    # Adding element type (line 1563)
    # Getting the type of 'c' (line 1563)
    c_173198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1563, 15), tuple_173197, c_173198)
    # Adding element type (line 1563)
    
    # Obtaining an instance of the builtin type 'list' (line 1563)
    list_173199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1563, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1563)
    # Adding element type (line 1563)
    # Getting the type of 'resids' (line 1563)
    resids_173200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 19), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1563, 18), list_173199, resids_173200)
    # Adding element type (line 1563)
    # Getting the type of 'rank' (line 1563)
    rank_173201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 27), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1563, 18), list_173199, rank_173201)
    # Adding element type (line 1563)
    # Getting the type of 's' (line 1563)
    s_173202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 33), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1563, 18), list_173199, s_173202)
    # Adding element type (line 1563)
    # Getting the type of 'rcond' (line 1563)
    rcond_173203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 36), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1563, 18), list_173199, rcond_173203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1563, 15), tuple_173197, list_173199)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1563, 8), 'stypy_return_type', tuple_173197)
    # SSA branch for the else part of an if statement (line 1562)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 1565)
    c_173204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1565, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1565, 8), 'stypy_return_type', c_173204)
    # SSA join for if statement (line 1562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lagfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagfit' in the type store
    # Getting the type of 'stypy_return_type' (line 1367)
    stypy_return_type_173205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173205)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagfit'
    return stypy_return_type_173205

# Assigning a type to the variable 'lagfit' (line 1367)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1367, 0), 'lagfit', lagfit)

@norecursion
def lagcompanion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagcompanion'
    module_type_store = module_type_store.open_function_context('lagcompanion', 1568, 0, False)
    
    # Passed parameters checking function
    lagcompanion.stypy_localization = localization
    lagcompanion.stypy_type_of_self = None
    lagcompanion.stypy_type_store = module_type_store
    lagcompanion.stypy_function_name = 'lagcompanion'
    lagcompanion.stypy_param_names_list = ['c']
    lagcompanion.stypy_varargs_param_name = None
    lagcompanion.stypy_kwargs_param_name = None
    lagcompanion.stypy_call_defaults = defaults
    lagcompanion.stypy_call_varargs = varargs
    lagcompanion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagcompanion', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagcompanion', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagcompanion(...)' code ##################

    str_173206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, (-1)), 'str', '\n    Return the companion matrix of c.\n\n    The usual companion matrix of the Laguerre polynomials is already\n    symmetric when `c` is a basis Laguerre polynomial, so no scaling is\n    applied.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Laguerre series coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a List (line 1594):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1594)
    # Processing the call arguments (line 1594)
    
    # Obtaining an instance of the builtin type 'list' (line 1594)
    list_173209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1594)
    # Adding element type (line 1594)
    # Getting the type of 'c' (line 1594)
    c_173210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1594, 23), list_173209, c_173210)
    
    # Processing the call keyword arguments (line 1594)
    kwargs_173211 = {}
    # Getting the type of 'pu' (line 1594)
    pu_173207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1594)
    as_series_173208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 10), pu_173207, 'as_series')
    # Calling as_series(args, kwargs) (line 1594)
    as_series_call_result_173212 = invoke(stypy.reporting.localization.Localization(__file__, 1594, 10), as_series_173208, *[list_173209], **kwargs_173211)
    
    # Assigning a type to the variable 'call_assignment_170813' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_170813', as_series_call_result_173212)
    
    # Assigning a Call to a Name (line 1594):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173216 = {}
    # Getting the type of 'call_assignment_170813' (line 1594)
    call_assignment_170813_173213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_170813', False)
    # Obtaining the member '__getitem__' of a type (line 1594)
    getitem___173214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 4), call_assignment_170813_173213, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173217 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173214, *[int_173215], **kwargs_173216)
    
    # Assigning a type to the variable 'call_assignment_170814' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_170814', getitem___call_result_173217)
    
    # Assigning a Name to a Name (line 1594):
    # Getting the type of 'call_assignment_170814' (line 1594)
    call_assignment_170814_173218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_170814')
    # Assigning a type to the variable 'c' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 5), 'c', call_assignment_170814_173218)
    
    
    
    # Call to len(...): (line 1595)
    # Processing the call arguments (line 1595)
    # Getting the type of 'c' (line 1595)
    c_173220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 11), 'c', False)
    # Processing the call keyword arguments (line 1595)
    kwargs_173221 = {}
    # Getting the type of 'len' (line 1595)
    len_173219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 7), 'len', False)
    # Calling len(args, kwargs) (line 1595)
    len_call_result_173222 = invoke(stypy.reporting.localization.Localization(__file__, 1595, 7), len_173219, *[c_173220], **kwargs_173221)
    
    int_173223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1595, 16), 'int')
    # Applying the binary operator '<' (line 1595)
    result_lt_173224 = python_operator(stypy.reporting.localization.Localization(__file__, 1595, 7), '<', len_call_result_173222, int_173223)
    
    # Testing the type of an if condition (line 1595)
    if_condition_173225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1595, 4), result_lt_173224)
    # Assigning a type to the variable 'if_condition_173225' (line 1595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1595, 4), 'if_condition_173225', if_condition_173225)
    # SSA begins for if statement (line 1595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1596)
    # Processing the call arguments (line 1596)
    str_173227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 25), 'str', 'Series must have maximum degree of at least 1.')
    # Processing the call keyword arguments (line 1596)
    kwargs_173228 = {}
    # Getting the type of 'ValueError' (line 1596)
    ValueError_173226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1596)
    ValueError_call_result_173229 = invoke(stypy.reporting.localization.Localization(__file__, 1596, 14), ValueError_173226, *[str_173227], **kwargs_173228)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1596, 8), ValueError_call_result_173229, 'raise parameter', BaseException)
    # SSA join for if statement (line 1595)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1597)
    # Processing the call arguments (line 1597)
    # Getting the type of 'c' (line 1597)
    c_173231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 11), 'c', False)
    # Processing the call keyword arguments (line 1597)
    kwargs_173232 = {}
    # Getting the type of 'len' (line 1597)
    len_173230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 7), 'len', False)
    # Calling len(args, kwargs) (line 1597)
    len_call_result_173233 = invoke(stypy.reporting.localization.Localization(__file__, 1597, 7), len_173230, *[c_173231], **kwargs_173232)
    
    int_173234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1597, 17), 'int')
    # Applying the binary operator '==' (line 1597)
    result_eq_173235 = python_operator(stypy.reporting.localization.Localization(__file__, 1597, 7), '==', len_call_result_173233, int_173234)
    
    # Testing the type of an if condition (line 1597)
    if_condition_173236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1597, 4), result_eq_173235)
    # Assigning a type to the variable 'if_condition_173236' (line 1597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1597, 4), 'if_condition_173236', if_condition_173236)
    # SSA begins for if statement (line 1597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1598)
    # Processing the call arguments (line 1598)
    
    # Obtaining an instance of the builtin type 'list' (line 1598)
    list_173239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1598)
    # Adding element type (line 1598)
    
    # Obtaining an instance of the builtin type 'list' (line 1598)
    list_173240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1598)
    # Adding element type (line 1598)
    int_173241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 26), 'int')
    
    # Obtaining the type of the subscript
    int_173242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 32), 'int')
    # Getting the type of 'c' (line 1598)
    c_173243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 30), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1598)
    getitem___173244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1598, 30), c_173243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1598)
    subscript_call_result_173245 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 30), getitem___173244, int_173242)
    
    
    # Obtaining the type of the subscript
    int_173246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 37), 'int')
    # Getting the type of 'c' (line 1598)
    c_173247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 35), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1598)
    getitem___173248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1598, 35), c_173247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1598)
    subscript_call_result_173249 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 35), getitem___173248, int_173246)
    
    # Applying the binary operator 'div' (line 1598)
    result_div_173250 = python_operator(stypy.reporting.localization.Localization(__file__, 1598, 30), 'div', subscript_call_result_173245, subscript_call_result_173249)
    
    # Applying the binary operator '+' (line 1598)
    result_add_173251 = python_operator(stypy.reporting.localization.Localization(__file__, 1598, 26), '+', int_173241, result_div_173250)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1598, 25), list_173240, result_add_173251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1598, 24), list_173239, list_173240)
    
    # Processing the call keyword arguments (line 1598)
    kwargs_173252 = {}
    # Getting the type of 'np' (line 1598)
    np_173237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1598)
    array_173238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1598, 15), np_173237, 'array')
    # Calling array(args, kwargs) (line 1598)
    array_call_result_173253 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 15), array_173238, *[list_173239], **kwargs_173252)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1598, 8), 'stypy_return_type', array_call_result_173253)
    # SSA join for if statement (line 1597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1600):
    
    # Assigning a BinOp to a Name (line 1600):
    
    # Call to len(...): (line 1600)
    # Processing the call arguments (line 1600)
    # Getting the type of 'c' (line 1600)
    c_173255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 12), 'c', False)
    # Processing the call keyword arguments (line 1600)
    kwargs_173256 = {}
    # Getting the type of 'len' (line 1600)
    len_173254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 8), 'len', False)
    # Calling len(args, kwargs) (line 1600)
    len_call_result_173257 = invoke(stypy.reporting.localization.Localization(__file__, 1600, 8), len_173254, *[c_173255], **kwargs_173256)
    
    int_173258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 17), 'int')
    # Applying the binary operator '-' (line 1600)
    result_sub_173259 = python_operator(stypy.reporting.localization.Localization(__file__, 1600, 8), '-', len_call_result_173257, int_173258)
    
    # Assigning a type to the variable 'n' (line 1600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1600, 4), 'n', result_sub_173259)
    
    # Assigning a Call to a Name (line 1601):
    
    # Assigning a Call to a Name (line 1601):
    
    # Call to zeros(...): (line 1601)
    # Processing the call arguments (line 1601)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1601)
    tuple_173262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1601, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1601)
    # Adding element type (line 1601)
    # Getting the type of 'n' (line 1601)
    n_173263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1601, 20), tuple_173262, n_173263)
    # Adding element type (line 1601)
    # Getting the type of 'n' (line 1601)
    n_173264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1601, 20), tuple_173262, n_173264)
    
    # Processing the call keyword arguments (line 1601)
    # Getting the type of 'c' (line 1601)
    c_173265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 33), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1601)
    dtype_173266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 33), c_173265, 'dtype')
    keyword_173267 = dtype_173266
    kwargs_173268 = {'dtype': keyword_173267}
    # Getting the type of 'np' (line 1601)
    np_173260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1601)
    zeros_173261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 10), np_173260, 'zeros')
    # Calling zeros(args, kwargs) (line 1601)
    zeros_call_result_173269 = invoke(stypy.reporting.localization.Localization(__file__, 1601, 10), zeros_173261, *[tuple_173262], **kwargs_173268)
    
    # Assigning a type to the variable 'mat' (line 1601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1601, 4), 'mat', zeros_call_result_173269)
    
    # Assigning a Subscript to a Name (line 1602):
    
    # Assigning a Subscript to a Name (line 1602):
    
    # Obtaining the type of the subscript
    int_173270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 26), 'int')
    # Getting the type of 'n' (line 1602)
    n_173271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 29), 'n')
    int_173272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 31), 'int')
    # Applying the binary operator '+' (line 1602)
    result_add_173273 = python_operator(stypy.reporting.localization.Localization(__file__, 1602, 29), '+', n_173271, int_173272)
    
    slice_173274 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1602, 10), int_173270, None, result_add_173273)
    
    # Call to reshape(...): (line 1602)
    # Processing the call arguments (line 1602)
    int_173277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 22), 'int')
    # Processing the call keyword arguments (line 1602)
    kwargs_173278 = {}
    # Getting the type of 'mat' (line 1602)
    mat_173275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1602)
    reshape_173276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 10), mat_173275, 'reshape')
    # Calling reshape(args, kwargs) (line 1602)
    reshape_call_result_173279 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 10), reshape_173276, *[int_173277], **kwargs_173278)
    
    # Obtaining the member '__getitem__' of a type (line 1602)
    getitem___173280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 10), reshape_call_result_173279, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1602)
    subscript_call_result_173281 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 10), getitem___173280, slice_173274)
    
    # Assigning a type to the variable 'top' (line 1602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 4), 'top', subscript_call_result_173281)
    
    # Assigning a Subscript to a Name (line 1603):
    
    # Assigning a Subscript to a Name (line 1603):
    
    # Obtaining the type of the subscript
    int_173282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1603, 26), 'int')
    # Getting the type of 'n' (line 1603)
    n_173283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 29), 'n')
    int_173284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1603, 31), 'int')
    # Applying the binary operator '+' (line 1603)
    result_add_173285 = python_operator(stypy.reporting.localization.Localization(__file__, 1603, 29), '+', n_173283, int_173284)
    
    slice_173286 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1603, 10), int_173282, None, result_add_173285)
    
    # Call to reshape(...): (line 1603)
    # Processing the call arguments (line 1603)
    int_173289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1603, 22), 'int')
    # Processing the call keyword arguments (line 1603)
    kwargs_173290 = {}
    # Getting the type of 'mat' (line 1603)
    mat_173287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1603)
    reshape_173288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 10), mat_173287, 'reshape')
    # Calling reshape(args, kwargs) (line 1603)
    reshape_call_result_173291 = invoke(stypy.reporting.localization.Localization(__file__, 1603, 10), reshape_173288, *[int_173289], **kwargs_173290)
    
    # Obtaining the member '__getitem__' of a type (line 1603)
    getitem___173292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 10), reshape_call_result_173291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1603)
    subscript_call_result_173293 = invoke(stypy.reporting.localization.Localization(__file__, 1603, 10), getitem___173292, slice_173286)
    
    # Assigning a type to the variable 'mid' (line 1603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1603, 4), 'mid', subscript_call_result_173293)
    
    # Assigning a Subscript to a Name (line 1604):
    
    # Assigning a Subscript to a Name (line 1604):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1604)
    n_173294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 26), 'n')
    # Getting the type of 'n' (line 1604)
    n_173295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 29), 'n')
    int_173296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 31), 'int')
    # Applying the binary operator '+' (line 1604)
    result_add_173297 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 29), '+', n_173295, int_173296)
    
    slice_173298 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1604, 10), n_173294, None, result_add_173297)
    
    # Call to reshape(...): (line 1604)
    # Processing the call arguments (line 1604)
    int_173301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 22), 'int')
    # Processing the call keyword arguments (line 1604)
    kwargs_173302 = {}
    # Getting the type of 'mat' (line 1604)
    mat_173299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1604)
    reshape_173300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 10), mat_173299, 'reshape')
    # Calling reshape(args, kwargs) (line 1604)
    reshape_call_result_173303 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 10), reshape_173300, *[int_173301], **kwargs_173302)
    
    # Obtaining the member '__getitem__' of a type (line 1604)
    getitem___173304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 10), reshape_call_result_173303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1604)
    subscript_call_result_173305 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 10), getitem___173304, slice_173298)
    
    # Assigning a type to the variable 'bot' (line 1604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1604, 4), 'bot', subscript_call_result_173305)
    
    # Assigning a UnaryOp to a Subscript (line 1605):
    
    # Assigning a UnaryOp to a Subscript (line 1605):
    
    
    # Call to arange(...): (line 1605)
    # Processing the call arguments (line 1605)
    int_173308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 26), 'int')
    # Getting the type of 'n' (line 1605)
    n_173309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 29), 'n', False)
    # Processing the call keyword arguments (line 1605)
    kwargs_173310 = {}
    # Getting the type of 'np' (line 1605)
    np_173306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 1605)
    arange_173307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1605, 16), np_173306, 'arange')
    # Calling arange(args, kwargs) (line 1605)
    arange_call_result_173311 = invoke(stypy.reporting.localization.Localization(__file__, 1605, 16), arange_173307, *[int_173308, n_173309], **kwargs_173310)
    
    # Applying the 'usub' unary operator (line 1605)
    result___neg___173312 = python_operator(stypy.reporting.localization.Localization(__file__, 1605, 15), 'usub', arange_call_result_173311)
    
    # Getting the type of 'top' (line 1605)
    top_173313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 4), 'top')
    Ellipsis_173314 = Ellipsis
    # Storing an element on a container (line 1605)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1605, 4), top_173313, (Ellipsis_173314, result___neg___173312))
    
    # Assigning a BinOp to a Subscript (line 1606):
    
    # Assigning a BinOp to a Subscript (line 1606):
    float_173315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 15), 'float')
    
    # Call to arange(...): (line 1606)
    # Processing the call arguments (line 1606)
    # Getting the type of 'n' (line 1606)
    n_173318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 28), 'n', False)
    # Processing the call keyword arguments (line 1606)
    kwargs_173319 = {}
    # Getting the type of 'np' (line 1606)
    np_173316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 1606)
    arange_173317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1606, 18), np_173316, 'arange')
    # Calling arange(args, kwargs) (line 1606)
    arange_call_result_173320 = invoke(stypy.reporting.localization.Localization(__file__, 1606, 18), arange_173317, *[n_173318], **kwargs_173319)
    
    # Applying the binary operator '*' (line 1606)
    result_mul_173321 = python_operator(stypy.reporting.localization.Localization(__file__, 1606, 15), '*', float_173315, arange_call_result_173320)
    
    float_173322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 33), 'float')
    # Applying the binary operator '+' (line 1606)
    result_add_173323 = python_operator(stypy.reporting.localization.Localization(__file__, 1606, 15), '+', result_mul_173321, float_173322)
    
    # Getting the type of 'mid' (line 1606)
    mid_173324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 4), 'mid')
    Ellipsis_173325 = Ellipsis
    # Storing an element on a container (line 1606)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1606, 4), mid_173324, (Ellipsis_173325, result_add_173323))
    
    # Assigning a Name to a Subscript (line 1607):
    
    # Assigning a Name to a Subscript (line 1607):
    # Getting the type of 'top' (line 1607)
    top_173326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 15), 'top')
    # Getting the type of 'bot' (line 1607)
    bot_173327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 4), 'bot')
    Ellipsis_173328 = Ellipsis
    # Storing an element on a container (line 1607)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1607, 4), bot_173327, (Ellipsis_173328, top_173326))
    
    # Getting the type of 'mat' (line 1608)
    mat_173329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'mat')
    
    # Obtaining the type of the subscript
    slice_173330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1608, 4), None, None, None)
    int_173331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 11), 'int')
    # Getting the type of 'mat' (line 1608)
    mat_173332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'mat')
    # Obtaining the member '__getitem__' of a type (line 1608)
    getitem___173333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 4), mat_173332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1608)
    subscript_call_result_173334 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 4), getitem___173333, (slice_173330, int_173331))
    
    
    # Obtaining the type of the subscript
    int_173335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 22), 'int')
    slice_173336 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1608, 19), None, int_173335, None)
    # Getting the type of 'c' (line 1608)
    c_173337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 1608)
    getitem___173338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 19), c_173337, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1608)
    subscript_call_result_173339 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 19), getitem___173338, slice_173336)
    
    
    # Obtaining the type of the subscript
    int_173340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 28), 'int')
    # Getting the type of 'c' (line 1608)
    c_173341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 26), 'c')
    # Obtaining the member '__getitem__' of a type (line 1608)
    getitem___173342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 26), c_173341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1608)
    subscript_call_result_173343 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 26), getitem___173342, int_173340)
    
    # Applying the binary operator 'div' (line 1608)
    result_div_173344 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 19), 'div', subscript_call_result_173339, subscript_call_result_173343)
    
    # Getting the type of 'n' (line 1608)
    n_173345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 33), 'n')
    # Applying the binary operator '*' (line 1608)
    result_mul_173346 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 18), '*', result_div_173344, n_173345)
    
    # Applying the binary operator '+=' (line 1608)
    result_iadd_173347 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 4), '+=', subscript_call_result_173334, result_mul_173346)
    # Getting the type of 'mat' (line 1608)
    mat_173348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'mat')
    slice_173349 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1608, 4), None, None, None)
    int_173350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 11), 'int')
    # Storing an element on a container (line 1608)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1608, 4), mat_173348, ((slice_173349, int_173350), result_iadd_173347))
    
    # Getting the type of 'mat' (line 1609)
    mat_173351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 11), 'mat')
    # Assigning a type to the variable 'stypy_return_type' (line 1609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1609, 4), 'stypy_return_type', mat_173351)
    
    # ################# End of 'lagcompanion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagcompanion' in the type store
    # Getting the type of 'stypy_return_type' (line 1568)
    stypy_return_type_173352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1568, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagcompanion'
    return stypy_return_type_173352

# Assigning a type to the variable 'lagcompanion' (line 1568)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1568, 0), 'lagcompanion', lagcompanion)

@norecursion
def lagroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagroots'
    module_type_store = module_type_store.open_function_context('lagroots', 1612, 0, False)
    
    # Passed parameters checking function
    lagroots.stypy_localization = localization
    lagroots.stypy_type_of_self = None
    lagroots.stypy_type_store = module_type_store
    lagroots.stypy_function_name = 'lagroots'
    lagroots.stypy_param_names_list = ['c']
    lagroots.stypy_varargs_param_name = None
    lagroots.stypy_kwargs_param_name = None
    lagroots.stypy_call_defaults = defaults
    lagroots.stypy_call_varargs = varargs
    lagroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagroots', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagroots', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagroots(...)' code ##################

    str_173353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, (-1)), 'str', '\n    Compute the roots of a Laguerre series.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * L_i(x).\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the series. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    polyroots, legroots, chebroots, hermroots, hermeroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the series for such\n    values. Roots with multiplicity greater than 1 will also show larger\n    errors as the value of the series near such points is relatively\n    insensitive to errors in the roots. Isolated roots near the origin can\n    be improved by a few iterations of Newton\'s method.\n\n    The Laguerre series basis polynomials aren\'t powers of `x` so the\n    results of this function may seem unintuitive.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagroots, lagfromroots\n    >>> coef = lagfromroots([0, 1, 2])\n    >>> coef\n    array([  2.,  -8.,  12.,  -6.])\n    >>> lagroots(coef)\n    array([ -4.44089210e-16,   1.00000000e+00,   2.00000000e+00])\n\n    ')
    
    # Assigning a Call to a List (line 1659):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1659)
    # Processing the call arguments (line 1659)
    
    # Obtaining an instance of the builtin type 'list' (line 1659)
    list_173356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1659)
    # Adding element type (line 1659)
    # Getting the type of 'c' (line 1659)
    c_173357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1659, 23), list_173356, c_173357)
    
    # Processing the call keyword arguments (line 1659)
    kwargs_173358 = {}
    # Getting the type of 'pu' (line 1659)
    pu_173354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1659)
    as_series_173355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1659, 10), pu_173354, 'as_series')
    # Calling as_series(args, kwargs) (line 1659)
    as_series_call_result_173359 = invoke(stypy.reporting.localization.Localization(__file__, 1659, 10), as_series_173355, *[list_173356], **kwargs_173358)
    
    # Assigning a type to the variable 'call_assignment_170815' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_170815', as_series_call_result_173359)
    
    # Assigning a Call to a Name (line 1659):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_173362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 4), 'int')
    # Processing the call keyword arguments
    kwargs_173363 = {}
    # Getting the type of 'call_assignment_170815' (line 1659)
    call_assignment_170815_173360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_170815', False)
    # Obtaining the member '__getitem__' of a type (line 1659)
    getitem___173361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1659, 4), call_assignment_170815_173360, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_173364 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173361, *[int_173362], **kwargs_173363)
    
    # Assigning a type to the variable 'call_assignment_170816' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_170816', getitem___call_result_173364)
    
    # Assigning a Name to a Name (line 1659):
    # Getting the type of 'call_assignment_170816' (line 1659)
    call_assignment_170816_173365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_170816')
    # Assigning a type to the variable 'c' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 5), 'c', call_assignment_170816_173365)
    
    
    
    # Call to len(...): (line 1660)
    # Processing the call arguments (line 1660)
    # Getting the type of 'c' (line 1660)
    c_173367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 11), 'c', False)
    # Processing the call keyword arguments (line 1660)
    kwargs_173368 = {}
    # Getting the type of 'len' (line 1660)
    len_173366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 7), 'len', False)
    # Calling len(args, kwargs) (line 1660)
    len_call_result_173369 = invoke(stypy.reporting.localization.Localization(__file__, 1660, 7), len_173366, *[c_173367], **kwargs_173368)
    
    int_173370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 17), 'int')
    # Applying the binary operator '<=' (line 1660)
    result_le_173371 = python_operator(stypy.reporting.localization.Localization(__file__, 1660, 7), '<=', len_call_result_173369, int_173370)
    
    # Testing the type of an if condition (line 1660)
    if_condition_173372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1660, 4), result_le_173371)
    # Assigning a type to the variable 'if_condition_173372' (line 1660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1660, 4), 'if_condition_173372', if_condition_173372)
    # SSA begins for if statement (line 1660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1661)
    # Processing the call arguments (line 1661)
    
    # Obtaining an instance of the builtin type 'list' (line 1661)
    list_173375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1661)
    
    # Processing the call keyword arguments (line 1661)
    # Getting the type of 'c' (line 1661)
    c_173376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 34), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1661)
    dtype_173377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1661, 34), c_173376, 'dtype')
    keyword_173378 = dtype_173377
    kwargs_173379 = {'dtype': keyword_173378}
    # Getting the type of 'np' (line 1661)
    np_173373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1661)
    array_173374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1661, 15), np_173373, 'array')
    # Calling array(args, kwargs) (line 1661)
    array_call_result_173380 = invoke(stypy.reporting.localization.Localization(__file__, 1661, 15), array_173374, *[list_173375], **kwargs_173379)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1661, 8), 'stypy_return_type', array_call_result_173380)
    # SSA join for if statement (line 1660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1662)
    # Processing the call arguments (line 1662)
    # Getting the type of 'c' (line 1662)
    c_173382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 11), 'c', False)
    # Processing the call keyword arguments (line 1662)
    kwargs_173383 = {}
    # Getting the type of 'len' (line 1662)
    len_173381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 7), 'len', False)
    # Calling len(args, kwargs) (line 1662)
    len_call_result_173384 = invoke(stypy.reporting.localization.Localization(__file__, 1662, 7), len_173381, *[c_173382], **kwargs_173383)
    
    int_173385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 17), 'int')
    # Applying the binary operator '==' (line 1662)
    result_eq_173386 = python_operator(stypy.reporting.localization.Localization(__file__, 1662, 7), '==', len_call_result_173384, int_173385)
    
    # Testing the type of an if condition (line 1662)
    if_condition_173387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1662, 4), result_eq_173386)
    # Assigning a type to the variable 'if_condition_173387' (line 1662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1662, 4), 'if_condition_173387', if_condition_173387)
    # SSA begins for if statement (line 1662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1663)
    # Processing the call arguments (line 1663)
    
    # Obtaining an instance of the builtin type 'list' (line 1663)
    list_173390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1663)
    # Adding element type (line 1663)
    int_173391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 25), 'int')
    
    # Obtaining the type of the subscript
    int_173392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 31), 'int')
    # Getting the type of 'c' (line 1663)
    c_173393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1663)
    getitem___173394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 29), c_173393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1663)
    subscript_call_result_173395 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 29), getitem___173394, int_173392)
    
    
    # Obtaining the type of the subscript
    int_173396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 36), 'int')
    # Getting the type of 'c' (line 1663)
    c_173397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 34), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1663)
    getitem___173398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 34), c_173397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1663)
    subscript_call_result_173399 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 34), getitem___173398, int_173396)
    
    # Applying the binary operator 'div' (line 1663)
    result_div_173400 = python_operator(stypy.reporting.localization.Localization(__file__, 1663, 29), 'div', subscript_call_result_173395, subscript_call_result_173399)
    
    # Applying the binary operator '+' (line 1663)
    result_add_173401 = python_operator(stypy.reporting.localization.Localization(__file__, 1663, 25), '+', int_173391, result_div_173400)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 24), list_173390, result_add_173401)
    
    # Processing the call keyword arguments (line 1663)
    kwargs_173402 = {}
    # Getting the type of 'np' (line 1663)
    np_173388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1663)
    array_173389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 15), np_173388, 'array')
    # Calling array(args, kwargs) (line 1663)
    array_call_result_173403 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 15), array_173389, *[list_173390], **kwargs_173402)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1663, 8), 'stypy_return_type', array_call_result_173403)
    # SSA join for if statement (line 1662)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1665):
    
    # Assigning a Call to a Name (line 1665):
    
    # Call to lagcompanion(...): (line 1665)
    # Processing the call arguments (line 1665)
    # Getting the type of 'c' (line 1665)
    c_173405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 21), 'c', False)
    # Processing the call keyword arguments (line 1665)
    kwargs_173406 = {}
    # Getting the type of 'lagcompanion' (line 1665)
    lagcompanion_173404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 8), 'lagcompanion', False)
    # Calling lagcompanion(args, kwargs) (line 1665)
    lagcompanion_call_result_173407 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 8), lagcompanion_173404, *[c_173405], **kwargs_173406)
    
    # Assigning a type to the variable 'm' (line 1665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1665, 4), 'm', lagcompanion_call_result_173407)
    
    # Assigning a Call to a Name (line 1666):
    
    # Assigning a Call to a Name (line 1666):
    
    # Call to eigvals(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'm' (line 1666)
    m_173410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 19), 'm', False)
    # Processing the call keyword arguments (line 1666)
    kwargs_173411 = {}
    # Getting the type of 'la' (line 1666)
    la_173408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 8), 'la', False)
    # Obtaining the member 'eigvals' of a type (line 1666)
    eigvals_173409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 8), la_173408, 'eigvals')
    # Calling eigvals(args, kwargs) (line 1666)
    eigvals_call_result_173412 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 8), eigvals_173409, *[m_173410], **kwargs_173411)
    
    # Assigning a type to the variable 'r' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'r', eigvals_call_result_173412)
    
    # Call to sort(...): (line 1667)
    # Processing the call keyword arguments (line 1667)
    kwargs_173415 = {}
    # Getting the type of 'r' (line 1667)
    r_173413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 4), 'r', False)
    # Obtaining the member 'sort' of a type (line 1667)
    sort_173414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1667, 4), r_173413, 'sort')
    # Calling sort(args, kwargs) (line 1667)
    sort_call_result_173416 = invoke(stypy.reporting.localization.Localization(__file__, 1667, 4), sort_173414, *[], **kwargs_173415)
    
    # Getting the type of 'r' (line 1668)
    r_173417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 1668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1668, 4), 'stypy_return_type', r_173417)
    
    # ################# End of 'lagroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagroots' in the type store
    # Getting the type of 'stypy_return_type' (line 1612)
    stypy_return_type_173418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1612, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173418)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagroots'
    return stypy_return_type_173418

# Assigning a type to the variable 'lagroots' (line 1612)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1612, 0), 'lagroots', lagroots)

@norecursion
def laggauss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'laggauss'
    module_type_store = module_type_store.open_function_context('laggauss', 1671, 0, False)
    
    # Passed parameters checking function
    laggauss.stypy_localization = localization
    laggauss.stypy_type_of_self = None
    laggauss.stypy_type_store = module_type_store
    laggauss.stypy_function_name = 'laggauss'
    laggauss.stypy_param_names_list = ['deg']
    laggauss.stypy_varargs_param_name = None
    laggauss.stypy_kwargs_param_name = None
    laggauss.stypy_call_defaults = defaults
    laggauss.stypy_call_varargs = varargs
    laggauss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'laggauss', ['deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'laggauss', localization, ['deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'laggauss(...)' code ##################

    str_173419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1706, (-1)), 'str', "\n    Gauss-Laguerre quadrature.\n\n    Computes the sample points and weights for Gauss-Laguerre quadrature.\n    These sample points and weights will correctly integrate polynomials of\n    degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`\n    with the weight function :math:`f(x) = \\exp(-x)`.\n\n    Parameters\n    ----------\n    deg : int\n        Number of sample points and weights. It must be >= 1.\n\n    Returns\n    -------\n    x : ndarray\n        1-D ndarray containing the sample points.\n    y : ndarray\n        1-D ndarray containing the weights.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    The results have only been tested up to degree 100 higher degrees may\n    be problematic. The weights are determined by using the fact that\n\n    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))\n\n    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`\n    is the k'th root of :math:`L_n`, and then scaling the results to get\n    the right value when integrating 1.\n\n    ")
    
    # Assigning a Call to a Name (line 1707):
    
    # Assigning a Call to a Name (line 1707):
    
    # Call to int(...): (line 1707)
    # Processing the call arguments (line 1707)
    # Getting the type of 'deg' (line 1707)
    deg_173421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1707, 15), 'deg', False)
    # Processing the call keyword arguments (line 1707)
    kwargs_173422 = {}
    # Getting the type of 'int' (line 1707)
    int_173420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1707, 11), 'int', False)
    # Calling int(args, kwargs) (line 1707)
    int_call_result_173423 = invoke(stypy.reporting.localization.Localization(__file__, 1707, 11), int_173420, *[deg_173421], **kwargs_173422)
    
    # Assigning a type to the variable 'ideg' (line 1707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1707, 4), 'ideg', int_call_result_173423)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ideg' (line 1708)
    ideg_173424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 7), 'ideg')
    # Getting the type of 'deg' (line 1708)
    deg_173425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 15), 'deg')
    # Applying the binary operator '!=' (line 1708)
    result_ne_173426 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 7), '!=', ideg_173424, deg_173425)
    
    
    # Getting the type of 'ideg' (line 1708)
    ideg_173427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 22), 'ideg')
    int_173428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1708, 29), 'int')
    # Applying the binary operator '<' (line 1708)
    result_lt_173429 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 22), '<', ideg_173427, int_173428)
    
    # Applying the binary operator 'or' (line 1708)
    result_or_keyword_173430 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 7), 'or', result_ne_173426, result_lt_173429)
    
    # Testing the type of an if condition (line 1708)
    if_condition_173431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1708, 4), result_or_keyword_173430)
    # Assigning a type to the variable 'if_condition_173431' (line 1708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1708, 4), 'if_condition_173431', if_condition_173431)
    # SSA begins for if statement (line 1708)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1709)
    # Processing the call arguments (line 1709)
    str_173433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1709, 25), 'str', 'deg must be a non-negative integer')
    # Processing the call keyword arguments (line 1709)
    kwargs_173434 = {}
    # Getting the type of 'ValueError' (line 1709)
    ValueError_173432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1709)
    ValueError_call_result_173435 = invoke(stypy.reporting.localization.Localization(__file__, 1709, 14), ValueError_173432, *[str_173433], **kwargs_173434)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1709, 8), ValueError_call_result_173435, 'raise parameter', BaseException)
    # SSA join for if statement (line 1708)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1713):
    
    # Assigning a Call to a Name (line 1713):
    
    # Call to array(...): (line 1713)
    # Processing the call arguments (line 1713)
    
    # Obtaining an instance of the builtin type 'list' (line 1713)
    list_173438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1713, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1713)
    # Adding element type (line 1713)
    int_173439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1713, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1713, 17), list_173438, int_173439)
    
    # Getting the type of 'deg' (line 1713)
    deg_173440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1713, 21), 'deg', False)
    # Applying the binary operator '*' (line 1713)
    result_mul_173441 = python_operator(stypy.reporting.localization.Localization(__file__, 1713, 17), '*', list_173438, deg_173440)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1713)
    list_173442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1713, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1713)
    # Adding element type (line 1713)
    int_173443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1713, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1713, 27), list_173442, int_173443)
    
    # Applying the binary operator '+' (line 1713)
    result_add_173444 = python_operator(stypy.reporting.localization.Localization(__file__, 1713, 17), '+', result_mul_173441, list_173442)
    
    # Processing the call keyword arguments (line 1713)
    kwargs_173445 = {}
    # Getting the type of 'np' (line 1713)
    np_173436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1713, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1713)
    array_173437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1713, 8), np_173436, 'array')
    # Calling array(args, kwargs) (line 1713)
    array_call_result_173446 = invoke(stypy.reporting.localization.Localization(__file__, 1713, 8), array_173437, *[result_add_173444], **kwargs_173445)
    
    # Assigning a type to the variable 'c' (line 1713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1713, 4), 'c', array_call_result_173446)
    
    # Assigning a Call to a Name (line 1714):
    
    # Assigning a Call to a Name (line 1714):
    
    # Call to lagcompanion(...): (line 1714)
    # Processing the call arguments (line 1714)
    # Getting the type of 'c' (line 1714)
    c_173448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1714, 21), 'c', False)
    # Processing the call keyword arguments (line 1714)
    kwargs_173449 = {}
    # Getting the type of 'lagcompanion' (line 1714)
    lagcompanion_173447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1714, 8), 'lagcompanion', False)
    # Calling lagcompanion(args, kwargs) (line 1714)
    lagcompanion_call_result_173450 = invoke(stypy.reporting.localization.Localization(__file__, 1714, 8), lagcompanion_173447, *[c_173448], **kwargs_173449)
    
    # Assigning a type to the variable 'm' (line 1714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1714, 4), 'm', lagcompanion_call_result_173450)
    
    # Assigning a Call to a Name (line 1715):
    
    # Assigning a Call to a Name (line 1715):
    
    # Call to eigvalsh(...): (line 1715)
    # Processing the call arguments (line 1715)
    # Getting the type of 'm' (line 1715)
    m_173453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1715, 20), 'm', False)
    # Processing the call keyword arguments (line 1715)
    kwargs_173454 = {}
    # Getting the type of 'la' (line 1715)
    la_173451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1715, 8), 'la', False)
    # Obtaining the member 'eigvalsh' of a type (line 1715)
    eigvalsh_173452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1715, 8), la_173451, 'eigvalsh')
    # Calling eigvalsh(args, kwargs) (line 1715)
    eigvalsh_call_result_173455 = invoke(stypy.reporting.localization.Localization(__file__, 1715, 8), eigvalsh_173452, *[m_173453], **kwargs_173454)
    
    # Assigning a type to the variable 'x' (line 1715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1715, 4), 'x', eigvalsh_call_result_173455)
    
    # Assigning a Call to a Name (line 1718):
    
    # Assigning a Call to a Name (line 1718):
    
    # Call to lagval(...): (line 1718)
    # Processing the call arguments (line 1718)
    # Getting the type of 'x' (line 1718)
    x_173457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 16), 'x', False)
    # Getting the type of 'c' (line 1718)
    c_173458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 19), 'c', False)
    # Processing the call keyword arguments (line 1718)
    kwargs_173459 = {}
    # Getting the type of 'lagval' (line 1718)
    lagval_173456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 9), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1718)
    lagval_call_result_173460 = invoke(stypy.reporting.localization.Localization(__file__, 1718, 9), lagval_173456, *[x_173457, c_173458], **kwargs_173459)
    
    # Assigning a type to the variable 'dy' (line 1718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1718, 4), 'dy', lagval_call_result_173460)
    
    # Assigning a Call to a Name (line 1719):
    
    # Assigning a Call to a Name (line 1719):
    
    # Call to lagval(...): (line 1719)
    # Processing the call arguments (line 1719)
    # Getting the type of 'x' (line 1719)
    x_173462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1719, 16), 'x', False)
    
    # Call to lagder(...): (line 1719)
    # Processing the call arguments (line 1719)
    # Getting the type of 'c' (line 1719)
    c_173464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1719, 26), 'c', False)
    # Processing the call keyword arguments (line 1719)
    kwargs_173465 = {}
    # Getting the type of 'lagder' (line 1719)
    lagder_173463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1719, 19), 'lagder', False)
    # Calling lagder(args, kwargs) (line 1719)
    lagder_call_result_173466 = invoke(stypy.reporting.localization.Localization(__file__, 1719, 19), lagder_173463, *[c_173464], **kwargs_173465)
    
    # Processing the call keyword arguments (line 1719)
    kwargs_173467 = {}
    # Getting the type of 'lagval' (line 1719)
    lagval_173461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1719, 9), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1719)
    lagval_call_result_173468 = invoke(stypy.reporting.localization.Localization(__file__, 1719, 9), lagval_173461, *[x_173462, lagder_call_result_173466], **kwargs_173467)
    
    # Assigning a type to the variable 'df' (line 1719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1719, 4), 'df', lagval_call_result_173468)
    
    # Getting the type of 'x' (line 1720)
    x_173469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1720, 4), 'x')
    # Getting the type of 'dy' (line 1720)
    dy_173470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1720, 9), 'dy')
    # Getting the type of 'df' (line 1720)
    df_173471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1720, 12), 'df')
    # Applying the binary operator 'div' (line 1720)
    result_div_173472 = python_operator(stypy.reporting.localization.Localization(__file__, 1720, 9), 'div', dy_173470, df_173471)
    
    # Applying the binary operator '-=' (line 1720)
    result_isub_173473 = python_operator(stypy.reporting.localization.Localization(__file__, 1720, 4), '-=', x_173469, result_div_173472)
    # Assigning a type to the variable 'x' (line 1720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1720, 4), 'x', result_isub_173473)
    
    
    # Assigning a Call to a Name (line 1724):
    
    # Assigning a Call to a Name (line 1724):
    
    # Call to lagval(...): (line 1724)
    # Processing the call arguments (line 1724)
    # Getting the type of 'x' (line 1724)
    x_173475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1724, 16), 'x', False)
    
    # Obtaining the type of the subscript
    int_173476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1724, 21), 'int')
    slice_173477 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1724, 19), int_173476, None, None)
    # Getting the type of 'c' (line 1724)
    c_173478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1724, 19), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1724)
    getitem___173479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1724, 19), c_173478, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1724)
    subscript_call_result_173480 = invoke(stypy.reporting.localization.Localization(__file__, 1724, 19), getitem___173479, slice_173477)
    
    # Processing the call keyword arguments (line 1724)
    kwargs_173481 = {}
    # Getting the type of 'lagval' (line 1724)
    lagval_173474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1724, 9), 'lagval', False)
    # Calling lagval(args, kwargs) (line 1724)
    lagval_call_result_173482 = invoke(stypy.reporting.localization.Localization(__file__, 1724, 9), lagval_173474, *[x_173475, subscript_call_result_173480], **kwargs_173481)
    
    # Assigning a type to the variable 'fm' (line 1724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1724, 4), 'fm', lagval_call_result_173482)
    
    # Getting the type of 'fm' (line 1725)
    fm_173483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1725, 4), 'fm')
    
    # Call to max(...): (line 1725)
    # Processing the call keyword arguments (line 1725)
    kwargs_173490 = {}
    
    # Call to abs(...): (line 1725)
    # Processing the call arguments (line 1725)
    # Getting the type of 'fm' (line 1725)
    fm_173486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1725, 17), 'fm', False)
    # Processing the call keyword arguments (line 1725)
    kwargs_173487 = {}
    # Getting the type of 'np' (line 1725)
    np_173484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1725, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1725)
    abs_173485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1725, 10), np_173484, 'abs')
    # Calling abs(args, kwargs) (line 1725)
    abs_call_result_173488 = invoke(stypy.reporting.localization.Localization(__file__, 1725, 10), abs_173485, *[fm_173486], **kwargs_173487)
    
    # Obtaining the member 'max' of a type (line 1725)
    max_173489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1725, 10), abs_call_result_173488, 'max')
    # Calling max(args, kwargs) (line 1725)
    max_call_result_173491 = invoke(stypy.reporting.localization.Localization(__file__, 1725, 10), max_173489, *[], **kwargs_173490)
    
    # Applying the binary operator 'div=' (line 1725)
    result_div_173492 = python_operator(stypy.reporting.localization.Localization(__file__, 1725, 4), 'div=', fm_173483, max_call_result_173491)
    # Assigning a type to the variable 'fm' (line 1725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1725, 4), 'fm', result_div_173492)
    
    
    # Getting the type of 'df' (line 1726)
    df_173493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1726, 4), 'df')
    
    # Call to max(...): (line 1726)
    # Processing the call keyword arguments (line 1726)
    kwargs_173500 = {}
    
    # Call to abs(...): (line 1726)
    # Processing the call arguments (line 1726)
    # Getting the type of 'df' (line 1726)
    df_173496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1726, 17), 'df', False)
    # Processing the call keyword arguments (line 1726)
    kwargs_173497 = {}
    # Getting the type of 'np' (line 1726)
    np_173494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1726, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1726)
    abs_173495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1726, 10), np_173494, 'abs')
    # Calling abs(args, kwargs) (line 1726)
    abs_call_result_173498 = invoke(stypy.reporting.localization.Localization(__file__, 1726, 10), abs_173495, *[df_173496], **kwargs_173497)
    
    # Obtaining the member 'max' of a type (line 1726)
    max_173499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1726, 10), abs_call_result_173498, 'max')
    # Calling max(args, kwargs) (line 1726)
    max_call_result_173501 = invoke(stypy.reporting.localization.Localization(__file__, 1726, 10), max_173499, *[], **kwargs_173500)
    
    # Applying the binary operator 'div=' (line 1726)
    result_div_173502 = python_operator(stypy.reporting.localization.Localization(__file__, 1726, 4), 'div=', df_173493, max_call_result_173501)
    # Assigning a type to the variable 'df' (line 1726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1726, 4), 'df', result_div_173502)
    
    
    # Assigning a BinOp to a Name (line 1727):
    
    # Assigning a BinOp to a Name (line 1727):
    int_173503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1727, 8), 'int')
    # Getting the type of 'fm' (line 1727)
    fm_173504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1727, 11), 'fm')
    # Getting the type of 'df' (line 1727)
    df_173505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1727, 16), 'df')
    # Applying the binary operator '*' (line 1727)
    result_mul_173506 = python_operator(stypy.reporting.localization.Localization(__file__, 1727, 11), '*', fm_173504, df_173505)
    
    # Applying the binary operator 'div' (line 1727)
    result_div_173507 = python_operator(stypy.reporting.localization.Localization(__file__, 1727, 8), 'div', int_173503, result_mul_173506)
    
    # Assigning a type to the variable 'w' (line 1727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1727, 4), 'w', result_div_173507)
    
    # Getting the type of 'w' (line 1730)
    w_173508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1730, 4), 'w')
    
    # Call to sum(...): (line 1730)
    # Processing the call keyword arguments (line 1730)
    kwargs_173511 = {}
    # Getting the type of 'w' (line 1730)
    w_173509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1730, 9), 'w', False)
    # Obtaining the member 'sum' of a type (line 1730)
    sum_173510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1730, 9), w_173509, 'sum')
    # Calling sum(args, kwargs) (line 1730)
    sum_call_result_173512 = invoke(stypy.reporting.localization.Localization(__file__, 1730, 9), sum_173510, *[], **kwargs_173511)
    
    # Applying the binary operator 'div=' (line 1730)
    result_div_173513 = python_operator(stypy.reporting.localization.Localization(__file__, 1730, 4), 'div=', w_173508, sum_call_result_173512)
    # Assigning a type to the variable 'w' (line 1730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1730, 4), 'w', result_div_173513)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1732)
    tuple_173514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1732, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1732)
    # Adding element type (line 1732)
    # Getting the type of 'x' (line 1732)
    x_173515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1732, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1732, 11), tuple_173514, x_173515)
    # Adding element type (line 1732)
    # Getting the type of 'w' (line 1732)
    w_173516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1732, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1732, 11), tuple_173514, w_173516)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1732, 4), 'stypy_return_type', tuple_173514)
    
    # ################# End of 'laggauss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'laggauss' in the type store
    # Getting the type of 'stypy_return_type' (line 1671)
    stypy_return_type_173517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1671, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'laggauss'
    return stypy_return_type_173517

# Assigning a type to the variable 'laggauss' (line 1671)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1671, 0), 'laggauss', laggauss)

@norecursion
def lagweight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagweight'
    module_type_store = module_type_store.open_function_context('lagweight', 1735, 0, False)
    
    # Passed parameters checking function
    lagweight.stypy_localization = localization
    lagweight.stypy_type_of_self = None
    lagweight.stypy_type_store = module_type_store
    lagweight.stypy_function_name = 'lagweight'
    lagweight.stypy_param_names_list = ['x']
    lagweight.stypy_varargs_param_name = None
    lagweight.stypy_kwargs_param_name = None
    lagweight.stypy_call_defaults = defaults
    lagweight.stypy_call_varargs = varargs
    lagweight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagweight', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagweight', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagweight(...)' code ##################

    str_173518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1757, (-1)), 'str', 'Weight function of the Laguerre polynomials.\n\n    The weight function is :math:`exp(-x)` and the interval of integration\n    is :math:`[0, \\inf]`. The Laguerre polynomials are orthogonal, but not\n    normalized, with respect to this weight function.\n\n    Parameters\n    ----------\n    x : array_like\n       Values at which the weight function will be computed.\n\n    Returns\n    -------\n    w : ndarray\n       The weight function at `x`.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a Name (line 1758):
    
    # Assigning a Call to a Name (line 1758):
    
    # Call to exp(...): (line 1758)
    # Processing the call arguments (line 1758)
    
    # Getting the type of 'x' (line 1758)
    x_173521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 16), 'x', False)
    # Applying the 'usub' unary operator (line 1758)
    result___neg___173522 = python_operator(stypy.reporting.localization.Localization(__file__, 1758, 15), 'usub', x_173521)
    
    # Processing the call keyword arguments (line 1758)
    kwargs_173523 = {}
    # Getting the type of 'np' (line 1758)
    np_173519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 8), 'np', False)
    # Obtaining the member 'exp' of a type (line 1758)
    exp_173520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1758, 8), np_173519, 'exp')
    # Calling exp(args, kwargs) (line 1758)
    exp_call_result_173524 = invoke(stypy.reporting.localization.Localization(__file__, 1758, 8), exp_173520, *[result___neg___173522], **kwargs_173523)
    
    # Assigning a type to the variable 'w' (line 1758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1758, 4), 'w', exp_call_result_173524)
    # Getting the type of 'w' (line 1759)
    w_173525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 11), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 1759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1759, 4), 'stypy_return_type', w_173525)
    
    # ################# End of 'lagweight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagweight' in the type store
    # Getting the type of 'stypy_return_type' (line 1735)
    stypy_return_type_173526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1735, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_173526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagweight'
    return stypy_return_type_173526

# Assigning a type to the variable 'lagweight' (line 1735)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1735, 0), 'lagweight', lagweight)
# Declaration of the 'Laguerre' class
# Getting the type of 'ABCPolyBase' (line 1765)
ABCPolyBase_173527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 15), 'ABCPolyBase')

class Laguerre(ABCPolyBase_173527, ):
    str_173528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1786, (-1)), 'str', "A Laguerre series class.\n\n    The Laguerre class provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the\n    attributes and methods listed in the `ABCPolyBase` documentation.\n\n    Parameters\n    ----------\n    coef : array_like\n        Laguerre coefficients in order of increasing degree, i.e,\n        ``(1, 2, 3)`` gives ``1*L_0(x) + 2*L_1(X) + 3*L_2(x)``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is [0, 1].\n    window : (2,) array_like, optional\n        Window, see `domain` for its use. The default value is [0, 1].\n\n        .. versionadded:: 1.6.0\n\n    ")
    
    # Assigning a Call to a Name (line 1788):
    
    # Assigning a Call to a Name (line 1789):
    
    # Assigning a Call to a Name (line 1790):
    
    # Assigning a Call to a Name (line 1791):
    
    # Assigning a Call to a Name (line 1792):
    
    # Assigning a Call to a Name (line 1793):
    
    # Assigning a Call to a Name (line 1794):
    
    # Assigning a Call to a Name (line 1795):
    
    # Assigning a Call to a Name (line 1796):
    
    # Assigning a Call to a Name (line 1797):
    
    # Assigning a Call to a Name (line 1798):
    
    # Assigning a Call to a Name (line 1799):
    
    # Assigning a Str to a Name (line 1802):
    
    # Assigning a Call to a Name (line 1803):
    
    # Assigning a Call to a Name (line 1804):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1765, 0, False)
        # Assigning a type to the variable 'self' (line 1766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1766, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Laguerre.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Laguerre' (line 1765)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1765, 0), 'Laguerre', Laguerre)

# Assigning a Call to a Name (line 1788):

# Call to staticmethod(...): (line 1788)
# Processing the call arguments (line 1788)
# Getting the type of 'lagadd' (line 1788)
lagadd_173530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 24), 'lagadd', False)
# Processing the call keyword arguments (line 1788)
kwargs_173531 = {}
# Getting the type of 'staticmethod' (line 1788)
staticmethod_173529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1788)
staticmethod_call_result_173532 = invoke(stypy.reporting.localization.Localization(__file__, 1788, 11), staticmethod_173529, *[lagadd_173530], **kwargs_173531)

# Getting the type of 'Laguerre'
Laguerre_173533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173533, '_add', staticmethod_call_result_173532)

# Assigning a Call to a Name (line 1789):

# Call to staticmethod(...): (line 1789)
# Processing the call arguments (line 1789)
# Getting the type of 'lagsub' (line 1789)
lagsub_173535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1789, 24), 'lagsub', False)
# Processing the call keyword arguments (line 1789)
kwargs_173536 = {}
# Getting the type of 'staticmethod' (line 1789)
staticmethod_173534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1789, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1789)
staticmethod_call_result_173537 = invoke(stypy.reporting.localization.Localization(__file__, 1789, 11), staticmethod_173534, *[lagsub_173535], **kwargs_173536)

# Getting the type of 'Laguerre'
Laguerre_173538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_sub' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173538, '_sub', staticmethod_call_result_173537)

# Assigning a Call to a Name (line 1790):

# Call to staticmethod(...): (line 1790)
# Processing the call arguments (line 1790)
# Getting the type of 'lagmul' (line 1790)
lagmul_173540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1790, 24), 'lagmul', False)
# Processing the call keyword arguments (line 1790)
kwargs_173541 = {}
# Getting the type of 'staticmethod' (line 1790)
staticmethod_173539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1790, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1790)
staticmethod_call_result_173542 = invoke(stypy.reporting.localization.Localization(__file__, 1790, 11), staticmethod_173539, *[lagmul_173540], **kwargs_173541)

# Getting the type of 'Laguerre'
Laguerre_173543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_mul' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173543, '_mul', staticmethod_call_result_173542)

# Assigning a Call to a Name (line 1791):

# Call to staticmethod(...): (line 1791)
# Processing the call arguments (line 1791)
# Getting the type of 'lagdiv' (line 1791)
lagdiv_173545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1791, 24), 'lagdiv', False)
# Processing the call keyword arguments (line 1791)
kwargs_173546 = {}
# Getting the type of 'staticmethod' (line 1791)
staticmethod_173544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1791, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1791)
staticmethod_call_result_173547 = invoke(stypy.reporting.localization.Localization(__file__, 1791, 11), staticmethod_173544, *[lagdiv_173545], **kwargs_173546)

# Getting the type of 'Laguerre'
Laguerre_173548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_div' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173548, '_div', staticmethod_call_result_173547)

# Assigning a Call to a Name (line 1792):

# Call to staticmethod(...): (line 1792)
# Processing the call arguments (line 1792)
# Getting the type of 'lagpow' (line 1792)
lagpow_173550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1792, 24), 'lagpow', False)
# Processing the call keyword arguments (line 1792)
kwargs_173551 = {}
# Getting the type of 'staticmethod' (line 1792)
staticmethod_173549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1792, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1792)
staticmethod_call_result_173552 = invoke(stypy.reporting.localization.Localization(__file__, 1792, 11), staticmethod_173549, *[lagpow_173550], **kwargs_173551)

# Getting the type of 'Laguerre'
Laguerre_173553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_pow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173553, '_pow', staticmethod_call_result_173552)

# Assigning a Call to a Name (line 1793):

# Call to staticmethod(...): (line 1793)
# Processing the call arguments (line 1793)
# Getting the type of 'lagval' (line 1793)
lagval_173555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1793, 24), 'lagval', False)
# Processing the call keyword arguments (line 1793)
kwargs_173556 = {}
# Getting the type of 'staticmethod' (line 1793)
staticmethod_173554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1793, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1793)
staticmethod_call_result_173557 = invoke(stypy.reporting.localization.Localization(__file__, 1793, 11), staticmethod_173554, *[lagval_173555], **kwargs_173556)

# Getting the type of 'Laguerre'
Laguerre_173558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_val' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173558, '_val', staticmethod_call_result_173557)

# Assigning a Call to a Name (line 1794):

# Call to staticmethod(...): (line 1794)
# Processing the call arguments (line 1794)
# Getting the type of 'lagint' (line 1794)
lagint_173560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1794, 24), 'lagint', False)
# Processing the call keyword arguments (line 1794)
kwargs_173561 = {}
# Getting the type of 'staticmethod' (line 1794)
staticmethod_173559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1794, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1794)
staticmethod_call_result_173562 = invoke(stypy.reporting.localization.Localization(__file__, 1794, 11), staticmethod_173559, *[lagint_173560], **kwargs_173561)

# Getting the type of 'Laguerre'
Laguerre_173563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_int' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173563, '_int', staticmethod_call_result_173562)

# Assigning a Call to a Name (line 1795):

# Call to staticmethod(...): (line 1795)
# Processing the call arguments (line 1795)
# Getting the type of 'lagder' (line 1795)
lagder_173565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1795, 24), 'lagder', False)
# Processing the call keyword arguments (line 1795)
kwargs_173566 = {}
# Getting the type of 'staticmethod' (line 1795)
staticmethod_173564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1795, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1795)
staticmethod_call_result_173567 = invoke(stypy.reporting.localization.Localization(__file__, 1795, 11), staticmethod_173564, *[lagder_173565], **kwargs_173566)

# Getting the type of 'Laguerre'
Laguerre_173568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_der' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173568, '_der', staticmethod_call_result_173567)

# Assigning a Call to a Name (line 1796):

# Call to staticmethod(...): (line 1796)
# Processing the call arguments (line 1796)
# Getting the type of 'lagfit' (line 1796)
lagfit_173570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1796, 24), 'lagfit', False)
# Processing the call keyword arguments (line 1796)
kwargs_173571 = {}
# Getting the type of 'staticmethod' (line 1796)
staticmethod_173569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1796, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1796)
staticmethod_call_result_173572 = invoke(stypy.reporting.localization.Localization(__file__, 1796, 11), staticmethod_173569, *[lagfit_173570], **kwargs_173571)

# Getting the type of 'Laguerre'
Laguerre_173573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_fit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173573, '_fit', staticmethod_call_result_173572)

# Assigning a Call to a Name (line 1797):

# Call to staticmethod(...): (line 1797)
# Processing the call arguments (line 1797)
# Getting the type of 'lagline' (line 1797)
lagline_173575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1797, 25), 'lagline', False)
# Processing the call keyword arguments (line 1797)
kwargs_173576 = {}
# Getting the type of 'staticmethod' (line 1797)
staticmethod_173574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1797, 12), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1797)
staticmethod_call_result_173577 = invoke(stypy.reporting.localization.Localization(__file__, 1797, 12), staticmethod_173574, *[lagline_173575], **kwargs_173576)

# Getting the type of 'Laguerre'
Laguerre_173578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_line' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173578, '_line', staticmethod_call_result_173577)

# Assigning a Call to a Name (line 1798):

# Call to staticmethod(...): (line 1798)
# Processing the call arguments (line 1798)
# Getting the type of 'lagroots' (line 1798)
lagroots_173580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1798, 26), 'lagroots', False)
# Processing the call keyword arguments (line 1798)
kwargs_173581 = {}
# Getting the type of 'staticmethod' (line 1798)
staticmethod_173579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1798, 13), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1798)
staticmethod_call_result_173582 = invoke(stypy.reporting.localization.Localization(__file__, 1798, 13), staticmethod_173579, *[lagroots_173580], **kwargs_173581)

# Getting the type of 'Laguerre'
Laguerre_173583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_roots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173583, '_roots', staticmethod_call_result_173582)

# Assigning a Call to a Name (line 1799):

# Call to staticmethod(...): (line 1799)
# Processing the call arguments (line 1799)
# Getting the type of 'lagfromroots' (line 1799)
lagfromroots_173585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1799, 30), 'lagfromroots', False)
# Processing the call keyword arguments (line 1799)
kwargs_173586 = {}
# Getting the type of 'staticmethod' (line 1799)
staticmethod_173584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1799, 17), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1799)
staticmethod_call_result_173587 = invoke(stypy.reporting.localization.Localization(__file__, 1799, 17), staticmethod_173584, *[lagfromroots_173585], **kwargs_173586)

# Getting the type of 'Laguerre'
Laguerre_173588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member '_fromroots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173588, '_fromroots', staticmethod_call_result_173587)

# Assigning a Str to a Name (line 1802):
str_173589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1802, 15), 'str', 'lag')
# Getting the type of 'Laguerre'
Laguerre_173590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member 'nickname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173590, 'nickname', str_173589)

# Assigning a Call to a Name (line 1803):

# Call to array(...): (line 1803)
# Processing the call arguments (line 1803)
# Getting the type of 'lagdomain' (line 1803)
lagdomain_173593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1803, 22), 'lagdomain', False)
# Processing the call keyword arguments (line 1803)
kwargs_173594 = {}
# Getting the type of 'np' (line 1803)
np_173591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1803, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1803)
array_173592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1803, 13), np_173591, 'array')
# Calling array(args, kwargs) (line 1803)
array_call_result_173595 = invoke(stypy.reporting.localization.Localization(__file__, 1803, 13), array_173592, *[lagdomain_173593], **kwargs_173594)

# Getting the type of 'Laguerre'
Laguerre_173596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member 'domain' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173596, 'domain', array_call_result_173595)

# Assigning a Call to a Name (line 1804):

# Call to array(...): (line 1804)
# Processing the call arguments (line 1804)
# Getting the type of 'lagdomain' (line 1804)
lagdomain_173599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1804, 22), 'lagdomain', False)
# Processing the call keyword arguments (line 1804)
kwargs_173600 = {}
# Getting the type of 'np' (line 1804)
np_173597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1804, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1804)
array_173598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1804, 13), np_173597, 'array')
# Calling array(args, kwargs) (line 1804)
array_call_result_173601 = invoke(stypy.reporting.localization.Localization(__file__, 1804, 13), array_173598, *[lagdomain_173599], **kwargs_173600)

# Getting the type of 'Laguerre'
Laguerre_173602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Laguerre')
# Setting the type of the member 'window' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Laguerre_173602, 'window', array_call_result_173601)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
