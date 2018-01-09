
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Objects for dealing with polynomials.
3: 
4: This module provides a number of objects (mostly functions) useful for
5: dealing with polynomials, including a `Polynomial` class that
6: encapsulates the usual arithmetic operations.  (General information
7: on how this module represents and works with polynomial objects is in
8: the docstring for its "parent" sub-package, `numpy.polynomial`).
9: 
10: Constants
11: ---------
12: - `polydomain` -- Polynomial default domain, [-1,1].
13: - `polyzero` -- (Coefficients of the) "zero polynomial."
14: - `polyone` -- (Coefficients of the) constant polynomial 1.
15: - `polyx` -- (Coefficients of the) identity map polynomial, ``f(x) = x``.
16: 
17: Arithmetic
18: ----------
19: - `polyadd` -- add two polynomials.
20: - `polysub` -- subtract one polynomial from another.
21: - `polymul` -- multiply two polynomials.
22: - `polydiv` -- divide one polynomial by another.
23: - `polypow` -- raise a polynomial to an positive integer power
24: - `polyval` -- evaluate a polynomial at given points.
25: - `polyval2d` -- evaluate a 2D polynomial at given points.
26: - `polyval3d` -- evaluate a 3D polynomial at given points.
27: - `polygrid2d` -- evaluate a 2D polynomial on a Cartesian product.
28: - `polygrid3d` -- evaluate a 3D polynomial on a Cartesian product.
29: 
30: Calculus
31: --------
32: - `polyder` -- differentiate a polynomial.
33: - `polyint` -- integrate a polynomial.
34: 
35: Misc Functions
36: --------------
37: - `polyfromroots` -- create a polynomial with specified roots.
38: - `polyroots` -- find the roots of a polynomial.
39: - `polyvander` -- Vandermonde-like matrix for powers.
40: - `polyvander2d` -- Vandermonde-like matrix for 2D power series.
41: - `polyvander3d` -- Vandermonde-like matrix for 3D power series.
42: - `polycompanion` -- companion matrix in power series form.
43: - `polyfit` -- least-squares fit returning a polynomial.
44: - `polytrim` -- trim leading coefficients from a polynomial.
45: - `polyline` -- polynomial representing given straight line.
46: 
47: Classes
48: -------
49: - `Polynomial` -- polynomial class.
50: 
51: See Also
52: --------
53: `numpy.polynomial`
54: 
55: '''
56: from __future__ import division, absolute_import, print_function
57: 
58: __all__ = [
59:     'polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd',
60:     'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval',
61:     'polyder', 'polyint', 'polyfromroots', 'polyvander', 'polyfit',
62:     'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d',
63:     'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d']
64: 
65: import warnings
66: import numpy as np
67: import numpy.linalg as la
68: 
69: from . import polyutils as pu
70: from ._polybase import ABCPolyBase
71: 
72: polytrim = pu.trimcoef
73: 
74: #
75: # These are constant arrays are of integer type so as to be compatible
76: # with the widest range of other types, such as Decimal.
77: #
78: 
79: # Polynomial default domain.
80: polydomain = np.array([-1, 1])
81: 
82: # Polynomial coefficients representing zero.
83: polyzero = np.array([0])
84: 
85: # Polynomial coefficients representing one.
86: polyone = np.array([1])
87: 
88: # Polynomial coefficients representing the identity x.
89: polyx = np.array([0, 1])
90: 
91: #
92: # Polynomial series functions
93: #
94: 
95: 
96: def polyline(off, scl):
97:     '''
98:     Returns an array representing a linear polynomial.
99: 
100:     Parameters
101:     ----------
102:     off, scl : scalars
103:         The "y-intercept" and "slope" of the line, respectively.
104: 
105:     Returns
106:     -------
107:     y : ndarray
108:         This module's representation of the linear polynomial ``off +
109:         scl*x``.
110: 
111:     See Also
112:     --------
113:     chebline
114: 
115:     Examples
116:     --------
117:     >>> from numpy.polynomial import polynomial as P
118:     >>> P.polyline(1,-1)
119:     array([ 1, -1])
120:     >>> P.polyval(1, P.polyline(1,-1)) # should be 0
121:     0.0
122: 
123:     '''
124:     if scl != 0:
125:         return np.array([off, scl])
126:     else:
127:         return np.array([off])
128: 
129: 
130: def polyfromroots(roots):
131:     '''
132:     Generate a monic polynomial with given roots.
133: 
134:     Return the coefficients of the polynomial
135: 
136:     .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
137: 
138:     where the `r_n` are the roots specified in `roots`.  If a zero has
139:     multiplicity n, then it must appear in `roots` n times. For instance,
140:     if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,
141:     then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear
142:     in any order.
143: 
144:     If the returned coefficients are `c`, then
145: 
146:     .. math:: p(x) = c_0 + c_1 * x + ... +  x^n
147: 
148:     The coefficient of the last term is 1 for monic polynomials in this
149:     form.
150: 
151:     Parameters
152:     ----------
153:     roots : array_like
154:         Sequence containing the roots.
155: 
156:     Returns
157:     -------
158:     out : ndarray
159:         1-D array of the polynomial's coefficients If all the roots are
160:         real, then `out` is also real, otherwise it is complex.  (see
161:         Examples below).
162: 
163:     See Also
164:     --------
165:     chebfromroots, legfromroots, lagfromroots, hermfromroots
166:     hermefromroots
167: 
168:     Notes
169:     -----
170:     The coefficients are determined by multiplying together linear factors
171:     of the form `(x - r_i)`, i.e.
172: 
173:     .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)
174: 
175:     where ``n == len(roots) - 1``; note that this implies that `1` is always
176:     returned for :math:`a_n`.
177: 
178:     Examples
179:     --------
180:     >>> from numpy.polynomial import polynomial as P
181:     >>> P.polyfromroots((-1,0,1)) # x(x - 1)(x + 1) = x^3 - x
182:     array([ 0., -1.,  0.,  1.])
183:     >>> j = complex(0,1)
184:     >>> P.polyfromroots((-j,j)) # complex returned, though values are real
185:     array([ 1.+0.j,  0.+0.j,  1.+0.j])
186: 
187:     '''
188:     if len(roots) == 0:
189:         return np.ones(1)
190:     else:
191:         [roots] = pu.as_series([roots], trim=False)
192:         roots.sort()
193:         p = [polyline(-r, 1) for r in roots]
194:         n = len(p)
195:         while n > 1:
196:             m, r = divmod(n, 2)
197:             tmp = [polymul(p[i], p[i+m]) for i in range(m)]
198:             if r:
199:                 tmp[0] = polymul(tmp[0], p[-1])
200:             p = tmp
201:             n = m
202:         return p[0]
203: 
204: 
205: def polyadd(c1, c2):
206:     '''
207:     Add one polynomial to another.
208: 
209:     Returns the sum of two polynomials `c1` + `c2`.  The arguments are
210:     sequences of coefficients from lowest order term to highest, i.e.,
211:     [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.
212: 
213:     Parameters
214:     ----------
215:     c1, c2 : array_like
216:         1-D arrays of polynomial coefficients ordered from low to high.
217: 
218:     Returns
219:     -------
220:     out : ndarray
221:         The coefficient array representing their sum.
222: 
223:     See Also
224:     --------
225:     polysub, polymul, polydiv, polypow
226: 
227:     Examples
228:     --------
229:     >>> from numpy.polynomial import polynomial as P
230:     >>> c1 = (1,2,3)
231:     >>> c2 = (3,2,1)
232:     >>> sum = P.polyadd(c1,c2); sum
233:     array([ 4.,  4.,  4.])
234:     >>> P.polyval(2, sum) # 4 + 4(2) + 4(2**2)
235:     28.0
236: 
237:     '''
238:     # c1, c2 are trimmed copies
239:     [c1, c2] = pu.as_series([c1, c2])
240:     if len(c1) > len(c2):
241:         c1[:c2.size] += c2
242:         ret = c1
243:     else:
244:         c2[:c1.size] += c1
245:         ret = c2
246:     return pu.trimseq(ret)
247: 
248: 
249: def polysub(c1, c2):
250:     '''
251:     Subtract one polynomial from another.
252: 
253:     Returns the difference of two polynomials `c1` - `c2`.  The arguments
254:     are sequences of coefficients from lowest order term to highest, i.e.,
255:     [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.
256: 
257:     Parameters
258:     ----------
259:     c1, c2 : array_like
260:         1-D arrays of polynomial coefficients ordered from low to
261:         high.
262: 
263:     Returns
264:     -------
265:     out : ndarray
266:         Of coefficients representing their difference.
267: 
268:     See Also
269:     --------
270:     polyadd, polymul, polydiv, polypow
271: 
272:     Examples
273:     --------
274:     >>> from numpy.polynomial import polynomial as P
275:     >>> c1 = (1,2,3)
276:     >>> c2 = (3,2,1)
277:     >>> P.polysub(c1,c2)
278:     array([-2.,  0.,  2.])
279:     >>> P.polysub(c2,c1) # -P.polysub(c1,c2)
280:     array([ 2.,  0., -2.])
281: 
282:     '''
283:     # c1, c2 are trimmed copies
284:     [c1, c2] = pu.as_series([c1, c2])
285:     if len(c1) > len(c2):
286:         c1[:c2.size] -= c2
287:         ret = c1
288:     else:
289:         c2 = -c2
290:         c2[:c1.size] += c1
291:         ret = c2
292:     return pu.trimseq(ret)
293: 
294: 
295: def polymulx(c):
296:     '''Multiply a polynomial by x.
297: 
298:     Multiply the polynomial `c` by x, where x is the independent
299:     variable.
300: 
301: 
302:     Parameters
303:     ----------
304:     c : array_like
305:         1-D array of polynomial coefficients ordered from low to
306:         high.
307: 
308:     Returns
309:     -------
310:     out : ndarray
311:         Array representing the result of the multiplication.
312: 
313:     Notes
314:     -----
315: 
316:     .. versionadded:: 1.5.0
317: 
318:     '''
319:     # c is a trimmed copy
320:     [c] = pu.as_series([c])
321:     # The zero series needs special treatment
322:     if len(c) == 1 and c[0] == 0:
323:         return c
324: 
325:     prd = np.empty(len(c) + 1, dtype=c.dtype)
326:     prd[0] = c[0]*0
327:     prd[1:] = c
328:     return prd
329: 
330: 
331: def polymul(c1, c2):
332:     '''
333:     Multiply one polynomial by another.
334: 
335:     Returns the product of two polynomials `c1` * `c2`.  The arguments are
336:     sequences of coefficients, from lowest order term to highest, e.g.,
337:     [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2.``
338: 
339:     Parameters
340:     ----------
341:     c1, c2 : array_like
342:         1-D arrays of coefficients representing a polynomial, relative to the
343:         "standard" basis, and ordered from lowest order term to highest.
344: 
345:     Returns
346:     -------
347:     out : ndarray
348:         Of the coefficients of their product.
349: 
350:     See Also
351:     --------
352:     polyadd, polysub, polydiv, polypow
353: 
354:     Examples
355:     --------
356:     >>> from numpy.polynomial import polynomial as P
357:     >>> c1 = (1,2,3)
358:     >>> c2 = (3,2,1)
359:     >>> P.polymul(c1,c2)
360:     array([  3.,   8.,  14.,   8.,   3.])
361: 
362:     '''
363:     # c1, c2 are trimmed copies
364:     [c1, c2] = pu.as_series([c1, c2])
365:     ret = np.convolve(c1, c2)
366:     return pu.trimseq(ret)
367: 
368: 
369: def polydiv(c1, c2):
370:     '''
371:     Divide one polynomial by another.
372: 
373:     Returns the quotient-with-remainder of two polynomials `c1` / `c2`.
374:     The arguments are sequences of coefficients, from lowest order term
375:     to highest, e.g., [1,2,3] represents ``1 + 2*x + 3*x**2``.
376: 
377:     Parameters
378:     ----------
379:     c1, c2 : array_like
380:         1-D arrays of polynomial coefficients ordered from low to high.
381: 
382:     Returns
383:     -------
384:     [quo, rem] : ndarrays
385:         Of coefficient series representing the quotient and remainder.
386: 
387:     See Also
388:     --------
389:     polyadd, polysub, polymul, polypow
390: 
391:     Examples
392:     --------
393:     >>> from numpy.polynomial import polynomial as P
394:     >>> c1 = (1,2,3)
395:     >>> c2 = (3,2,1)
396:     >>> P.polydiv(c1,c2)
397:     (array([ 3.]), array([-8., -4.]))
398:     >>> P.polydiv(c2,c1)
399:     (array([ 0.33333333]), array([ 2.66666667,  1.33333333]))
400: 
401:     '''
402:     # c1, c2 are trimmed copies
403:     [c1, c2] = pu.as_series([c1, c2])
404:     if c2[-1] == 0:
405:         raise ZeroDivisionError()
406: 
407:     len1 = len(c1)
408:     len2 = len(c2)
409:     if len2 == 1:
410:         return c1/c2[-1], c1[:1]*0
411:     elif len1 < len2:
412:         return c1[:1]*0, c1
413:     else:
414:         dlen = len1 - len2
415:         scl = c2[-1]
416:         c2 = c2[:-1]/scl
417:         i = dlen
418:         j = len1 - 1
419:         while i >= 0:
420:             c1[i:j] -= c2*c1[j]
421:             i -= 1
422:             j -= 1
423:         return c1[j+1:]/scl, pu.trimseq(c1[:j+1])
424: 
425: 
426: def polypow(c, pow, maxpower=None):
427:     '''Raise a polynomial to a power.
428: 
429:     Returns the polynomial `c` raised to the power `pow`. The argument
430:     `c` is a sequence of coefficients ordered from low to high. i.e.,
431:     [1,2,3] is the series  ``1 + 2*x + 3*x**2.``
432: 
433:     Parameters
434:     ----------
435:     c : array_like
436:         1-D array of array of series coefficients ordered from low to
437:         high degree.
438:     pow : integer
439:         Power to which the series will be raised
440:     maxpower : integer, optional
441:         Maximum power allowed. This is mainly to limit growth of the series
442:         to unmanageable size. Default is 16
443: 
444:     Returns
445:     -------
446:     coef : ndarray
447:         Power series of power.
448: 
449:     See Also
450:     --------
451:     polyadd, polysub, polymul, polydiv
452: 
453:     Examples
454:     --------
455: 
456:     '''
457:     # c is a trimmed copy
458:     [c] = pu.as_series([c])
459:     power = int(pow)
460:     if power != pow or power < 0:
461:         raise ValueError("Power must be a non-negative integer.")
462:     elif maxpower is not None and power > maxpower:
463:         raise ValueError("Power is too large")
464:     elif power == 0:
465:         return np.array([1], dtype=c.dtype)
466:     elif power == 1:
467:         return c
468:     else:
469:         # This can be made more efficient by using powers of two
470:         # in the usual way.
471:         prd = c
472:         for i in range(2, power + 1):
473:             prd = np.convolve(prd, c)
474:         return prd
475: 
476: 
477: def polyder(c, m=1, scl=1, axis=0):
478:     '''
479:     Differentiate a polynomial.
480: 
481:     Returns the polynomial coefficients `c` differentiated `m` times along
482:     `axis`.  At each iteration the result is multiplied by `scl` (the
483:     scaling factor is for use in a linear change of variable).  The
484:     argument `c` is an array of coefficients from low to high degree along
485:     each axis, e.g., [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``
486:     while [[1,2],[1,2]] represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is
487:     ``x`` and axis=1 is ``y``.
488: 
489:     Parameters
490:     ----------
491:     c : array_like
492:         Array of polynomial coefficients. If c is multidimensional the
493:         different axis correspond to different variables with the degree
494:         in each axis given by the corresponding index.
495:     m : int, optional
496:         Number of derivatives taken, must be non-negative. (Default: 1)
497:     scl : scalar, optional
498:         Each differentiation is multiplied by `scl`.  The end result is
499:         multiplication by ``scl**m``.  This is for use in a linear change
500:         of variable. (Default: 1)
501:     axis : int, optional
502:         Axis over which the derivative is taken. (Default: 0).
503: 
504:         .. versionadded:: 1.7.0
505: 
506:     Returns
507:     -------
508:     der : ndarray
509:         Polynomial coefficients of the derivative.
510: 
511:     See Also
512:     --------
513:     polyint
514: 
515:     Examples
516:     --------
517:     >>> from numpy.polynomial import polynomial as P
518:     >>> c = (1,2,3,4) # 1 + 2x + 3x**2 + 4x**3
519:     >>> P.polyder(c) # (d/dx)(c) = 2 + 6x + 12x**2
520:     array([  2.,   6.,  12.])
521:     >>> P.polyder(c,3) # (d**3/dx**3)(c) = 24
522:     array([ 24.])
523:     >>> P.polyder(c,scl=-1) # (d/d(-x))(c) = -2 - 6x - 12x**2
524:     array([ -2.,  -6., -12.])
525:     >>> P.polyder(c,2,-1) # (d**2/d(-x)**2)(c) = 6 + 24x
526:     array([  6.,  24.])
527: 
528:     '''
529:     c = np.array(c, ndmin=1, copy=1)
530:     if c.dtype.char in '?bBhHiIlLqQpP':
531:         # astype fails with NA
532:         c = c + 0.0
533:     cdt = c.dtype
534:     cnt, iaxis = [int(t) for t in [m, axis]]
535: 
536:     if cnt != m:
537:         raise ValueError("The order of derivation must be integer")
538:     if cnt < 0:
539:         raise ValueError("The order of derivation must be non-negative")
540:     if iaxis != axis:
541:         raise ValueError("The axis must be integer")
542:     if not -c.ndim <= iaxis < c.ndim:
543:         raise ValueError("The axis is out of range")
544:     if iaxis < 0:
545:         iaxis += c.ndim
546: 
547:     if cnt == 0:
548:         return c
549: 
550:     c = np.rollaxis(c, iaxis)
551:     n = len(c)
552:     if cnt >= n:
553:         c = c[:1]*0
554:     else:
555:         for i in range(cnt):
556:             n = n - 1
557:             c *= scl
558:             der = np.empty((n,) + c.shape[1:], dtype=cdt)
559:             for j in range(n, 0, -1):
560:                 der[j - 1] = j*c[j]
561:             c = der
562:     c = np.rollaxis(c, 0, iaxis + 1)
563:     return c
564: 
565: 
566: def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
567:     '''
568:     Integrate a polynomial.
569: 
570:     Returns the polynomial coefficients `c` integrated `m` times from
571:     `lbnd` along `axis`.  At each iteration the resulting series is
572:     **multiplied** by `scl` and an integration constant, `k`, is added.
573:     The scaling factor is for use in a linear change of variable.  ("Buyer
574:     beware": note that, depending on what one is doing, one may want `scl`
575:     to be the reciprocal of what one might expect; for more information,
576:     see the Notes section below.) The argument `c` is an array of
577:     coefficients, from low to high degree along each axis, e.g., [1,2,3]
578:     represents the polynomial ``1 + 2*x + 3*x**2`` while [[1,2],[1,2]]
579:     represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is ``x`` and axis=1 is
580:     ``y``.
581: 
582:     Parameters
583:     ----------
584:     c : array_like
585:         1-D array of polynomial coefficients, ordered from low to high.
586:     m : int, optional
587:         Order of integration, must be positive. (Default: 1)
588:     k : {[], list, scalar}, optional
589:         Integration constant(s).  The value of the first integral at zero
590:         is the first value in the list, the value of the second integral
591:         at zero is the second value, etc.  If ``k == []`` (the default),
592:         all constants are set to zero.  If ``m == 1``, a single scalar can
593:         be given instead of a list.
594:     lbnd : scalar, optional
595:         The lower bound of the integral. (Default: 0)
596:     scl : scalar, optional
597:         Following each integration the result is *multiplied* by `scl`
598:         before the integration constant is added. (Default: 1)
599:     axis : int, optional
600:         Axis over which the integral is taken. (Default: 0).
601: 
602:         .. versionadded:: 1.7.0
603: 
604:     Returns
605:     -------
606:     S : ndarray
607:         Coefficient array of the integral.
608: 
609:     Raises
610:     ------
611:     ValueError
612:         If ``m < 1``, ``len(k) > m``.
613: 
614:     See Also
615:     --------
616:     polyder
617: 
618:     Notes
619:     -----
620:     Note that the result of each integration is *multiplied* by `scl`.  Why
621:     is this important to note?  Say one is making a linear change of
622:     variable :math:`u = ax + b` in an integral relative to `x`. Then
623:     .. math::`dx = du/a`, so one will need to set `scl` equal to
624:     :math:`1/a` - perhaps not what one would have first thought.
625: 
626:     Examples
627:     --------
628:     >>> from numpy.polynomial import polynomial as P
629:     >>> c = (1,2,3)
630:     >>> P.polyint(c) # should return array([0, 1, 1, 1])
631:     array([ 0.,  1.,  1.,  1.])
632:     >>> P.polyint(c,3) # should return array([0, 0, 0, 1/6, 1/12, 1/20])
633:     array([ 0.        ,  0.        ,  0.        ,  0.16666667,  0.08333333,
634:             0.05      ])
635:     >>> P.polyint(c,k=3) # should return array([3, 1, 1, 1])
636:     array([ 3.,  1.,  1.,  1.])
637:     >>> P.polyint(c,lbnd=-2) # should return array([6, 1, 1, 1])
638:     array([ 6.,  1.,  1.,  1.])
639:     >>> P.polyint(c,scl=-2) # should return array([0, -2, -2, -2])
640:     array([ 0., -2., -2., -2.])
641: 
642:     '''
643:     c = np.array(c, ndmin=1, copy=1)
644:     if c.dtype.char in '?bBhHiIlLqQpP':
645:         # astype doesn't preserve mask attribute.
646:         c = c + 0.0
647:     cdt = c.dtype
648:     if not np.iterable(k):
649:         k = [k]
650:     cnt, iaxis = [int(t) for t in [m, axis]]
651: 
652:     if cnt != m:
653:         raise ValueError("The order of integration must be integer")
654:     if cnt < 0:
655:         raise ValueError("The order of integration must be non-negative")
656:     if len(k) > cnt:
657:         raise ValueError("Too many integration constants")
658:     if iaxis != axis:
659:         raise ValueError("The axis must be integer")
660:     if not -c.ndim <= iaxis < c.ndim:
661:         raise ValueError("The axis is out of range")
662:     if iaxis < 0:
663:         iaxis += c.ndim
664: 
665:     if cnt == 0:
666:         return c
667: 
668:     k = list(k) + [0]*(cnt - len(k))
669:     c = np.rollaxis(c, iaxis)
670:     for i in range(cnt):
671:         n = len(c)
672:         c *= scl
673:         if n == 1 and np.all(c[0] == 0):
674:             c[0] += k[i]
675:         else:
676:             tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
677:             tmp[0] = c[0]*0
678:             tmp[1] = c[0]
679:             for j in range(1, n):
680:                 tmp[j + 1] = c[j]/(j + 1)
681:             tmp[0] += k[i] - polyval(lbnd, tmp)
682:             c = tmp
683:     c = np.rollaxis(c, 0, iaxis + 1)
684:     return c
685: 
686: 
687: def polyval(x, c, tensor=True):
688:     '''
689:     Evaluate a polynomial at points x.
690: 
691:     If `c` is of length `n + 1`, this function returns the value
692: 
693:     .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n
694: 
695:     The parameter `x` is converted to an array only if it is a tuple or a
696:     list, otherwise it is treated as a scalar. In either case, either `x`
697:     or its elements must support multiplication and addition both with
698:     themselves and with the elements of `c`.
699: 
700:     If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
701:     `c` is multidimensional, then the shape of the result depends on the
702:     value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
703:     x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
704:     scalars have shape (,).
705: 
706:     Trailing zeros in the coefficients will be used in the evaluation, so
707:     they should be avoided if efficiency is a concern.
708: 
709:     Parameters
710:     ----------
711:     x : array_like, compatible object
712:         If `x` is a list or tuple, it is converted to an ndarray, otherwise
713:         it is left unchanged and treated as a scalar. In either case, `x`
714:         or its elements must support addition and multiplication with
715:         with themselves and with the elements of `c`.
716:     c : array_like
717:         Array of coefficients ordered so that the coefficients for terms of
718:         degree n are contained in c[n]. If `c` is multidimensional the
719:         remaining indices enumerate multiple polynomials. In the two
720:         dimensional case the coefficients may be thought of as stored in
721:         the columns of `c`.
722:     tensor : boolean, optional
723:         If True, the shape of the coefficient array is extended with ones
724:         on the right, one for each dimension of `x`. Scalars have dimension 0
725:         for this action. The result is that every column of coefficients in
726:         `c` is evaluated for every element of `x`. If False, `x` is broadcast
727:         over the columns of `c` for the evaluation.  This keyword is useful
728:         when `c` is multidimensional. The default value is True.
729: 
730:         .. versionadded:: 1.7.0
731: 
732:     Returns
733:     -------
734:     values : ndarray, compatible object
735:         The shape of the returned array is described above.
736: 
737:     See Also
738:     --------
739:     polyval2d, polygrid2d, polyval3d, polygrid3d
740: 
741:     Notes
742:     -----
743:     The evaluation uses Horner's method.
744: 
745:     Examples
746:     --------
747:     >>> from numpy.polynomial.polynomial import polyval
748:     >>> polyval(1, [1,2,3])
749:     6.0
750:     >>> a = np.arange(4).reshape(2,2)
751:     >>> a
752:     array([[0, 1],
753:            [2, 3]])
754:     >>> polyval(a, [1,2,3])
755:     array([[  1.,   6.],
756:            [ 17.,  34.]])
757:     >>> coef = np.arange(4).reshape(2,2) # multidimensional coefficients
758:     >>> coef
759:     array([[0, 1],
760:            [2, 3]])
761:     >>> polyval([1,2], coef, tensor=True)
762:     array([[ 2.,  4.],
763:            [ 4.,  7.]])
764:     >>> polyval([1,2], coef, tensor=False)
765:     array([ 2.,  7.])
766: 
767:     '''
768:     c = np.array(c, ndmin=1, copy=0)
769:     if c.dtype.char in '?bBhHiIlLqQpP':
770:         # astype fails with NA
771:         c = c + 0.0
772:     if isinstance(x, (tuple, list)):
773:         x = np.asarray(x)
774:     if isinstance(x, np.ndarray) and tensor:
775:         c = c.reshape(c.shape + (1,)*x.ndim)
776: 
777:     c0 = c[-1] + x*0
778:     for i in range(2, len(c) + 1):
779:         c0 = c[-i] + c0*x
780:     return c0
781: 
782: 
783: def polyval2d(x, y, c):
784:     '''
785:     Evaluate a 2-D polynomial at points (x, y).
786: 
787:     This function returns the value
788: 
789:     .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j
790: 
791:     The parameters `x` and `y` are converted to arrays only if they are
792:     tuples or a lists, otherwise they are treated as a scalars and they
793:     must have the same shape after conversion. In either case, either `x`
794:     and `y` or their elements must support multiplication and addition both
795:     with themselves and with the elements of `c`.
796: 
797:     If `c` has fewer than two dimensions, ones are implicitly appended to
798:     its shape to make it 2-D. The shape of the result will be c.shape[2:] +
799:     x.shape.
800: 
801:     Parameters
802:     ----------
803:     x, y : array_like, compatible objects
804:         The two dimensional series is evaluated at the points `(x, y)`,
805:         where `x` and `y` must have the same shape. If `x` or `y` is a list
806:         or tuple, it is first converted to an ndarray, otherwise it is left
807:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
808:     c : array_like
809:         Array of coefficients ordered so that the coefficient of the term
810:         of multi-degree i,j is contained in `c[i,j]`. If `c` has
811:         dimension greater than two the remaining indices enumerate multiple
812:         sets of coefficients.
813: 
814:     Returns
815:     -------
816:     values : ndarray, compatible object
817:         The values of the two dimensional polynomial at points formed with
818:         pairs of corresponding values from `x` and `y`.
819: 
820:     See Also
821:     --------
822:     polyval, polygrid2d, polyval3d, polygrid3d
823: 
824:     Notes
825:     -----
826: 
827:     .. versionadded:: 1.7.0
828: 
829:     '''
830:     try:
831:         x, y = np.array((x, y), copy=0)
832:     except:
833:         raise ValueError('x, y are incompatible')
834: 
835:     c = polyval(x, c)
836:     c = polyval(y, c, tensor=False)
837:     return c
838: 
839: 
840: def polygrid2d(x, y, c):
841:     '''
842:     Evaluate a 2-D polynomial on the Cartesian product of x and y.
843: 
844:     This function returns the values:
845: 
846:     .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * a^i * b^j
847: 
848:     where the points `(a, b)` consist of all pairs formed by taking
849:     `a` from `x` and `b` from `y`. The resulting points form a grid with
850:     `x` in the first dimension and `y` in the second.
851: 
852:     The parameters `x` and `y` are converted to arrays only if they are
853:     tuples or a lists, otherwise they are treated as a scalars. In either
854:     case, either `x` and `y` or their elements must support multiplication
855:     and addition both with themselves and with the elements of `c`.
856: 
857:     If `c` has fewer than two dimensions, ones are implicitly appended to
858:     its shape to make it 2-D. The shape of the result will be c.shape[2:] +
859:     x.shape + y.shape.
860: 
861:     Parameters
862:     ----------
863:     x, y : array_like, compatible objects
864:         The two dimensional series is evaluated at the points in the
865:         Cartesian product of `x` and `y`.  If `x` or `y` is a list or
866:         tuple, it is first converted to an ndarray, otherwise it is left
867:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
868:     c : array_like
869:         Array of coefficients ordered so that the coefficients for terms of
870:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
871:         greater than two the remaining indices enumerate multiple sets of
872:         coefficients.
873: 
874:     Returns
875:     -------
876:     values : ndarray, compatible object
877:         The values of the two dimensional polynomial at points in the Cartesian
878:         product of `x` and `y`.
879: 
880:     See Also
881:     --------
882:     polyval, polyval2d, polyval3d, polygrid3d
883: 
884:     Notes
885:     -----
886: 
887:     .. versionadded:: 1.7.0
888: 
889:     '''
890:     c = polyval(x, c)
891:     c = polyval(y, c)
892:     return c
893: 
894: 
895: def polyval3d(x, y, z, c):
896:     '''
897:     Evaluate a 3-D polynomial at points (x, y, z).
898: 
899:     This function returns the values:
900: 
901:     .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * x^i * y^j * z^k
902: 
903:     The parameters `x`, `y`, and `z` are converted to arrays only if
904:     they are tuples or a lists, otherwise they are treated as a scalars and
905:     they must have the same shape after conversion. In either case, either
906:     `x`, `y`, and `z` or their elements must support multiplication and
907:     addition both with themselves and with the elements of `c`.
908: 
909:     If `c` has fewer than 3 dimensions, ones are implicitly appended to its
910:     shape to make it 3-D. The shape of the result will be c.shape[3:] +
911:     x.shape.
912: 
913:     Parameters
914:     ----------
915:     x, y, z : array_like, compatible object
916:         The three dimensional series is evaluated at the points
917:         `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
918:         any of `x`, `y`, or `z` is a list or tuple, it is first converted
919:         to an ndarray, otherwise it is left unchanged and if it isn't an
920:         ndarray it is  treated as a scalar.
921:     c : array_like
922:         Array of coefficients ordered so that the coefficient of the term of
923:         multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
924:         greater than 3 the remaining indices enumerate multiple sets of
925:         coefficients.
926: 
927:     Returns
928:     -------
929:     values : ndarray, compatible object
930:         The values of the multidimensional polynomial on points formed with
931:         triples of corresponding values from `x`, `y`, and `z`.
932: 
933:     See Also
934:     --------
935:     polyval, polyval2d, polygrid2d, polygrid3d
936: 
937:     Notes
938:     -----
939: 
940:     .. versionadded:: 1.7.0
941: 
942:     '''
943:     try:
944:         x, y, z = np.array((x, y, z), copy=0)
945:     except:
946:         raise ValueError('x, y, z are incompatible')
947: 
948:     c = polyval(x, c)
949:     c = polyval(y, c, tensor=False)
950:     c = polyval(z, c, tensor=False)
951:     return c
952: 
953: 
954: def polygrid3d(x, y, z, c):
955:     '''
956:     Evaluate a 3-D polynomial on the Cartesian product of x, y and z.
957: 
958:     This function returns the values:
959: 
960:     .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * a^i * b^j * c^k
961: 
962:     where the points `(a, b, c)` consist of all triples formed by taking
963:     `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
964:     a grid with `x` in the first dimension, `y` in the second, and `z` in
965:     the third.
966: 
967:     The parameters `x`, `y`, and `z` are converted to arrays only if they
968:     are tuples or a lists, otherwise they are treated as a scalars. In
969:     either case, either `x`, `y`, and `z` or their elements must support
970:     multiplication and addition both with themselves and with the elements
971:     of `c`.
972: 
973:     If `c` has fewer than three dimensions, ones are implicitly appended to
974:     its shape to make it 3-D. The shape of the result will be c.shape[3:] +
975:     x.shape + y.shape + z.shape.
976: 
977:     Parameters
978:     ----------
979:     x, y, z : array_like, compatible objects
980:         The three dimensional series is evaluated at the points in the
981:         Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
982:         list or tuple, it is first converted to an ndarray, otherwise it is
983:         left unchanged and, if it isn't an ndarray, it is treated as a
984:         scalar.
985:     c : array_like
986:         Array of coefficients ordered so that the coefficients for terms of
987:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
988:         greater than two the remaining indices enumerate multiple sets of
989:         coefficients.
990: 
991:     Returns
992:     -------
993:     values : ndarray, compatible object
994:         The values of the two dimensional polynomial at points in the Cartesian
995:         product of `x` and `y`.
996: 
997:     See Also
998:     --------
999:     polyval, polyval2d, polygrid2d, polyval3d
1000: 
1001:     Notes
1002:     -----
1003: 
1004:     .. versionadded:: 1.7.0
1005: 
1006:     '''
1007:     c = polyval(x, c)
1008:     c = polyval(y, c)
1009:     c = polyval(z, c)
1010:     return c
1011: 
1012: 
1013: def polyvander(x, deg):
1014:     '''Vandermonde matrix of given degree.
1015: 
1016:     Returns the Vandermonde matrix of degree `deg` and sample points
1017:     `x`. The Vandermonde matrix is defined by
1018: 
1019:     .. math:: V[..., i] = x^i,
1020: 
1021:     where `0 <= i <= deg`. The leading indices of `V` index the elements of
1022:     `x` and the last index is the power of `x`.
1023: 
1024:     If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
1025:     matrix ``V = polyvander(x, n)``, then ``np.dot(V, c)`` and
1026:     ``polyval(x, c)`` are the same up to roundoff. This equivalence is
1027:     useful both for least squares fitting and for the evaluation of a large
1028:     number of polynomials of the same degree and sample points.
1029: 
1030:     Parameters
1031:     ----------
1032:     x : array_like
1033:         Array of points. The dtype is converted to float64 or complex128
1034:         depending on whether any of the elements are complex. If `x` is
1035:         scalar it is converted to a 1-D array.
1036:     deg : int
1037:         Degree of the resulting matrix.
1038: 
1039:     Returns
1040:     -------
1041:     vander : ndarray.
1042:         The Vandermonde matrix. The shape of the returned matrix is
1043:         ``x.shape + (deg + 1,)``, where the last index is the power of `x`.
1044:         The dtype will be the same as the converted `x`.
1045: 
1046:     See Also
1047:     --------
1048:     polyvander2d, polyvander3d
1049: 
1050:     '''
1051:     ideg = int(deg)
1052:     if ideg != deg:
1053:         raise ValueError("deg must be integer")
1054:     if ideg < 0:
1055:         raise ValueError("deg must be non-negative")
1056: 
1057:     x = np.array(x, copy=0, ndmin=1) + 0.0
1058:     dims = (ideg + 1,) + x.shape
1059:     dtyp = x.dtype
1060:     v = np.empty(dims, dtype=dtyp)
1061:     v[0] = x*0 + 1
1062:     if ideg > 0:
1063:         v[1] = x
1064:         for i in range(2, ideg + 1):
1065:             v[i] = v[i-1]*x
1066:     return np.rollaxis(v, 0, v.ndim)
1067: 
1068: 
1069: def polyvander2d(x, y, deg):
1070:     '''Pseudo-Vandermonde matrix of given degrees.
1071: 
1072:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1073:     points `(x, y)`. The pseudo-Vandermonde matrix is defined by
1074: 
1075:     .. math:: V[..., deg[1]*i + j] = x^i * y^j,
1076: 
1077:     where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
1078:     `V` index the points `(x, y)` and the last index encodes the powers of
1079:     `x` and `y`.
1080: 
1081:     If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
1082:     correspond to the elements of a 2-D coefficient array `c` of shape
1083:     (xdeg + 1, ydeg + 1) in the order
1084: 
1085:     .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
1086: 
1087:     and ``np.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same
1088:     up to roundoff. This equivalence is useful both for least squares
1089:     fitting and for the evaluation of a large number of 2-D polynomials
1090:     of the same degrees and sample points.
1091: 
1092:     Parameters
1093:     ----------
1094:     x, y : array_like
1095:         Arrays of point coordinates, all of the same shape. The dtypes
1096:         will be converted to either float64 or complex128 depending on
1097:         whether any of the elements are complex. Scalars are converted to
1098:         1-D arrays.
1099:     deg : list of ints
1100:         List of maximum degrees of the form [x_deg, y_deg].
1101: 
1102:     Returns
1103:     -------
1104:     vander2d : ndarray
1105:         The shape of the returned matrix is ``x.shape + (order,)``, where
1106:         :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
1107:         as the converted `x` and `y`.
1108: 
1109:     See Also
1110:     --------
1111:     polyvander, polyvander3d. polyval2d, polyval3d
1112: 
1113:     '''
1114:     ideg = [int(d) for d in deg]
1115:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1116:     if is_valid != [1, 1]:
1117:         raise ValueError("degrees must be non-negative integers")
1118:     degx, degy = ideg
1119:     x, y = np.array((x, y), copy=0) + 0.0
1120: 
1121:     vx = polyvander(x, degx)
1122:     vy = polyvander(y, degy)
1123:     v = vx[..., None]*vy[..., None,:]
1124:     # einsum bug
1125:     #v = np.einsum("...i,...j->...ij", vx, vy)
1126:     return v.reshape(v.shape[:-2] + (-1,))
1127: 
1128: 
1129: def polyvander3d(x, y, z, deg):
1130:     '''Pseudo-Vandermonde matrix of given degrees.
1131: 
1132:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1133:     points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
1134:     then The pseudo-Vandermonde matrix is defined by
1135: 
1136:     .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,
1137: 
1138:     where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
1139:     indices of `V` index the points `(x, y, z)` and the last index encodes
1140:     the powers of `x`, `y`, and `z`.
1141: 
1142:     If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
1143:     of `V` correspond to the elements of a 3-D coefficient array `c` of
1144:     shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
1145: 
1146:     .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
1147: 
1148:     and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the
1149:     same up to roundoff. This equivalence is useful both for least squares
1150:     fitting and for the evaluation of a large number of 3-D polynomials
1151:     of the same degrees and sample points.
1152: 
1153:     Parameters
1154:     ----------
1155:     x, y, z : array_like
1156:         Arrays of point coordinates, all of the same shape. The dtypes will
1157:         be converted to either float64 or complex128 depending on whether
1158:         any of the elements are complex. Scalars are converted to 1-D
1159:         arrays.
1160:     deg : list of ints
1161:         List of maximum degrees of the form [x_deg, y_deg, z_deg].
1162: 
1163:     Returns
1164:     -------
1165:     vander3d : ndarray
1166:         The shape of the returned matrix is ``x.shape + (order,)``, where
1167:         :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
1168:         be the same as the converted `x`, `y`, and `z`.
1169: 
1170:     See Also
1171:     --------
1172:     polyvander, polyvander3d. polyval2d, polyval3d
1173: 
1174:     Notes
1175:     -----
1176: 
1177:     .. versionadded:: 1.7.0
1178: 
1179:     '''
1180:     ideg = [int(d) for d in deg]
1181:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1182:     if is_valid != [1, 1, 1]:
1183:         raise ValueError("degrees must be non-negative integers")
1184:     degx, degy, degz = ideg
1185:     x, y, z = np.array((x, y, z), copy=0) + 0.0
1186: 
1187:     vx = polyvander(x, degx)
1188:     vy = polyvander(y, degy)
1189:     vz = polyvander(z, degz)
1190:     v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
1191:     # einsum bug
1192:     #v = np.einsum("...i, ...j, ...k->...ijk", vx, vy, vz)
1193:     return v.reshape(v.shape[:-3] + (-1,))
1194: 
1195: 
1196: def polyfit(x, y, deg, rcond=None, full=False, w=None):
1197:     '''
1198:     Least-squares fit of a polynomial to data.
1199: 
1200:     Return the coefficients of a polynomial of degree `deg` that is the
1201:     least squares fit to the data values `y` given at points `x`. If `y` is
1202:     1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
1203:     fits are done, one for each column of `y`, and the resulting
1204:     coefficients are stored in the corresponding columns of a 2-D return.
1205:     The fitted polynomial(s) are in the form
1206: 
1207:     .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,
1208: 
1209:     where `n` is `deg`.
1210: 
1211:     Parameters
1212:     ----------
1213:     x : array_like, shape (`M`,)
1214:         x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
1215:     y : array_like, shape (`M`,) or (`M`, `K`)
1216:         y-coordinates of the sample points.  Several sets of sample points
1217:         sharing the same x-coordinates can be (independently) fit with one
1218:         call to `polyfit` by passing in for `y` a 2-D array that contains
1219:         one data set per column.
1220:     deg : int or 1-D array_like
1221:         Degree(s) of the fitting polynomials. If `deg` is a single integer
1222:         all terms up to and including the `deg`'th term are included in the
1223:         fit. For Numpy versions >= 1.11 a list of integers specifying the
1224:         degrees of the terms to include may be used instead.
1225:     rcond : float, optional
1226:         Relative condition number of the fit.  Singular values smaller
1227:         than `rcond`, relative to the largest singular value, will be
1228:         ignored.  The default value is ``len(x)*eps``, where `eps` is the
1229:         relative precision of the platform's float type, about 2e-16 in
1230:         most cases.
1231:     full : bool, optional
1232:         Switch determining the nature of the return value.  When ``False``
1233:         (the default) just the coefficients are returned; when ``True``,
1234:         diagnostic information from the singular value decomposition (used
1235:         to solve the fit's matrix equation) is also returned.
1236:     w : array_like, shape (`M`,), optional
1237:         Weights. If not None, the contribution of each point
1238:         ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
1239:         weights are chosen so that the errors of the products ``w[i]*y[i]``
1240:         all have the same variance.  The default value is None.
1241: 
1242:         .. versionadded:: 1.5.0
1243: 
1244:     Returns
1245:     -------
1246:     coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
1247:         Polynomial coefficients ordered from low to high.  If `y` was 2-D,
1248:         the coefficients in column `k` of `coef` represent the polynomial
1249:         fit to the data in `y`'s `k`-th column.
1250: 
1251:     [residuals, rank, singular_values, rcond] : list
1252:         These values are only returned if `full` = True
1253: 
1254:         resid -- sum of squared residuals of the least squares fit
1255:         rank -- the numerical rank of the scaled Vandermonde matrix
1256:         sv -- singular values of the scaled Vandermonde matrix
1257:         rcond -- value of `rcond`.
1258: 
1259:         For more details, see `linalg.lstsq`.
1260: 
1261:     Raises
1262:     ------
1263:     RankWarning
1264:         Raised if the matrix in the least-squares fit is rank deficient.
1265:         The warning is only raised if `full` == False.  The warnings can
1266:         be turned off by:
1267: 
1268:         >>> import warnings
1269:         >>> warnings.simplefilter('ignore', RankWarning)
1270: 
1271:     See Also
1272:     --------
1273:     chebfit, legfit, lagfit, hermfit, hermefit
1274:     polyval : Evaluates a polynomial.
1275:     polyvander : Vandermonde matrix for powers.
1276:     linalg.lstsq : Computes a least-squares fit from the matrix.
1277:     scipy.interpolate.UnivariateSpline : Computes spline fits.
1278: 
1279:     Notes
1280:     -----
1281:     The solution is the coefficients of the polynomial `p` that minimizes
1282:     the sum of the weighted squared errors
1283: 
1284:     .. math :: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
1285: 
1286:     where the :math:`w_j` are the weights. This problem is solved by
1287:     setting up the (typically) over-determined matrix equation:
1288: 
1289:     .. math :: V(x) * c = w * y,
1290: 
1291:     where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
1292:     coefficients to be solved for, `w` are the weights, and `y` are the
1293:     observed values.  This equation is then solved using the singular value
1294:     decomposition of `V`.
1295: 
1296:     If some of the singular values of `V` are so small that they are
1297:     neglected (and `full` == ``False``), a `RankWarning` will be raised.
1298:     This means that the coefficient values may be poorly determined.
1299:     Fitting to a lower order polynomial will usually get rid of the warning
1300:     (but may not be what you want, of course; if you have independent
1301:     reason(s) for choosing the degree which isn't working, you may have to:
1302:     a) reconsider those reasons, and/or b) reconsider the quality of your
1303:     data).  The `rcond` parameter can also be set to a value smaller than
1304:     its default, but the resulting fit may be spurious and have large
1305:     contributions from roundoff error.
1306: 
1307:     Polynomial fits using double precision tend to "fail" at about
1308:     (polynomial) degree 20. Fits using Chebyshev or Legendre series are
1309:     generally better conditioned, but much can still depend on the
1310:     distribution of the sample points and the smoothness of the data.  If
1311:     the quality of the fit is inadequate, splines may be a good
1312:     alternative.
1313: 
1314:     Examples
1315:     --------
1316:     >>> from numpy.polynomial import polynomial as P
1317:     >>> x = np.linspace(-1,1,51) # x "data": [-1, -0.96, ..., 0.96, 1]
1318:     >>> y = x**3 - x + np.random.randn(len(x)) # x^3 - x + N(0,1) "noise"
1319:     >>> c, stats = P.polyfit(x,y,3,full=True)
1320:     >>> c # c[0], c[2] should be approx. 0, c[1] approx. -1, c[3] approx. 1
1321:     array([ 0.01909725, -1.30598256, -0.00577963,  1.02644286])
1322:     >>> stats # note the large SSR, explaining the rather poor results
1323:     [array([ 38.06116253]), 4, array([ 1.38446749,  1.32119158,  0.50443316,
1324:     0.28853036]), 1.1324274851176597e-014]
1325: 
1326:     Same thing without the added noise
1327: 
1328:     >>> y = x**3 - x
1329:     >>> c, stats = P.polyfit(x,y,3,full=True)
1330:     >>> c # c[0], c[2] should be "very close to 0", c[1] ~= -1, c[3] ~= 1
1331:     array([ -1.73362882e-17,  -1.00000000e+00,  -2.67471909e-16,
1332:              1.00000000e+00])
1333:     >>> stats # note the minuscule SSR
1334:     [array([  7.46346754e-31]), 4, array([ 1.38446749,  1.32119158,
1335:     0.50443316,  0.28853036]), 1.1324274851176597e-014]
1336: 
1337:     '''
1338:     x = np.asarray(x) + 0.0
1339:     y = np.asarray(y) + 0.0
1340:     deg = np.asarray(deg)
1341: 
1342:     # check arguments.
1343:     if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
1344:         raise TypeError("deg must be an int or non-empty 1-D array of int")
1345:     if deg.min() < 0:
1346:         raise ValueError("expected deg >= 0")
1347:     if x.ndim != 1:
1348:         raise TypeError("expected 1D vector for x")
1349:     if x.size == 0:
1350:         raise TypeError("expected non-empty vector for x")
1351:     if y.ndim < 1 or y.ndim > 2:
1352:         raise TypeError("expected 1D or 2D array for y")
1353:     if len(x) != len(y):
1354:         raise TypeError("expected x and y to have same length")
1355: 
1356:     if deg.ndim == 0:
1357:         lmax = deg
1358:         order = lmax + 1
1359:         van = polyvander(x, lmax)
1360:     else:
1361:         deg = np.sort(deg)
1362:         lmax = deg[-1]
1363:         order = len(deg)
1364:         van = polyvander(x, lmax)[:, deg]
1365: 
1366:     # set up the least squares matrices in transposed form
1367:     lhs = van.T
1368:     rhs = y.T
1369:     if w is not None:
1370:         w = np.asarray(w) + 0.0
1371:         if w.ndim != 1:
1372:             raise TypeError("expected 1D vector for w")
1373:         if len(x) != len(w):
1374:             raise TypeError("expected x and w to have same length")
1375:         # apply weights. Don't use inplace operations as they
1376:         # can cause problems with NA.
1377:         lhs = lhs * w
1378:         rhs = rhs * w
1379: 
1380:     # set rcond
1381:     if rcond is None:
1382:         rcond = len(x)*np.finfo(x.dtype).eps
1383: 
1384:     # Determine the norms of the design matrix columns.
1385:     if issubclass(lhs.dtype.type, np.complexfloating):
1386:         scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
1387:     else:
1388:         scl = np.sqrt(np.square(lhs).sum(1))
1389:     scl[scl == 0] = 1
1390: 
1391:     # Solve the least squares problem.
1392:     c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
1393:     c = (c.T/scl).T
1394: 
1395:     # Expand c to include non-fitted coefficients which are set to zero
1396:     if deg.ndim == 1:
1397:         if c.ndim == 2:
1398:             cc = np.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
1399:         else:
1400:             cc = np.zeros(lmax + 1, dtype=c.dtype)
1401:         cc[deg] = c
1402:         c = cc
1403: 
1404:     # warn on rank reduction
1405:     if rank != order and not full:
1406:         msg = "The fit may be poorly conditioned"
1407:         warnings.warn(msg, pu.RankWarning)
1408: 
1409:     if full:
1410:         return c, [resids, rank, s, rcond]
1411:     else:
1412:         return c
1413: 
1414: 
1415: def polycompanion(c):
1416:     '''
1417:     Return the companion matrix of c.
1418: 
1419:     The companion matrix for power series cannot be made symmetric by
1420:     scaling the basis, so this function differs from those for the
1421:     orthogonal polynomials.
1422: 
1423:     Parameters
1424:     ----------
1425:     c : array_like
1426:         1-D array of polynomial coefficients ordered from low to high
1427:         degree.
1428: 
1429:     Returns
1430:     -------
1431:     mat : ndarray
1432:         Companion matrix of dimensions (deg, deg).
1433: 
1434:     Notes
1435:     -----
1436: 
1437:     .. versionadded:: 1.7.0
1438: 
1439:     '''
1440:     # c is a trimmed copy
1441:     [c] = pu.as_series([c])
1442:     if len(c) < 2:
1443:         raise ValueError('Series must have maximum degree of at least 1.')
1444:     if len(c) == 2:
1445:         return np.array([[-c[0]/c[1]]])
1446: 
1447:     n = len(c) - 1
1448:     mat = np.zeros((n, n), dtype=c.dtype)
1449:     bot = mat.reshape(-1)[n::n+1]
1450:     bot[...] = 1
1451:     mat[:, -1] -= c[:-1]/c[-1]
1452:     return mat
1453: 
1454: 
1455: def polyroots(c):
1456:     '''
1457:     Compute the roots of a polynomial.
1458: 
1459:     Return the roots (a.k.a. "zeros") of the polynomial
1460: 
1461:     .. math:: p(x) = \\sum_i c[i] * x^i.
1462: 
1463:     Parameters
1464:     ----------
1465:     c : 1-D array_like
1466:         1-D array of polynomial coefficients.
1467: 
1468:     Returns
1469:     -------
1470:     out : ndarray
1471:         Array of the roots of the polynomial. If all the roots are real,
1472:         then `out` is also real, otherwise it is complex.
1473: 
1474:     See Also
1475:     --------
1476:     chebroots
1477: 
1478:     Notes
1479:     -----
1480:     The root estimates are obtained as the eigenvalues of the companion
1481:     matrix, Roots far from the origin of the complex plane may have large
1482:     errors due to the numerical instability of the power series for such
1483:     values. Roots with multiplicity greater than 1 will also show larger
1484:     errors as the value of the series near such points is relatively
1485:     insensitive to errors in the roots. Isolated roots near the origin can
1486:     be improved by a few iterations of Newton's method.
1487: 
1488:     Examples
1489:     --------
1490:     >>> import numpy.polynomial.polynomial as poly
1491:     >>> poly.polyroots(poly.polyfromroots((-1,0,1)))
1492:     array([-1.,  0.,  1.])
1493:     >>> poly.polyroots(poly.polyfromroots((-1,0,1))).dtype
1494:     dtype('float64')
1495:     >>> j = complex(0,1)
1496:     >>> poly.polyroots(poly.polyfromroots((-j,0,j)))
1497:     array([  0.00000000e+00+0.j,   0.00000000e+00+1.j,   2.77555756e-17-1.j])
1498: 
1499:     '''
1500:     # c is a trimmed copy
1501:     [c] = pu.as_series([c])
1502:     if len(c) < 2:
1503:         return np.array([], dtype=c.dtype)
1504:     if len(c) == 2:
1505:         return np.array([-c[0]/c[1]])
1506: 
1507:     m = polycompanion(c)
1508:     r = la.eigvals(m)
1509:     r.sort()
1510:     return r
1511: 
1512: 
1513: #
1514: # polynomial class
1515: #
1516: 
1517: class Polynomial(ABCPolyBase):
1518:     '''A power series class.
1519: 
1520:     The Polynomial class provides the standard Python numerical methods
1521:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
1522:     attributes and methods listed in the `ABCPolyBase` documentation.
1523: 
1524:     Parameters
1525:     ----------
1526:     coef : array_like
1527:         Polynomial coefficients in order of increasing degree, i.e.,
1528:         ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
1529:     domain : (2,) array_like, optional
1530:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
1531:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
1532:         The default value is [-1, 1].
1533:     window : (2,) array_like, optional
1534:         Window, see `domain` for its use. The default value is [-1, 1].
1535: 
1536:         .. versionadded:: 1.6.0
1537: 
1538:     '''
1539:     # Virtual Functions
1540:     _add = staticmethod(polyadd)
1541:     _sub = staticmethod(polysub)
1542:     _mul = staticmethod(polymul)
1543:     _div = staticmethod(polydiv)
1544:     _pow = staticmethod(polypow)
1545:     _val = staticmethod(polyval)
1546:     _int = staticmethod(polyint)
1547:     _der = staticmethod(polyder)
1548:     _fit = staticmethod(polyfit)
1549:     _line = staticmethod(polyline)
1550:     _roots = staticmethod(polyroots)
1551:     _fromroots = staticmethod(polyfromroots)
1552: 
1553:     # Virtual properties
1554:     nickname = 'poly'
1555:     domain = np.array(polydomain)
1556:     window = np.array(polydomain)
1557: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_176531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\nObjects for dealing with polynomials.\n\nThis module provides a number of objects (mostly functions) useful for\ndealing with polynomials, including a `Polynomial` class that\nencapsulates the usual arithmetic operations.  (General information\non how this module represents and works with polynomial objects is in\nthe docstring for its "parent" sub-package, `numpy.polynomial`).\n\nConstants\n---------\n- `polydomain` -- Polynomial default domain, [-1,1].\n- `polyzero` -- (Coefficients of the) "zero polynomial."\n- `polyone` -- (Coefficients of the) constant polynomial 1.\n- `polyx` -- (Coefficients of the) identity map polynomial, ``f(x) = x``.\n\nArithmetic\n----------\n- `polyadd` -- add two polynomials.\n- `polysub` -- subtract one polynomial from another.\n- `polymul` -- multiply two polynomials.\n- `polydiv` -- divide one polynomial by another.\n- `polypow` -- raise a polynomial to an positive integer power\n- `polyval` -- evaluate a polynomial at given points.\n- `polyval2d` -- evaluate a 2D polynomial at given points.\n- `polyval3d` -- evaluate a 3D polynomial at given points.\n- `polygrid2d` -- evaluate a 2D polynomial on a Cartesian product.\n- `polygrid3d` -- evaluate a 3D polynomial on a Cartesian product.\n\nCalculus\n--------\n- `polyder` -- differentiate a polynomial.\n- `polyint` -- integrate a polynomial.\n\nMisc Functions\n--------------\n- `polyfromroots` -- create a polynomial with specified roots.\n- `polyroots` -- find the roots of a polynomial.\n- `polyvander` -- Vandermonde-like matrix for powers.\n- `polyvander2d` -- Vandermonde-like matrix for 2D power series.\n- `polyvander3d` -- Vandermonde-like matrix for 3D power series.\n- `polycompanion` -- companion matrix in power series form.\n- `polyfit` -- least-squares fit returning a polynomial.\n- `polytrim` -- trim leading coefficients from a polynomial.\n- `polyline` -- polynomial representing given straight line.\n\nClasses\n-------\n- `Polynomial` -- polynomial class.\n\nSee Also\n--------\n`numpy.polynomial`\n\n')

# Assigning a List to a Name (line 58):

# Assigning a List to a Name (line 58):
__all__ = ['polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd', 'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval', 'polyder', 'polyint', 'polyfromroots', 'polyvander', 'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d', 'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d']
module_type_store.set_exportable_members(['polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd', 'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval', 'polyder', 'polyint', 'polyfromroots', 'polyvander', 'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d', 'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d'])

# Obtaining an instance of the builtin type 'list' (line 58)
list_176532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 58)
# Adding element type (line 58)
str_176533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'str', 'polyzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176533)
# Adding element type (line 58)
str_176534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'str', 'polyone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176534)
# Adding element type (line 58)
str_176535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'str', 'polyx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176535)
# Adding element type (line 58)
str_176536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'str', 'polydomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176536)
# Adding element type (line 58)
str_176537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 50), 'str', 'polyline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176537)
# Adding element type (line 58)
str_176538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 62), 'str', 'polyadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176538)
# Adding element type (line 58)
str_176539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'polysub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176539)
# Adding element type (line 58)
str_176540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'str', 'polymulx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176540)
# Adding element type (line 58)
str_176541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'str', 'polymul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176541)
# Adding element type (line 58)
str_176542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 38), 'str', 'polydiv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176542)
# Adding element type (line 58)
str_176543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 49), 'str', 'polypow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176543)
# Adding element type (line 58)
str_176544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 60), 'str', 'polyval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176544)
# Adding element type (line 58)
str_176545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'polyder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176545)
# Adding element type (line 58)
str_176546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'str', 'polyint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176546)
# Adding element type (line 58)
str_176547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'str', 'polyfromroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176547)
# Adding element type (line 58)
str_176548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 43), 'str', 'polyvander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176548)
# Adding element type (line 58)
str_176549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 57), 'str', 'polyfit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176549)
# Adding element type (line 58)
str_176550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'str', 'polytrim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176550)
# Adding element type (line 58)
str_176551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'str', 'polyroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176551)
# Adding element type (line 58)
str_176552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'str', 'Polynomial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176552)
# Adding element type (line 58)
str_176553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'str', 'polyval2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176553)
# Adding element type (line 58)
str_176554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 56), 'str', 'polyval3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176554)
# Adding element type (line 58)
str_176555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'str', 'polygrid2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176555)
# Adding element type (line 58)
str_176556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'str', 'polygrid3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176556)
# Adding element type (line 58)
str_176557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'str', 'polyvander2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176557)
# Adding element type (line 58)
str_176558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 48), 'str', 'polyvander3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 10), list_176532, str_176558)

# Assigning a type to the variable '__all__' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), '__all__', list_176532)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 0))

# 'import warnings' statement (line 65)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'import numpy' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_176559 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy')

if (type(import_176559) is not StypyTypeError):

    if (import_176559 != 'pyd_module'):
        __import__(import_176559)
        sys_modules_176560 = sys.modules[import_176559]
        import_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'np', sys_modules_176560.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy', import_176559)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 67, 0))

# 'import numpy.linalg' statement (line 67)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_176561 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.linalg')

if (type(import_176561) is not StypyTypeError):

    if (import_176561 != 'pyd_module'):
        __import__(import_176561)
        sys_modules_176562 = sys.modules[import_176561]
        import_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'la', sys_modules_176562.module_type_store, module_type_store)
    else:
        import numpy.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'la', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.linalg', import_176561)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 69, 0))

# 'from numpy.polynomial import pu' statement (line 69)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_176563 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 69, 0), 'numpy.polynomial')

if (type(import_176563) is not StypyTypeError):

    if (import_176563 != 'pyd_module'):
        __import__(import_176563)
        sys_modules_176564 = sys.modules[import_176563]
        import_from_module(stypy.reporting.localization.Localization(__file__, 69, 0), 'numpy.polynomial', sys_modules_176564.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 69, 0), __file__, sys_modules_176564, sys_modules_176564.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 69, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'numpy.polynomial', import_176563)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 70, 0))

# 'from numpy.polynomial._polybase import ABCPolyBase' statement (line 70)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_176565 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy.polynomial._polybase')

if (type(import_176565) is not StypyTypeError):

    if (import_176565 != 'pyd_module'):
        __import__(import_176565)
        sys_modules_176566 = sys.modules[import_176565]
        import_from_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy.polynomial._polybase', sys_modules_176566.module_type_store, module_type_store, ['ABCPolyBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 70, 0), __file__, sys_modules_176566, sys_modules_176566.module_type_store, module_type_store)
    else:
        from numpy.polynomial._polybase import ABCPolyBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy.polynomial._polybase', None, module_type_store, ['ABCPolyBase'], [ABCPolyBase])

else:
    # Assigning a type to the variable 'numpy.polynomial._polybase' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy.polynomial._polybase', import_176565)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a Attribute to a Name (line 72):

# Assigning a Attribute to a Name (line 72):
# Getting the type of 'pu' (line 72)
pu_176567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'pu')
# Obtaining the member 'trimcoef' of a type (line 72)
trimcoef_176568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), pu_176567, 'trimcoef')
# Assigning a type to the variable 'polytrim' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'polytrim', trimcoef_176568)

# Assigning a Call to a Name (line 80):

# Assigning a Call to a Name (line 80):

# Call to array(...): (line 80)
# Processing the call arguments (line 80)

# Obtaining an instance of the builtin type 'list' (line 80)
list_176571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 80)
# Adding element type (line 80)
int_176572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 22), list_176571, int_176572)
# Adding element type (line 80)
int_176573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 22), list_176571, int_176573)

# Processing the call keyword arguments (line 80)
kwargs_176574 = {}
# Getting the type of 'np' (line 80)
np_176569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'np', False)
# Obtaining the member 'array' of a type (line 80)
array_176570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 13), np_176569, 'array')
# Calling array(args, kwargs) (line 80)
array_call_result_176575 = invoke(stypy.reporting.localization.Localization(__file__, 80, 13), array_176570, *[list_176571], **kwargs_176574)

# Assigning a type to the variable 'polydomain' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'polydomain', array_call_result_176575)

# Assigning a Call to a Name (line 83):

# Assigning a Call to a Name (line 83):

# Call to array(...): (line 83)
# Processing the call arguments (line 83)

# Obtaining an instance of the builtin type 'list' (line 83)
list_176578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 83)
# Adding element type (line 83)
int_176579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_176578, int_176579)

# Processing the call keyword arguments (line 83)
kwargs_176580 = {}
# Getting the type of 'np' (line 83)
np_176576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'np', False)
# Obtaining the member 'array' of a type (line 83)
array_176577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), np_176576, 'array')
# Calling array(args, kwargs) (line 83)
array_call_result_176581 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), array_176577, *[list_176578], **kwargs_176580)

# Assigning a type to the variable 'polyzero' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'polyzero', array_call_result_176581)

# Assigning a Call to a Name (line 86):

# Assigning a Call to a Name (line 86):

# Call to array(...): (line 86)
# Processing the call arguments (line 86)

# Obtaining an instance of the builtin type 'list' (line 86)
list_176584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 86)
# Adding element type (line 86)
int_176585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 19), list_176584, int_176585)

# Processing the call keyword arguments (line 86)
kwargs_176586 = {}
# Getting the type of 'np' (line 86)
np_176582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 10), 'np', False)
# Obtaining the member 'array' of a type (line 86)
array_176583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 10), np_176582, 'array')
# Calling array(args, kwargs) (line 86)
array_call_result_176587 = invoke(stypy.reporting.localization.Localization(__file__, 86, 10), array_176583, *[list_176584], **kwargs_176586)

# Assigning a type to the variable 'polyone' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'polyone', array_call_result_176587)

# Assigning a Call to a Name (line 89):

# Assigning a Call to a Name (line 89):

# Call to array(...): (line 89)
# Processing the call arguments (line 89)

# Obtaining an instance of the builtin type 'list' (line 89)
list_176590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 89)
# Adding element type (line 89)
int_176591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 17), list_176590, int_176591)
# Adding element type (line 89)
int_176592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 17), list_176590, int_176592)

# Processing the call keyword arguments (line 89)
kwargs_176593 = {}
# Getting the type of 'np' (line 89)
np_176588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'np', False)
# Obtaining the member 'array' of a type (line 89)
array_176589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), np_176588, 'array')
# Calling array(args, kwargs) (line 89)
array_call_result_176594 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), array_176589, *[list_176590], **kwargs_176593)

# Assigning a type to the variable 'polyx' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'polyx', array_call_result_176594)

@norecursion
def polyline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyline'
    module_type_store = module_type_store.open_function_context('polyline', 96, 0, False)
    
    # Passed parameters checking function
    polyline.stypy_localization = localization
    polyline.stypy_type_of_self = None
    polyline.stypy_type_store = module_type_store
    polyline.stypy_function_name = 'polyline'
    polyline.stypy_param_names_list = ['off', 'scl']
    polyline.stypy_varargs_param_name = None
    polyline.stypy_kwargs_param_name = None
    polyline.stypy_call_defaults = defaults
    polyline.stypy_call_varargs = varargs
    polyline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyline', ['off', 'scl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyline', localization, ['off', 'scl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyline(...)' code ##################

    str_176595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'str', '\n    Returns an array representing a linear polynomial.\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The "y-intercept" and "slope" of the line, respectively.\n\n    Returns\n    -------\n    y : ndarray\n        This module\'s representation of the linear polynomial ``off +\n        scl*x``.\n\n    See Also\n    --------\n    chebline\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> P.polyline(1,-1)\n    array([ 1, -1])\n    >>> P.polyval(1, P.polyline(1,-1)) # should be 0\n    0.0\n\n    ')
    
    
    # Getting the type of 'scl' (line 124)
    scl_176596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'scl')
    int_176597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'int')
    # Applying the binary operator '!=' (line 124)
    result_ne_176598 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), '!=', scl_176596, int_176597)
    
    # Testing the type of an if condition (line 124)
    if_condition_176599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_ne_176598)
    # Assigning a type to the variable 'if_condition_176599' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_176599', if_condition_176599)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_176602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    # Adding element type (line 125)
    # Getting the type of 'off' (line 125)
    off_176603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 24), list_176602, off_176603)
    # Adding element type (line 125)
    # Getting the type of 'scl' (line 125)
    scl_176604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 30), 'scl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 24), list_176602, scl_176604)
    
    # Processing the call keyword arguments (line 125)
    kwargs_176605 = {}
    # Getting the type of 'np' (line 125)
    np_176600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 125)
    array_176601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), np_176600, 'array')
    # Calling array(args, kwargs) (line 125)
    array_call_result_176606 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), array_176601, *[list_176602], **kwargs_176605)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', array_call_result_176606)
    # SSA branch for the else part of an if statement (line 124)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_176609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    # Adding element type (line 127)
    # Getting the type of 'off' (line 127)
    off_176610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 24), list_176609, off_176610)
    
    # Processing the call keyword arguments (line 127)
    kwargs_176611 = {}
    # Getting the type of 'np' (line 127)
    np_176607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 127)
    array_176608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), np_176607, 'array')
    # Calling array(args, kwargs) (line 127)
    array_call_result_176612 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), array_176608, *[list_176609], **kwargs_176611)
    
    # Assigning a type to the variable 'stypy_return_type' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', array_call_result_176612)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polyline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyline' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_176613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176613)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyline'
    return stypy_return_type_176613

# Assigning a type to the variable 'polyline' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'polyline', polyline)

@norecursion
def polyfromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyfromroots'
    module_type_store = module_type_store.open_function_context('polyfromroots', 130, 0, False)
    
    # Passed parameters checking function
    polyfromroots.stypy_localization = localization
    polyfromroots.stypy_type_of_self = None
    polyfromroots.stypy_type_store = module_type_store
    polyfromroots.stypy_function_name = 'polyfromroots'
    polyfromroots.stypy_param_names_list = ['roots']
    polyfromroots.stypy_varargs_param_name = None
    polyfromroots.stypy_kwargs_param_name = None
    polyfromroots.stypy_call_defaults = defaults
    polyfromroots.stypy_call_varargs = varargs
    polyfromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyfromroots', ['roots'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyfromroots', localization, ['roots'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyfromroots(...)' code ##################

    str_176614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', "\n    Generate a monic polynomial with given roots.\n\n    Return the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    where the `r_n` are the roots specified in `roots`.  If a zero has\n    multiplicity n, then it must appear in `roots` n times. For instance,\n    if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,\n    then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear\n    in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * x + ... +  x^n\n\n    The coefficient of the last term is 1 for monic polynomials in this\n    form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of the polynomial's coefficients If all the roots are\n        real, then `out` is also real, otherwise it is complex.  (see\n        Examples below).\n\n    See Also\n    --------\n    chebfromroots, legfromroots, lagfromroots, hermfromroots\n    hermefromroots\n\n    Notes\n    -----\n    The coefficients are determined by multiplying together linear factors\n    of the form `(x - r_i)`, i.e.\n\n    .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)\n\n    where ``n == len(roots) - 1``; note that this implies that `1` is always\n    returned for :math:`a_n`.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> P.polyfromroots((-1,0,1)) # x(x - 1)(x + 1) = x^3 - x\n    array([ 0., -1.,  0.,  1.])\n    >>> j = complex(0,1)\n    >>> P.polyfromroots((-j,j)) # complex returned, though values are real\n    array([ 1.+0.j,  0.+0.j,  1.+0.j])\n\n    ")
    
    
    
    # Call to len(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'roots' (line 188)
    roots_176616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'roots', False)
    # Processing the call keyword arguments (line 188)
    kwargs_176617 = {}
    # Getting the type of 'len' (line 188)
    len_176615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'len', False)
    # Calling len(args, kwargs) (line 188)
    len_call_result_176618 = invoke(stypy.reporting.localization.Localization(__file__, 188, 7), len_176615, *[roots_176616], **kwargs_176617)
    
    int_176619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 21), 'int')
    # Applying the binary operator '==' (line 188)
    result_eq_176620 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 7), '==', len_call_result_176618, int_176619)
    
    # Testing the type of an if condition (line 188)
    if_condition_176621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 4), result_eq_176620)
    # Assigning a type to the variable 'if_condition_176621' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'if_condition_176621', if_condition_176621)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 189)
    # Processing the call arguments (line 189)
    int_176624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 23), 'int')
    # Processing the call keyword arguments (line 189)
    kwargs_176625 = {}
    # Getting the type of 'np' (line 189)
    np_176622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 189)
    ones_176623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), np_176622, 'ones')
    # Calling ones(args, kwargs) (line 189)
    ones_call_result_176626 = invoke(stypy.reporting.localization.Localization(__file__, 189, 15), ones_176623, *[int_176624], **kwargs_176625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', ones_call_result_176626)
    # SSA branch for the else part of an if statement (line 188)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a List (line 191):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Obtaining an instance of the builtin type 'list' (line 191)
    list_176629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 191)
    # Adding element type (line 191)
    # Getting the type of 'roots' (line 191)
    roots_176630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 32), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 31), list_176629, roots_176630)
    
    # Processing the call keyword arguments (line 191)
    # Getting the type of 'False' (line 191)
    False_176631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 45), 'False', False)
    keyword_176632 = False_176631
    kwargs_176633 = {'trim': keyword_176632}
    # Getting the type of 'pu' (line 191)
    pu_176627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 18), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 191)
    as_series_176628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 18), pu_176627, 'as_series')
    # Calling as_series(args, kwargs) (line 191)
    as_series_call_result_176634 = invoke(stypy.reporting.localization.Localization(__file__, 191, 18), as_series_176628, *[list_176629], **kwargs_176633)
    
    # Assigning a type to the variable 'call_assignment_176480' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'call_assignment_176480', as_series_call_result_176634)
    
    # Assigning a Call to a Name (line 191):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
    # Processing the call keyword arguments
    kwargs_176638 = {}
    # Getting the type of 'call_assignment_176480' (line 191)
    call_assignment_176480_176635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'call_assignment_176480', False)
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___176636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), call_assignment_176480_176635, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176639 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176636, *[int_176637], **kwargs_176638)
    
    # Assigning a type to the variable 'call_assignment_176481' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'call_assignment_176481', getitem___call_result_176639)
    
    # Assigning a Name to a Name (line 191):
    # Getting the type of 'call_assignment_176481' (line 191)
    call_assignment_176481_176640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'call_assignment_176481')
    # Assigning a type to the variable 'roots' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'roots', call_assignment_176481_176640)
    
    # Call to sort(...): (line 192)
    # Processing the call keyword arguments (line 192)
    kwargs_176643 = {}
    # Getting the type of 'roots' (line 192)
    roots_176641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'roots', False)
    # Obtaining the member 'sort' of a type (line 192)
    sort_176642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), roots_176641, 'sort')
    # Calling sort(args, kwargs) (line 192)
    sort_call_result_176644 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), sort_176642, *[], **kwargs_176643)
    
    
    # Assigning a ListComp to a Name (line 193):
    
    # Assigning a ListComp to a Name (line 193):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'roots' (line 193)
    roots_176651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 38), 'roots')
    comprehension_176652 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 13), roots_176651)
    # Assigning a type to the variable 'r' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'r', comprehension_176652)
    
    # Call to polyline(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Getting the type of 'r' (line 193)
    r_176646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'r', False)
    # Applying the 'usub' unary operator (line 193)
    result___neg___176647 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 22), 'usub', r_176646)
    
    int_176648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'int')
    # Processing the call keyword arguments (line 193)
    kwargs_176649 = {}
    # Getting the type of 'polyline' (line 193)
    polyline_176645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'polyline', False)
    # Calling polyline(args, kwargs) (line 193)
    polyline_call_result_176650 = invoke(stypy.reporting.localization.Localization(__file__, 193, 13), polyline_176645, *[result___neg___176647, int_176648], **kwargs_176649)
    
    list_176653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 13), list_176653, polyline_call_result_176650)
    # Assigning a type to the variable 'p' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'p', list_176653)
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to len(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'p' (line 194)
    p_176655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'p', False)
    # Processing the call keyword arguments (line 194)
    kwargs_176656 = {}
    # Getting the type of 'len' (line 194)
    len_176654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'len', False)
    # Calling len(args, kwargs) (line 194)
    len_call_result_176657 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), len_176654, *[p_176655], **kwargs_176656)
    
    # Assigning a type to the variable 'n' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'n', len_call_result_176657)
    
    
    # Getting the type of 'n' (line 195)
    n_176658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'n')
    int_176659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'int')
    # Applying the binary operator '>' (line 195)
    result_gt_176660 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 14), '>', n_176658, int_176659)
    
    # Testing the type of an if condition (line 195)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), result_gt_176660)
    # SSA begins for while statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 196):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'n' (line 196)
    n_176662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'n', False)
    int_176663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 29), 'int')
    # Processing the call keyword arguments (line 196)
    kwargs_176664 = {}
    # Getting the type of 'divmod' (line 196)
    divmod_176661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 196)
    divmod_call_result_176665 = invoke(stypy.reporting.localization.Localization(__file__, 196, 19), divmod_176661, *[n_176662, int_176663], **kwargs_176664)
    
    # Assigning a type to the variable 'call_assignment_176482' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176482', divmod_call_result_176665)
    
    # Assigning a Call to a Name (line 196):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 12), 'int')
    # Processing the call keyword arguments
    kwargs_176669 = {}
    # Getting the type of 'call_assignment_176482' (line 196)
    call_assignment_176482_176666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176482', False)
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___176667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), call_assignment_176482_176666, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176670 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176667, *[int_176668], **kwargs_176669)
    
    # Assigning a type to the variable 'call_assignment_176483' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176483', getitem___call_result_176670)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'call_assignment_176483' (line 196)
    call_assignment_176483_176671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176483')
    # Assigning a type to the variable 'm' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'm', call_assignment_176483_176671)
    
    # Assigning a Call to a Name (line 196):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 12), 'int')
    # Processing the call keyword arguments
    kwargs_176675 = {}
    # Getting the type of 'call_assignment_176482' (line 196)
    call_assignment_176482_176672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176482', False)
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___176673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), call_assignment_176482_176672, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176676 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176673, *[int_176674], **kwargs_176675)
    
    # Assigning a type to the variable 'call_assignment_176484' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176484', getitem___call_result_176676)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'call_assignment_176484' (line 196)
    call_assignment_176484_176677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'call_assignment_176484')
    # Assigning a type to the variable 'r' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'r', call_assignment_176484_176677)
    
    # Assigning a ListComp to a Name (line 197):
    
    # Assigning a ListComp to a Name (line 197):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'm' (line 197)
    m_176692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 56), 'm', False)
    # Processing the call keyword arguments (line 197)
    kwargs_176693 = {}
    # Getting the type of 'range' (line 197)
    range_176691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 50), 'range', False)
    # Calling range(args, kwargs) (line 197)
    range_call_result_176694 = invoke(stypy.reporting.localization.Localization(__file__, 197, 50), range_176691, *[m_176692], **kwargs_176693)
    
    comprehension_176695 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 19), range_call_result_176694)
    # Assigning a type to the variable 'i' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'i', comprehension_176695)
    
    # Call to polymul(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 197)
    i_176679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 29), 'i', False)
    # Getting the type of 'p' (line 197)
    p_176680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___176681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 27), p_176680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_176682 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), getitem___176681, i_176679)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 197)
    i_176683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 35), 'i', False)
    # Getting the type of 'm' (line 197)
    m_176684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 37), 'm', False)
    # Applying the binary operator '+' (line 197)
    result_add_176685 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 35), '+', i_176683, m_176684)
    
    # Getting the type of 'p' (line 197)
    p_176686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___176687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 33), p_176686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_176688 = invoke(stypy.reporting.localization.Localization(__file__, 197, 33), getitem___176687, result_add_176685)
    
    # Processing the call keyword arguments (line 197)
    kwargs_176689 = {}
    # Getting the type of 'polymul' (line 197)
    polymul_176678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'polymul', False)
    # Calling polymul(args, kwargs) (line 197)
    polymul_call_result_176690 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), polymul_176678, *[subscript_call_result_176682, subscript_call_result_176688], **kwargs_176689)
    
    list_176696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 19), list_176696, polymul_call_result_176690)
    # Assigning a type to the variable 'tmp' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'tmp', list_176696)
    
    # Getting the type of 'r' (line 198)
    r_176697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'r')
    # Testing the type of an if condition (line 198)
    if_condition_176698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), r_176697)
    # Assigning a type to the variable 'if_condition_176698' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_176698', if_condition_176698)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 199):
    
    # Assigning a Call to a Subscript (line 199):
    
    # Call to polymul(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Obtaining the type of the subscript
    int_176700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 37), 'int')
    # Getting the type of 'tmp' (line 199)
    tmp_176701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___176702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 33), tmp_176701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_176703 = invoke(stypy.reporting.localization.Localization(__file__, 199, 33), getitem___176702, int_176700)
    
    
    # Obtaining the type of the subscript
    int_176704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 43), 'int')
    # Getting the type of 'p' (line 199)
    p_176705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___176706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), p_176705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_176707 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), getitem___176706, int_176704)
    
    # Processing the call keyword arguments (line 199)
    kwargs_176708 = {}
    # Getting the type of 'polymul' (line 199)
    polymul_176699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'polymul', False)
    # Calling polymul(args, kwargs) (line 199)
    polymul_call_result_176709 = invoke(stypy.reporting.localization.Localization(__file__, 199, 25), polymul_176699, *[subscript_call_result_176703, subscript_call_result_176707], **kwargs_176708)
    
    # Getting the type of 'tmp' (line 199)
    tmp_176710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tmp')
    int_176711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'int')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 16), tmp_176710, (int_176711, polymul_call_result_176709))
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 200):
    
    # Assigning a Name to a Name (line 200):
    # Getting the type of 'tmp' (line 200)
    tmp_176712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'tmp')
    # Assigning a type to the variable 'p' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'p', tmp_176712)
    
    # Assigning a Name to a Name (line 201):
    
    # Assigning a Name to a Name (line 201):
    # Getting the type of 'm' (line 201)
    m_176713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'm')
    # Assigning a type to the variable 'n' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'n', m_176713)
    # SSA join for while statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_176714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 17), 'int')
    # Getting the type of 'p' (line 202)
    p_176715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'p')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___176716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 15), p_176715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_176717 = invoke(stypy.reporting.localization.Localization(__file__, 202, 15), getitem___176716, int_176714)
    
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', subscript_call_result_176717)
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polyfromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyfromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_176718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyfromroots'
    return stypy_return_type_176718

# Assigning a type to the variable 'polyfromroots' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'polyfromroots', polyfromroots)

@norecursion
def polyadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyadd'
    module_type_store = module_type_store.open_function_context('polyadd', 205, 0, False)
    
    # Passed parameters checking function
    polyadd.stypy_localization = localization
    polyadd.stypy_type_of_self = None
    polyadd.stypy_type_store = module_type_store
    polyadd.stypy_function_name = 'polyadd'
    polyadd.stypy_param_names_list = ['c1', 'c2']
    polyadd.stypy_varargs_param_name = None
    polyadd.stypy_kwargs_param_name = None
    polyadd.stypy_call_defaults = defaults
    polyadd.stypy_call_varargs = varargs
    polyadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyadd', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyadd', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyadd(...)' code ##################

    str_176719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, (-1)), 'str', '\n    Add one polynomial to another.\n\n    Returns the sum of two polynomials `c1` + `c2`.  The arguments are\n    sequences of coefficients from lowest order term to highest, i.e.,\n    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of polynomial coefficients ordered from low to high.\n\n    Returns\n    -------\n    out : ndarray\n        The coefficient array representing their sum.\n\n    See Also\n    --------\n    polysub, polymul, polydiv, polypow\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> sum = P.polyadd(c1,c2); sum\n    array([ 4.,  4.,  4.])\n    >>> P.polyval(2, sum) # 4 + 4(2) + 4(2**2)\n    28.0\n\n    ')
    
    # Assigning a Call to a List (line 239):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 239)
    # Processing the call arguments (line 239)
    
    # Obtaining an instance of the builtin type 'list' (line 239)
    list_176722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 239)
    # Adding element type (line 239)
    # Getting the type of 'c1' (line 239)
    c1_176723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 28), list_176722, c1_176723)
    # Adding element type (line 239)
    # Getting the type of 'c2' (line 239)
    c2_176724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 28), list_176722, c2_176724)
    
    # Processing the call keyword arguments (line 239)
    kwargs_176725 = {}
    # Getting the type of 'pu' (line 239)
    pu_176720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 239)
    as_series_176721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), pu_176720, 'as_series')
    # Calling as_series(args, kwargs) (line 239)
    as_series_call_result_176726 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), as_series_176721, *[list_176722], **kwargs_176725)
    
    # Assigning a type to the variable 'call_assignment_176485' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176485', as_series_call_result_176726)
    
    # Assigning a Call to a Name (line 239):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176730 = {}
    # Getting the type of 'call_assignment_176485' (line 239)
    call_assignment_176485_176727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176485', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___176728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 4), call_assignment_176485_176727, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176731 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176728, *[int_176729], **kwargs_176730)
    
    # Assigning a type to the variable 'call_assignment_176486' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176486', getitem___call_result_176731)
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'call_assignment_176486' (line 239)
    call_assignment_176486_176732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176486')
    # Assigning a type to the variable 'c1' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 5), 'c1', call_assignment_176486_176732)
    
    # Assigning a Call to a Name (line 239):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176736 = {}
    # Getting the type of 'call_assignment_176485' (line 239)
    call_assignment_176485_176733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176485', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___176734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 4), call_assignment_176485_176733, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176737 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176734, *[int_176735], **kwargs_176736)
    
    # Assigning a type to the variable 'call_assignment_176487' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176487', getitem___call_result_176737)
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'call_assignment_176487' (line 239)
    call_assignment_176487_176738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'call_assignment_176487')
    # Assigning a type to the variable 'c2' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 9), 'c2', call_assignment_176487_176738)
    
    
    
    # Call to len(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'c1' (line 240)
    c1_176740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'c1', False)
    # Processing the call keyword arguments (line 240)
    kwargs_176741 = {}
    # Getting the type of 'len' (line 240)
    len_176739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'len', False)
    # Calling len(args, kwargs) (line 240)
    len_call_result_176742 = invoke(stypy.reporting.localization.Localization(__file__, 240, 7), len_176739, *[c1_176740], **kwargs_176741)
    
    
    # Call to len(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'c2' (line 240)
    c2_176744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'c2', False)
    # Processing the call keyword arguments (line 240)
    kwargs_176745 = {}
    # Getting the type of 'len' (line 240)
    len_176743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'len', False)
    # Calling len(args, kwargs) (line 240)
    len_call_result_176746 = invoke(stypy.reporting.localization.Localization(__file__, 240, 17), len_176743, *[c2_176744], **kwargs_176745)
    
    # Applying the binary operator '>' (line 240)
    result_gt_176747 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 7), '>', len_call_result_176742, len_call_result_176746)
    
    # Testing the type of an if condition (line 240)
    if_condition_176748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 4), result_gt_176747)
    # Assigning a type to the variable 'if_condition_176748' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'if_condition_176748', if_condition_176748)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 241)
    c1_176749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 241)
    c2_176750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'c2')
    # Obtaining the member 'size' of a type (line 241)
    size_176751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), c2_176750, 'size')
    slice_176752 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 8), None, size_176751, None)
    # Getting the type of 'c1' (line 241)
    c1_176753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___176754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), c1_176753, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_176755 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), getitem___176754, slice_176752)
    
    # Getting the type of 'c2' (line 241)
    c2_176756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'c2')
    # Applying the binary operator '+=' (line 241)
    result_iadd_176757 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 8), '+=', subscript_call_result_176755, c2_176756)
    # Getting the type of 'c1' (line 241)
    c1_176758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'c1')
    # Getting the type of 'c2' (line 241)
    c2_176759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'c2')
    # Obtaining the member 'size' of a type (line 241)
    size_176760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), c2_176759, 'size')
    slice_176761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 8), None, size_176760, None)
    # Storing an element on a container (line 241)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 8), c1_176758, (slice_176761, result_iadd_176757))
    
    
    # Assigning a Name to a Name (line 242):
    
    # Assigning a Name to a Name (line 242):
    # Getting the type of 'c1' (line 242)
    c1_176762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'ret', c1_176762)
    # SSA branch for the else part of an if statement (line 240)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'c2' (line 244)
    c2_176763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 244)
    c1_176764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'c1')
    # Obtaining the member 'size' of a type (line 244)
    size_176765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), c1_176764, 'size')
    slice_176766 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 244, 8), None, size_176765, None)
    # Getting the type of 'c2' (line 244)
    c2_176767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___176768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), c2_176767, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_176769 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), getitem___176768, slice_176766)
    
    # Getting the type of 'c1' (line 244)
    c1_176770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'c1')
    # Applying the binary operator '+=' (line 244)
    result_iadd_176771 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 8), '+=', subscript_call_result_176769, c1_176770)
    # Getting the type of 'c2' (line 244)
    c2_176772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'c2')
    # Getting the type of 'c1' (line 244)
    c1_176773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'c1')
    # Obtaining the member 'size' of a type (line 244)
    size_176774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), c1_176773, 'size')
    slice_176775 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 244, 8), None, size_176774, None)
    # Storing an element on a container (line 244)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 8), c2_176772, (slice_176775, result_iadd_176771))
    
    
    # Assigning a Name to a Name (line 245):
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'c2' (line 245)
    c2_176776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'ret', c2_176776)
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'ret' (line 246)
    ret_176779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'ret', False)
    # Processing the call keyword arguments (line 246)
    kwargs_176780 = {}
    # Getting the type of 'pu' (line 246)
    pu_176777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 246)
    trimseq_176778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 11), pu_176777, 'trimseq')
    # Calling trimseq(args, kwargs) (line 246)
    trimseq_call_result_176781 = invoke(stypy.reporting.localization.Localization(__file__, 246, 11), trimseq_176778, *[ret_176779], **kwargs_176780)
    
    # Assigning a type to the variable 'stypy_return_type' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type', trimseq_call_result_176781)
    
    # ################# End of 'polyadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyadd' in the type store
    # Getting the type of 'stypy_return_type' (line 205)
    stypy_return_type_176782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176782)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyadd'
    return stypy_return_type_176782

# Assigning a type to the variable 'polyadd' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'polyadd', polyadd)

@norecursion
def polysub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polysub'
    module_type_store = module_type_store.open_function_context('polysub', 249, 0, False)
    
    # Passed parameters checking function
    polysub.stypy_localization = localization
    polysub.stypy_type_of_self = None
    polysub.stypy_type_store = module_type_store
    polysub.stypy_function_name = 'polysub'
    polysub.stypy_param_names_list = ['c1', 'c2']
    polysub.stypy_varargs_param_name = None
    polysub.stypy_kwargs_param_name = None
    polysub.stypy_call_defaults = defaults
    polysub.stypy_call_varargs = varargs
    polysub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polysub', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polysub', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polysub(...)' code ##################

    str_176783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, (-1)), 'str', '\n    Subtract one polynomial from another.\n\n    Returns the difference of two polynomials `c1` - `c2`.  The arguments\n    are sequences of coefficients from lowest order term to highest, i.e.,\n    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of polynomial coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of coefficients representing their difference.\n\n    See Also\n    --------\n    polyadd, polymul, polydiv, polypow\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> P.polysub(c1,c2)\n    array([-2.,  0.,  2.])\n    >>> P.polysub(c2,c1) # -P.polysub(c1,c2)\n    array([ 2.,  0., -2.])\n\n    ')
    
    # Assigning a Call to a List (line 284):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Obtaining an instance of the builtin type 'list' (line 284)
    list_176786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 284)
    # Adding element type (line 284)
    # Getting the type of 'c1' (line 284)
    c1_176787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 28), list_176786, c1_176787)
    # Adding element type (line 284)
    # Getting the type of 'c2' (line 284)
    c2_176788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 28), list_176786, c2_176788)
    
    # Processing the call keyword arguments (line 284)
    kwargs_176789 = {}
    # Getting the type of 'pu' (line 284)
    pu_176784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 284)
    as_series_176785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 15), pu_176784, 'as_series')
    # Calling as_series(args, kwargs) (line 284)
    as_series_call_result_176790 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), as_series_176785, *[list_176786], **kwargs_176789)
    
    # Assigning a type to the variable 'call_assignment_176488' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176488', as_series_call_result_176790)
    
    # Assigning a Call to a Name (line 284):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176794 = {}
    # Getting the type of 'call_assignment_176488' (line 284)
    call_assignment_176488_176791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176488', False)
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___176792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 4), call_assignment_176488_176791, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176795 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176792, *[int_176793], **kwargs_176794)
    
    # Assigning a type to the variable 'call_assignment_176489' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176489', getitem___call_result_176795)
    
    # Assigning a Name to a Name (line 284):
    # Getting the type of 'call_assignment_176489' (line 284)
    call_assignment_176489_176796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176489')
    # Assigning a type to the variable 'c1' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 5), 'c1', call_assignment_176489_176796)
    
    # Assigning a Call to a Name (line 284):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176800 = {}
    # Getting the type of 'call_assignment_176488' (line 284)
    call_assignment_176488_176797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176488', False)
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___176798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 4), call_assignment_176488_176797, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176801 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176798, *[int_176799], **kwargs_176800)
    
    # Assigning a type to the variable 'call_assignment_176490' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176490', getitem___call_result_176801)
    
    # Assigning a Name to a Name (line 284):
    # Getting the type of 'call_assignment_176490' (line 284)
    call_assignment_176490_176802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'call_assignment_176490')
    # Assigning a type to the variable 'c2' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 9), 'c2', call_assignment_176490_176802)
    
    
    
    # Call to len(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'c1' (line 285)
    c1_176804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'c1', False)
    # Processing the call keyword arguments (line 285)
    kwargs_176805 = {}
    # Getting the type of 'len' (line 285)
    len_176803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'len', False)
    # Calling len(args, kwargs) (line 285)
    len_call_result_176806 = invoke(stypy.reporting.localization.Localization(__file__, 285, 7), len_176803, *[c1_176804], **kwargs_176805)
    
    
    # Call to len(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'c2' (line 285)
    c2_176808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'c2', False)
    # Processing the call keyword arguments (line 285)
    kwargs_176809 = {}
    # Getting the type of 'len' (line 285)
    len_176807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 17), 'len', False)
    # Calling len(args, kwargs) (line 285)
    len_call_result_176810 = invoke(stypy.reporting.localization.Localization(__file__, 285, 17), len_176807, *[c2_176808], **kwargs_176809)
    
    # Applying the binary operator '>' (line 285)
    result_gt_176811 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 7), '>', len_call_result_176806, len_call_result_176810)
    
    # Testing the type of an if condition (line 285)
    if_condition_176812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), result_gt_176811)
    # Assigning a type to the variable 'if_condition_176812' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'if_condition_176812', if_condition_176812)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 286)
    c1_176813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 286)
    c2_176814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'c2')
    # Obtaining the member 'size' of a type (line 286)
    size_176815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), c2_176814, 'size')
    slice_176816 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 286, 8), None, size_176815, None)
    # Getting the type of 'c1' (line 286)
    c1_176817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___176818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), c1_176817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_176819 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___176818, slice_176816)
    
    # Getting the type of 'c2' (line 286)
    c2_176820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'c2')
    # Applying the binary operator '-=' (line 286)
    result_isub_176821 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 8), '-=', subscript_call_result_176819, c2_176820)
    # Getting the type of 'c1' (line 286)
    c1_176822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'c1')
    # Getting the type of 'c2' (line 286)
    c2_176823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'c2')
    # Obtaining the member 'size' of a type (line 286)
    size_176824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), c2_176823, 'size')
    slice_176825 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 286, 8), None, size_176824, None)
    # Storing an element on a container (line 286)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 8), c1_176822, (slice_176825, result_isub_176821))
    
    
    # Assigning a Name to a Name (line 287):
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'c1' (line 287)
    c1_176826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'ret', c1_176826)
    # SSA branch for the else part of an if statement (line 285)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a UnaryOp to a Name (line 289):
    
    # Assigning a UnaryOp to a Name (line 289):
    
    # Getting the type of 'c2' (line 289)
    c2_176827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), 'c2')
    # Applying the 'usub' unary operator (line 289)
    result___neg___176828 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 13), 'usub', c2_176827)
    
    # Assigning a type to the variable 'c2' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'c2', result___neg___176828)
    
    # Getting the type of 'c2' (line 290)
    c2_176829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 290)
    c1_176830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'c1')
    # Obtaining the member 'size' of a type (line 290)
    size_176831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), c1_176830, 'size')
    slice_176832 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 290, 8), None, size_176831, None)
    # Getting the type of 'c2' (line 290)
    c2_176833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___176834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), c2_176833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 290)
    subscript_call_result_176835 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___176834, slice_176832)
    
    # Getting the type of 'c1' (line 290)
    c1_176836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'c1')
    # Applying the binary operator '+=' (line 290)
    result_iadd_176837 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 8), '+=', subscript_call_result_176835, c1_176836)
    # Getting the type of 'c2' (line 290)
    c2_176838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'c2')
    # Getting the type of 'c1' (line 290)
    c1_176839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'c1')
    # Obtaining the member 'size' of a type (line 290)
    size_176840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), c1_176839, 'size')
    slice_176841 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 290, 8), None, size_176840, None)
    # Storing an element on a container (line 290)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 8), c2_176838, (slice_176841, result_iadd_176837))
    
    
    # Assigning a Name to a Name (line 291):
    
    # Assigning a Name to a Name (line 291):
    # Getting the type of 'c2' (line 291)
    c2_176842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'ret', c2_176842)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'ret' (line 292)
    ret_176845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'ret', False)
    # Processing the call keyword arguments (line 292)
    kwargs_176846 = {}
    # Getting the type of 'pu' (line 292)
    pu_176843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 292)
    trimseq_176844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 11), pu_176843, 'trimseq')
    # Calling trimseq(args, kwargs) (line 292)
    trimseq_call_result_176847 = invoke(stypy.reporting.localization.Localization(__file__, 292, 11), trimseq_176844, *[ret_176845], **kwargs_176846)
    
    # Assigning a type to the variable 'stypy_return_type' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type', trimseq_call_result_176847)
    
    # ################# End of 'polysub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polysub' in the type store
    # Getting the type of 'stypy_return_type' (line 249)
    stypy_return_type_176848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polysub'
    return stypy_return_type_176848

# Assigning a type to the variable 'polysub' (line 249)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'polysub', polysub)

@norecursion
def polymulx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polymulx'
    module_type_store = module_type_store.open_function_context('polymulx', 295, 0, False)
    
    # Passed parameters checking function
    polymulx.stypy_localization = localization
    polymulx.stypy_type_of_self = None
    polymulx.stypy_type_store = module_type_store
    polymulx.stypy_function_name = 'polymulx'
    polymulx.stypy_param_names_list = ['c']
    polymulx.stypy_varargs_param_name = None
    polymulx.stypy_kwargs_param_name = None
    polymulx.stypy_call_defaults = defaults
    polymulx.stypy_call_varargs = varargs
    polymulx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polymulx', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polymulx', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polymulx(...)' code ##################

    str_176849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, (-1)), 'str', 'Multiply a polynomial by x.\n\n    Multiply the polynomial `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of polynomial coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.5.0\n\n    ')
    
    # Assigning a Call to a List (line 320):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 320)
    # Processing the call arguments (line 320)
    
    # Obtaining an instance of the builtin type 'list' (line 320)
    list_176852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 320)
    # Adding element type (line 320)
    # Getting the type of 'c' (line 320)
    c_176853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 23), list_176852, c_176853)
    
    # Processing the call keyword arguments (line 320)
    kwargs_176854 = {}
    # Getting the type of 'pu' (line 320)
    pu_176850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 320)
    as_series_176851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 10), pu_176850, 'as_series')
    # Calling as_series(args, kwargs) (line 320)
    as_series_call_result_176855 = invoke(stypy.reporting.localization.Localization(__file__, 320, 10), as_series_176851, *[list_176852], **kwargs_176854)
    
    # Assigning a type to the variable 'call_assignment_176491' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'call_assignment_176491', as_series_call_result_176855)
    
    # Assigning a Call to a Name (line 320):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176859 = {}
    # Getting the type of 'call_assignment_176491' (line 320)
    call_assignment_176491_176856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'call_assignment_176491', False)
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___176857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 4), call_assignment_176491_176856, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176860 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176857, *[int_176858], **kwargs_176859)
    
    # Assigning a type to the variable 'call_assignment_176492' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'call_assignment_176492', getitem___call_result_176860)
    
    # Assigning a Name to a Name (line 320):
    # Getting the type of 'call_assignment_176492' (line 320)
    call_assignment_176492_176861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'call_assignment_176492')
    # Assigning a type to the variable 'c' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 5), 'c', call_assignment_176492_176861)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'c' (line 322)
    c_176863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'c', False)
    # Processing the call keyword arguments (line 322)
    kwargs_176864 = {}
    # Getting the type of 'len' (line 322)
    len_176862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 7), 'len', False)
    # Calling len(args, kwargs) (line 322)
    len_call_result_176865 = invoke(stypy.reporting.localization.Localization(__file__, 322, 7), len_176862, *[c_176863], **kwargs_176864)
    
    int_176866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 17), 'int')
    # Applying the binary operator '==' (line 322)
    result_eq_176867 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), '==', len_call_result_176865, int_176866)
    
    
    
    # Obtaining the type of the subscript
    int_176868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 25), 'int')
    # Getting the type of 'c' (line 322)
    c_176869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'c')
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___176870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), c_176869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_176871 = invoke(stypy.reporting.localization.Localization(__file__, 322, 23), getitem___176870, int_176868)
    
    int_176872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 31), 'int')
    # Applying the binary operator '==' (line 322)
    result_eq_176873 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 23), '==', subscript_call_result_176871, int_176872)
    
    # Applying the binary operator 'and' (line 322)
    result_and_keyword_176874 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), 'and', result_eq_176867, result_eq_176873)
    
    # Testing the type of an if condition (line 322)
    if_condition_176875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_and_keyword_176874)
    # Assigning a type to the variable 'if_condition_176875' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_176875', if_condition_176875)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 323)
    c_176876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'stypy_return_type', c_176876)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to empty(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Call to len(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'c' (line 325)
    c_176880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'c', False)
    # Processing the call keyword arguments (line 325)
    kwargs_176881 = {}
    # Getting the type of 'len' (line 325)
    len_176879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'len', False)
    # Calling len(args, kwargs) (line 325)
    len_call_result_176882 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), len_176879, *[c_176880], **kwargs_176881)
    
    int_176883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'int')
    # Applying the binary operator '+' (line 325)
    result_add_176884 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 19), '+', len_call_result_176882, int_176883)
    
    # Processing the call keyword arguments (line 325)
    # Getting the type of 'c' (line 325)
    c_176885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 37), 'c', False)
    # Obtaining the member 'dtype' of a type (line 325)
    dtype_176886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 37), c_176885, 'dtype')
    keyword_176887 = dtype_176886
    kwargs_176888 = {'dtype': keyword_176887}
    # Getting the type of 'np' (line 325)
    np_176877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 325)
    empty_176878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 10), np_176877, 'empty')
    # Calling empty(args, kwargs) (line 325)
    empty_call_result_176889 = invoke(stypy.reporting.localization.Localization(__file__, 325, 10), empty_176878, *[result_add_176884], **kwargs_176888)
    
    # Assigning a type to the variable 'prd' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'prd', empty_call_result_176889)
    
    # Assigning a BinOp to a Subscript (line 326):
    
    # Assigning a BinOp to a Subscript (line 326):
    
    # Obtaining the type of the subscript
    int_176890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 15), 'int')
    # Getting the type of 'c' (line 326)
    c_176891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___176892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 13), c_176891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_176893 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), getitem___176892, int_176890)
    
    int_176894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 18), 'int')
    # Applying the binary operator '*' (line 326)
    result_mul_176895 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 13), '*', subscript_call_result_176893, int_176894)
    
    # Getting the type of 'prd' (line 326)
    prd_176896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'prd')
    int_176897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 8), 'int')
    # Storing an element on a container (line 326)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 4), prd_176896, (int_176897, result_mul_176895))
    
    # Assigning a Name to a Subscript (line 327):
    
    # Assigning a Name to a Subscript (line 327):
    # Getting the type of 'c' (line 327)
    c_176898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 14), 'c')
    # Getting the type of 'prd' (line 327)
    prd_176899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'prd')
    int_176900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'int')
    slice_176901 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 327, 4), int_176900, None, None)
    # Storing an element on a container (line 327)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 4), prd_176899, (slice_176901, c_176898))
    # Getting the type of 'prd' (line 328)
    prd_176902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type', prd_176902)
    
    # ################# End of 'polymulx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polymulx' in the type store
    # Getting the type of 'stypy_return_type' (line 295)
    stypy_return_type_176903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176903)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polymulx'
    return stypy_return_type_176903

# Assigning a type to the variable 'polymulx' (line 295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'polymulx', polymulx)

@norecursion
def polymul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polymul'
    module_type_store = module_type_store.open_function_context('polymul', 331, 0, False)
    
    # Passed parameters checking function
    polymul.stypy_localization = localization
    polymul.stypy_type_of_self = None
    polymul.stypy_type_store = module_type_store
    polymul.stypy_function_name = 'polymul'
    polymul.stypy_param_names_list = ['c1', 'c2']
    polymul.stypy_varargs_param_name = None
    polymul.stypy_kwargs_param_name = None
    polymul.stypy_call_defaults = defaults
    polymul.stypy_call_varargs = varargs
    polymul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polymul', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polymul', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polymul(...)' code ##################

    str_176904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, (-1)), 'str', '\n    Multiply one polynomial by another.\n\n    Returns the product of two polynomials `c1` * `c2`.  The arguments are\n    sequences of coefficients, from lowest order term to highest, e.g.,\n    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2.``\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of coefficients representing a polynomial, relative to the\n        "standard" basis, and ordered from lowest order term to highest.\n\n    Returns\n    -------\n    out : ndarray\n        Of the coefficients of their product.\n\n    See Also\n    --------\n    polyadd, polysub, polydiv, polypow\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> P.polymul(c1,c2)\n    array([  3.,   8.,  14.,   8.,   3.])\n\n    ')
    
    # Assigning a Call to a List (line 364):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 364)
    # Processing the call arguments (line 364)
    
    # Obtaining an instance of the builtin type 'list' (line 364)
    list_176907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 364)
    # Adding element type (line 364)
    # Getting the type of 'c1' (line 364)
    c1_176908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 28), list_176907, c1_176908)
    # Adding element type (line 364)
    # Getting the type of 'c2' (line 364)
    c2_176909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 28), list_176907, c2_176909)
    
    # Processing the call keyword arguments (line 364)
    kwargs_176910 = {}
    # Getting the type of 'pu' (line 364)
    pu_176905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 364)
    as_series_176906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), pu_176905, 'as_series')
    # Calling as_series(args, kwargs) (line 364)
    as_series_call_result_176911 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), as_series_176906, *[list_176907], **kwargs_176910)
    
    # Assigning a type to the variable 'call_assignment_176493' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176493', as_series_call_result_176911)
    
    # Assigning a Call to a Name (line 364):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176915 = {}
    # Getting the type of 'call_assignment_176493' (line 364)
    call_assignment_176493_176912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176493', False)
    # Obtaining the member '__getitem__' of a type (line 364)
    getitem___176913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 4), call_assignment_176493_176912, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176916 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176913, *[int_176914], **kwargs_176915)
    
    # Assigning a type to the variable 'call_assignment_176494' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176494', getitem___call_result_176916)
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'call_assignment_176494' (line 364)
    call_assignment_176494_176917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176494')
    # Assigning a type to the variable 'c1' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 5), 'c1', call_assignment_176494_176917)
    
    # Assigning a Call to a Name (line 364):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176921 = {}
    # Getting the type of 'call_assignment_176493' (line 364)
    call_assignment_176493_176918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176493', False)
    # Obtaining the member '__getitem__' of a type (line 364)
    getitem___176919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 4), call_assignment_176493_176918, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176922 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176919, *[int_176920], **kwargs_176921)
    
    # Assigning a type to the variable 'call_assignment_176495' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176495', getitem___call_result_176922)
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'call_assignment_176495' (line 364)
    call_assignment_176495_176923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'call_assignment_176495')
    # Assigning a type to the variable 'c2' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 9), 'c2', call_assignment_176495_176923)
    
    # Assigning a Call to a Name (line 365):
    
    # Assigning a Call to a Name (line 365):
    
    # Call to convolve(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'c1' (line 365)
    c1_176926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'c1', False)
    # Getting the type of 'c2' (line 365)
    c2_176927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 26), 'c2', False)
    # Processing the call keyword arguments (line 365)
    kwargs_176928 = {}
    # Getting the type of 'np' (line 365)
    np_176924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 10), 'np', False)
    # Obtaining the member 'convolve' of a type (line 365)
    convolve_176925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 10), np_176924, 'convolve')
    # Calling convolve(args, kwargs) (line 365)
    convolve_call_result_176929 = invoke(stypy.reporting.localization.Localization(__file__, 365, 10), convolve_176925, *[c1_176926, c2_176927], **kwargs_176928)
    
    # Assigning a type to the variable 'ret' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'ret', convolve_call_result_176929)
    
    # Call to trimseq(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'ret' (line 366)
    ret_176932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'ret', False)
    # Processing the call keyword arguments (line 366)
    kwargs_176933 = {}
    # Getting the type of 'pu' (line 366)
    pu_176930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 366)
    trimseq_176931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 11), pu_176930, 'trimseq')
    # Calling trimseq(args, kwargs) (line 366)
    trimseq_call_result_176934 = invoke(stypy.reporting.localization.Localization(__file__, 366, 11), trimseq_176931, *[ret_176932], **kwargs_176933)
    
    # Assigning a type to the variable 'stypy_return_type' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type', trimseq_call_result_176934)
    
    # ################# End of 'polymul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polymul' in the type store
    # Getting the type of 'stypy_return_type' (line 331)
    stypy_return_type_176935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_176935)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polymul'
    return stypy_return_type_176935

# Assigning a type to the variable 'polymul' (line 331)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 0), 'polymul', polymul)

@norecursion
def polydiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polydiv'
    module_type_store = module_type_store.open_function_context('polydiv', 369, 0, False)
    
    # Passed parameters checking function
    polydiv.stypy_localization = localization
    polydiv.stypy_type_of_self = None
    polydiv.stypy_type_store = module_type_store
    polydiv.stypy_function_name = 'polydiv'
    polydiv.stypy_param_names_list = ['c1', 'c2']
    polydiv.stypy_varargs_param_name = None
    polydiv.stypy_kwargs_param_name = None
    polydiv.stypy_call_defaults = defaults
    polydiv.stypy_call_varargs = varargs
    polydiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polydiv', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polydiv', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polydiv(...)' code ##################

    str_176936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, (-1)), 'str', '\n    Divide one polynomial by another.\n\n    Returns the quotient-with-remainder of two polynomials `c1` / `c2`.\n    The arguments are sequences of coefficients, from lowest order term\n    to highest, e.g., [1,2,3] represents ``1 + 2*x + 3*x**2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of polynomial coefficients ordered from low to high.\n\n    Returns\n    -------\n    [quo, rem] : ndarrays\n        Of coefficient series representing the quotient and remainder.\n\n    See Also\n    --------\n    polyadd, polysub, polymul, polypow\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> P.polydiv(c1,c2)\n    (array([ 3.]), array([-8., -4.]))\n    >>> P.polydiv(c2,c1)\n    (array([ 0.33333333]), array([ 2.66666667,  1.33333333]))\n\n    ')
    
    # Assigning a Call to a List (line 403):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 403)
    # Processing the call arguments (line 403)
    
    # Obtaining an instance of the builtin type 'list' (line 403)
    list_176939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 403)
    # Adding element type (line 403)
    # Getting the type of 'c1' (line 403)
    c1_176940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 28), list_176939, c1_176940)
    # Adding element type (line 403)
    # Getting the type of 'c2' (line 403)
    c2_176941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 28), list_176939, c2_176941)
    
    # Processing the call keyword arguments (line 403)
    kwargs_176942 = {}
    # Getting the type of 'pu' (line 403)
    pu_176937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 403)
    as_series_176938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 15), pu_176937, 'as_series')
    # Calling as_series(args, kwargs) (line 403)
    as_series_call_result_176943 = invoke(stypy.reporting.localization.Localization(__file__, 403, 15), as_series_176938, *[list_176939], **kwargs_176942)
    
    # Assigning a type to the variable 'call_assignment_176496' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176496', as_series_call_result_176943)
    
    # Assigning a Call to a Name (line 403):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176947 = {}
    # Getting the type of 'call_assignment_176496' (line 403)
    call_assignment_176496_176944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176496', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___176945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 4), call_assignment_176496_176944, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176948 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176945, *[int_176946], **kwargs_176947)
    
    # Assigning a type to the variable 'call_assignment_176497' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176497', getitem___call_result_176948)
    
    # Assigning a Name to a Name (line 403):
    # Getting the type of 'call_assignment_176497' (line 403)
    call_assignment_176497_176949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176497')
    # Assigning a type to the variable 'c1' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 5), 'c1', call_assignment_176497_176949)
    
    # Assigning a Call to a Name (line 403):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_176952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 4), 'int')
    # Processing the call keyword arguments
    kwargs_176953 = {}
    # Getting the type of 'call_assignment_176496' (line 403)
    call_assignment_176496_176950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176496', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___176951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 4), call_assignment_176496_176950, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_176954 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___176951, *[int_176952], **kwargs_176953)
    
    # Assigning a type to the variable 'call_assignment_176498' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176498', getitem___call_result_176954)
    
    # Assigning a Name to a Name (line 403):
    # Getting the type of 'call_assignment_176498' (line 403)
    call_assignment_176498_176955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'call_assignment_176498')
    # Assigning a type to the variable 'c2' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 9), 'c2', call_assignment_176498_176955)
    
    
    
    # Obtaining the type of the subscript
    int_176956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 10), 'int')
    # Getting the type of 'c2' (line 404)
    c2_176957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 7), 'c2')
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___176958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 7), c2_176957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_176959 = invoke(stypy.reporting.localization.Localization(__file__, 404, 7), getitem___176958, int_176956)
    
    int_176960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 17), 'int')
    # Applying the binary operator '==' (line 404)
    result_eq_176961 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 7), '==', subscript_call_result_176959, int_176960)
    
    # Testing the type of an if condition (line 404)
    if_condition_176962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 4), result_eq_176961)
    # Assigning a type to the variable 'if_condition_176962' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'if_condition_176962', if_condition_176962)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ZeroDivisionError(...): (line 405)
    # Processing the call keyword arguments (line 405)
    kwargs_176964 = {}
    # Getting the type of 'ZeroDivisionError' (line 405)
    ZeroDivisionError_176963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 14), 'ZeroDivisionError', False)
    # Calling ZeroDivisionError(args, kwargs) (line 405)
    ZeroDivisionError_call_result_176965 = invoke(stypy.reporting.localization.Localization(__file__, 405, 14), ZeroDivisionError_176963, *[], **kwargs_176964)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 405, 8), ZeroDivisionError_call_result_176965, 'raise parameter', BaseException)
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 407):
    
    # Assigning a Call to a Name (line 407):
    
    # Call to len(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'c1' (line 407)
    c1_176967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'c1', False)
    # Processing the call keyword arguments (line 407)
    kwargs_176968 = {}
    # Getting the type of 'len' (line 407)
    len_176966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'len', False)
    # Calling len(args, kwargs) (line 407)
    len_call_result_176969 = invoke(stypy.reporting.localization.Localization(__file__, 407, 11), len_176966, *[c1_176967], **kwargs_176968)
    
    # Assigning a type to the variable 'len1' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'len1', len_call_result_176969)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to len(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'c2' (line 408)
    c2_176971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'c2', False)
    # Processing the call keyword arguments (line 408)
    kwargs_176972 = {}
    # Getting the type of 'len' (line 408)
    len_176970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'len', False)
    # Calling len(args, kwargs) (line 408)
    len_call_result_176973 = invoke(stypy.reporting.localization.Localization(__file__, 408, 11), len_176970, *[c2_176971], **kwargs_176972)
    
    # Assigning a type to the variable 'len2' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'len2', len_call_result_176973)
    
    
    # Getting the type of 'len2' (line 409)
    len2_176974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 7), 'len2')
    int_176975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 15), 'int')
    # Applying the binary operator '==' (line 409)
    result_eq_176976 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 7), '==', len2_176974, int_176975)
    
    # Testing the type of an if condition (line 409)
    if_condition_176977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 4), result_eq_176976)
    # Assigning a type to the variable 'if_condition_176977' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'if_condition_176977', if_condition_176977)
    # SSA begins for if statement (line 409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 410)
    tuple_176978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 410)
    # Adding element type (line 410)
    # Getting the type of 'c1' (line 410)
    c1_176979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'c1')
    
    # Obtaining the type of the subscript
    int_176980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 21), 'int')
    # Getting the type of 'c2' (line 410)
    c2_176981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 18), 'c2')
    # Obtaining the member '__getitem__' of a type (line 410)
    getitem___176982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 18), c2_176981, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 410)
    subscript_call_result_176983 = invoke(stypy.reporting.localization.Localization(__file__, 410, 18), getitem___176982, int_176980)
    
    # Applying the binary operator 'div' (line 410)
    result_div_176984 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 15), 'div', c1_176979, subscript_call_result_176983)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 15), tuple_176978, result_div_176984)
    # Adding element type (line 410)
    
    # Obtaining the type of the subscript
    int_176985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 30), 'int')
    slice_176986 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 410, 26), None, int_176985, None)
    # Getting the type of 'c1' (line 410)
    c1_176987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 410)
    getitem___176988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 26), c1_176987, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 410)
    subscript_call_result_176989 = invoke(stypy.reporting.localization.Localization(__file__, 410, 26), getitem___176988, slice_176986)
    
    int_176990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 33), 'int')
    # Applying the binary operator '*' (line 410)
    result_mul_176991 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 26), '*', subscript_call_result_176989, int_176990)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 15), tuple_176978, result_mul_176991)
    
    # Assigning a type to the variable 'stypy_return_type' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'stypy_return_type', tuple_176978)
    # SSA branch for the else part of an if statement (line 409)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'len1' (line 411)
    len1_176992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 9), 'len1')
    # Getting the type of 'len2' (line 411)
    len2_176993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'len2')
    # Applying the binary operator '<' (line 411)
    result_lt_176994 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 9), '<', len1_176992, len2_176993)
    
    # Testing the type of an if condition (line 411)
    if_condition_176995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 9), result_lt_176994)
    # Assigning a type to the variable 'if_condition_176995' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 9), 'if_condition_176995', if_condition_176995)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 412)
    tuple_176996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 412)
    # Adding element type (line 412)
    
    # Obtaining the type of the subscript
    int_176997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 19), 'int')
    slice_176998 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 412, 15), None, int_176997, None)
    # Getting the type of 'c1' (line 412)
    c1_176999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___177000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 15), c1_176999, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_177001 = invoke(stypy.reporting.localization.Localization(__file__, 412, 15), getitem___177000, slice_176998)
    
    int_177002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 22), 'int')
    # Applying the binary operator '*' (line 412)
    result_mul_177003 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 15), '*', subscript_call_result_177001, int_177002)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 15), tuple_176996, result_mul_177003)
    # Adding element type (line 412)
    # Getting the type of 'c1' (line 412)
    c1_177004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 25), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 15), tuple_176996, c1_177004)
    
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'stypy_return_type', tuple_176996)
    # SSA branch for the else part of an if statement (line 411)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 414):
    
    # Assigning a BinOp to a Name (line 414):
    # Getting the type of 'len1' (line 414)
    len1_177005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'len1')
    # Getting the type of 'len2' (line 414)
    len2_177006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 22), 'len2')
    # Applying the binary operator '-' (line 414)
    result_sub_177007 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 15), '-', len1_177005, len2_177006)
    
    # Assigning a type to the variable 'dlen' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'dlen', result_sub_177007)
    
    # Assigning a Subscript to a Name (line 415):
    
    # Assigning a Subscript to a Name (line 415):
    
    # Obtaining the type of the subscript
    int_177008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 17), 'int')
    # Getting the type of 'c2' (line 415)
    c2_177009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 14), 'c2')
    # Obtaining the member '__getitem__' of a type (line 415)
    getitem___177010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 14), c2_177009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 415)
    subscript_call_result_177011 = invoke(stypy.reporting.localization.Localization(__file__, 415, 14), getitem___177010, int_177008)
    
    # Assigning a type to the variable 'scl' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'scl', subscript_call_result_177011)
    
    # Assigning a BinOp to a Name (line 416):
    
    # Assigning a BinOp to a Name (line 416):
    
    # Obtaining the type of the subscript
    int_177012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 17), 'int')
    slice_177013 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 416, 13), None, int_177012, None)
    # Getting the type of 'c2' (line 416)
    c2_177014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 13), 'c2')
    # Obtaining the member '__getitem__' of a type (line 416)
    getitem___177015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 13), c2_177014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 416)
    subscript_call_result_177016 = invoke(stypy.reporting.localization.Localization(__file__, 416, 13), getitem___177015, slice_177013)
    
    # Getting the type of 'scl' (line 416)
    scl_177017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 21), 'scl')
    # Applying the binary operator 'div' (line 416)
    result_div_177018 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 13), 'div', subscript_call_result_177016, scl_177017)
    
    # Assigning a type to the variable 'c2' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'c2', result_div_177018)
    
    # Assigning a Name to a Name (line 417):
    
    # Assigning a Name to a Name (line 417):
    # Getting the type of 'dlen' (line 417)
    dlen_177019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'dlen')
    # Assigning a type to the variable 'i' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'i', dlen_177019)
    
    # Assigning a BinOp to a Name (line 418):
    
    # Assigning a BinOp to a Name (line 418):
    # Getting the type of 'len1' (line 418)
    len1_177020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'len1')
    int_177021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 19), 'int')
    # Applying the binary operator '-' (line 418)
    result_sub_177022 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 12), '-', len1_177020, int_177021)
    
    # Assigning a type to the variable 'j' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'j', result_sub_177022)
    
    
    # Getting the type of 'i' (line 419)
    i_177023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 14), 'i')
    int_177024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 19), 'int')
    # Applying the binary operator '>=' (line 419)
    result_ge_177025 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 14), '>=', i_177023, int_177024)
    
    # Testing the type of an if condition (line 419)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 8), result_ge_177025)
    # SSA begins for while statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'c1' (line 420)
    c1_177026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 420)
    i_177027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'i')
    # Getting the type of 'j' (line 420)
    j_177028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 'j')
    slice_177029 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 420, 12), i_177027, j_177028, None)
    # Getting the type of 'c1' (line 420)
    c1_177030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'c1')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___177031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 12), c1_177030, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_177032 = invoke(stypy.reporting.localization.Localization(__file__, 420, 12), getitem___177031, slice_177029)
    
    # Getting the type of 'c2' (line 420)
    c2_177033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 420)
    j_177034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 29), 'j')
    # Getting the type of 'c1' (line 420)
    c1_177035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___177036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 26), c1_177035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_177037 = invoke(stypy.reporting.localization.Localization(__file__, 420, 26), getitem___177036, j_177034)
    
    # Applying the binary operator '*' (line 420)
    result_mul_177038 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 23), '*', c2_177033, subscript_call_result_177037)
    
    # Applying the binary operator '-=' (line 420)
    result_isub_177039 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 12), '-=', subscript_call_result_177032, result_mul_177038)
    # Getting the type of 'c1' (line 420)
    c1_177040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'c1')
    # Getting the type of 'i' (line 420)
    i_177041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'i')
    # Getting the type of 'j' (line 420)
    j_177042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 'j')
    slice_177043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 420, 12), i_177041, j_177042, None)
    # Storing an element on a container (line 420)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 12), c1_177040, (slice_177043, result_isub_177039))
    
    
    # Getting the type of 'i' (line 421)
    i_177044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'i')
    int_177045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 17), 'int')
    # Applying the binary operator '-=' (line 421)
    result_isub_177046 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 12), '-=', i_177044, int_177045)
    # Assigning a type to the variable 'i' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'i', result_isub_177046)
    
    
    # Getting the type of 'j' (line 422)
    j_177047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'j')
    int_177048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 17), 'int')
    # Applying the binary operator '-=' (line 422)
    result_isub_177049 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 12), '-=', j_177047, int_177048)
    # Assigning a type to the variable 'j' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'j', result_isub_177049)
    
    # SSA join for while statement (line 419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 423)
    tuple_177050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 423)
    # Adding element type (line 423)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 423)
    j_177051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 18), 'j')
    int_177052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 20), 'int')
    # Applying the binary operator '+' (line 423)
    result_add_177053 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 18), '+', j_177051, int_177052)
    
    slice_177054 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 423, 15), result_add_177053, None, None)
    # Getting the type of 'c1' (line 423)
    c1_177055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___177056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), c1_177055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_177057 = invoke(stypy.reporting.localization.Localization(__file__, 423, 15), getitem___177056, slice_177054)
    
    # Getting the type of 'scl' (line 423)
    scl_177058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), 'scl')
    # Applying the binary operator 'div' (line 423)
    result_div_177059 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 15), 'div', subscript_call_result_177057, scl_177058)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 15), tuple_177050, result_div_177059)
    # Adding element type (line 423)
    
    # Call to trimseq(...): (line 423)
    # Processing the call arguments (line 423)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 423)
    j_177062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 44), 'j', False)
    int_177063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 46), 'int')
    # Applying the binary operator '+' (line 423)
    result_add_177064 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 44), '+', j_177062, int_177063)
    
    slice_177065 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 423, 40), None, result_add_177064, None)
    # Getting the type of 'c1' (line 423)
    c1_177066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 40), 'c1', False)
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___177067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 40), c1_177066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_177068 = invoke(stypy.reporting.localization.Localization(__file__, 423, 40), getitem___177067, slice_177065)
    
    # Processing the call keyword arguments (line 423)
    kwargs_177069 = {}
    # Getting the type of 'pu' (line 423)
    pu_177060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 29), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 423)
    trimseq_177061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 29), pu_177060, 'trimseq')
    # Calling trimseq(args, kwargs) (line 423)
    trimseq_call_result_177070 = invoke(stypy.reporting.localization.Localization(__file__, 423, 29), trimseq_177061, *[subscript_call_result_177068], **kwargs_177069)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 15), tuple_177050, trimseq_call_result_177070)
    
    # Assigning a type to the variable 'stypy_return_type' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'stypy_return_type', tuple_177050)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 409)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polydiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polydiv' in the type store
    # Getting the type of 'stypy_return_type' (line 369)
    stypy_return_type_177071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177071)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polydiv'
    return stypy_return_type_177071

# Assigning a type to the variable 'polydiv' (line 369)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 0), 'polydiv', polydiv)

@norecursion
def polypow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 426)
    None_177072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 29), 'None')
    defaults = [None_177072]
    # Create a new context for function 'polypow'
    module_type_store = module_type_store.open_function_context('polypow', 426, 0, False)
    
    # Passed parameters checking function
    polypow.stypy_localization = localization
    polypow.stypy_type_of_self = None
    polypow.stypy_type_store = module_type_store
    polypow.stypy_function_name = 'polypow'
    polypow.stypy_param_names_list = ['c', 'pow', 'maxpower']
    polypow.stypy_varargs_param_name = None
    polypow.stypy_kwargs_param_name = None
    polypow.stypy_call_defaults = defaults
    polypow.stypy_call_varargs = varargs
    polypow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polypow', ['c', 'pow', 'maxpower'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polypow', localization, ['c', 'pow', 'maxpower'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polypow(...)' code ##################

    str_177073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'str', 'Raise a polynomial to a power.\n\n    Returns the polynomial `c` raised to the power `pow`. The argument\n    `c` is a sequence of coefficients ordered from low to high. i.e.,\n    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of array of series coefficients ordered from low to\n        high degree.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Power series of power.\n\n    See Also\n    --------\n    polyadd, polysub, polymul, polydiv\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a Call to a List (line 458):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 458)
    # Processing the call arguments (line 458)
    
    # Obtaining an instance of the builtin type 'list' (line 458)
    list_177076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 458)
    # Adding element type (line 458)
    # Getting the type of 'c' (line 458)
    c_177077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 23), list_177076, c_177077)
    
    # Processing the call keyword arguments (line 458)
    kwargs_177078 = {}
    # Getting the type of 'pu' (line 458)
    pu_177074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 458)
    as_series_177075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 10), pu_177074, 'as_series')
    # Calling as_series(args, kwargs) (line 458)
    as_series_call_result_177079 = invoke(stypy.reporting.localization.Localization(__file__, 458, 10), as_series_177075, *[list_177076], **kwargs_177078)
    
    # Assigning a type to the variable 'call_assignment_176499' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'call_assignment_176499', as_series_call_result_177079)
    
    # Assigning a Call to a Name (line 458):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_177082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 4), 'int')
    # Processing the call keyword arguments
    kwargs_177083 = {}
    # Getting the type of 'call_assignment_176499' (line 458)
    call_assignment_176499_177080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'call_assignment_176499', False)
    # Obtaining the member '__getitem__' of a type (line 458)
    getitem___177081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 4), call_assignment_176499_177080, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_177084 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___177081, *[int_177082], **kwargs_177083)
    
    # Assigning a type to the variable 'call_assignment_176500' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'call_assignment_176500', getitem___call_result_177084)
    
    # Assigning a Name to a Name (line 458):
    # Getting the type of 'call_assignment_176500' (line 458)
    call_assignment_176500_177085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'call_assignment_176500')
    # Assigning a type to the variable 'c' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 5), 'c', call_assignment_176500_177085)
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to int(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'pow' (line 459)
    pow_177087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'pow', False)
    # Processing the call keyword arguments (line 459)
    kwargs_177088 = {}
    # Getting the type of 'int' (line 459)
    int_177086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'int', False)
    # Calling int(args, kwargs) (line 459)
    int_call_result_177089 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), int_177086, *[pow_177087], **kwargs_177088)
    
    # Assigning a type to the variable 'power' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'power', int_call_result_177089)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'power' (line 460)
    power_177090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 7), 'power')
    # Getting the type of 'pow' (line 460)
    pow_177091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'pow')
    # Applying the binary operator '!=' (line 460)
    result_ne_177092 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 7), '!=', power_177090, pow_177091)
    
    
    # Getting the type of 'power' (line 460)
    power_177093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'power')
    int_177094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 31), 'int')
    # Applying the binary operator '<' (line 460)
    result_lt_177095 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 23), '<', power_177093, int_177094)
    
    # Applying the binary operator 'or' (line 460)
    result_or_keyword_177096 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 7), 'or', result_ne_177092, result_lt_177095)
    
    # Testing the type of an if condition (line 460)
    if_condition_177097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 4), result_or_keyword_177096)
    # Assigning a type to the variable 'if_condition_177097' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'if_condition_177097', if_condition_177097)
    # SSA begins for if statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 461)
    # Processing the call arguments (line 461)
    str_177099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 25), 'str', 'Power must be a non-negative integer.')
    # Processing the call keyword arguments (line 461)
    kwargs_177100 = {}
    # Getting the type of 'ValueError' (line 461)
    ValueError_177098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 461)
    ValueError_call_result_177101 = invoke(stypy.reporting.localization.Localization(__file__, 461, 14), ValueError_177098, *[str_177099], **kwargs_177100)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 461, 8), ValueError_call_result_177101, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 460)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxpower' (line 462)
    maxpower_177102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 9), 'maxpower')
    # Getting the type of 'None' (line 462)
    None_177103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 25), 'None')
    # Applying the binary operator 'isnot' (line 462)
    result_is_not_177104 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 9), 'isnot', maxpower_177102, None_177103)
    
    
    # Getting the type of 'power' (line 462)
    power_177105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 34), 'power')
    # Getting the type of 'maxpower' (line 462)
    maxpower_177106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'maxpower')
    # Applying the binary operator '>' (line 462)
    result_gt_177107 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 34), '>', power_177105, maxpower_177106)
    
    # Applying the binary operator 'and' (line 462)
    result_and_keyword_177108 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 9), 'and', result_is_not_177104, result_gt_177107)
    
    # Testing the type of an if condition (line 462)
    if_condition_177109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 9), result_and_keyword_177108)
    # Assigning a type to the variable 'if_condition_177109' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 9), 'if_condition_177109', if_condition_177109)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 463)
    # Processing the call arguments (line 463)
    str_177111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 25), 'str', 'Power is too large')
    # Processing the call keyword arguments (line 463)
    kwargs_177112 = {}
    # Getting the type of 'ValueError' (line 463)
    ValueError_177110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 463)
    ValueError_call_result_177113 = invoke(stypy.reporting.localization.Localization(__file__, 463, 14), ValueError_177110, *[str_177111], **kwargs_177112)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 463, 8), ValueError_call_result_177113, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 462)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 464)
    power_177114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 9), 'power')
    int_177115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 18), 'int')
    # Applying the binary operator '==' (line 464)
    result_eq_177116 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 9), '==', power_177114, int_177115)
    
    # Testing the type of an if condition (line 464)
    if_condition_177117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 9), result_eq_177116)
    # Assigning a type to the variable 'if_condition_177117' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 9), 'if_condition_177117', if_condition_177117)
    # SSA begins for if statement (line 464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 465)
    # Processing the call arguments (line 465)
    
    # Obtaining an instance of the builtin type 'list' (line 465)
    list_177120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 465)
    # Adding element type (line 465)
    int_177121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 24), list_177120, int_177121)
    
    # Processing the call keyword arguments (line 465)
    # Getting the type of 'c' (line 465)
    c_177122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 35), 'c', False)
    # Obtaining the member 'dtype' of a type (line 465)
    dtype_177123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 35), c_177122, 'dtype')
    keyword_177124 = dtype_177123
    kwargs_177125 = {'dtype': keyword_177124}
    # Getting the type of 'np' (line 465)
    np_177118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 465)
    array_177119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 15), np_177118, 'array')
    # Calling array(args, kwargs) (line 465)
    array_call_result_177126 = invoke(stypy.reporting.localization.Localization(__file__, 465, 15), array_177119, *[list_177120], **kwargs_177125)
    
    # Assigning a type to the variable 'stypy_return_type' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', array_call_result_177126)
    # SSA branch for the else part of an if statement (line 464)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 466)
    power_177127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 9), 'power')
    int_177128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 18), 'int')
    # Applying the binary operator '==' (line 466)
    result_eq_177129 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 9), '==', power_177127, int_177128)
    
    # Testing the type of an if condition (line 466)
    if_condition_177130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 9), result_eq_177129)
    # Assigning a type to the variable 'if_condition_177130' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 9), 'if_condition_177130', if_condition_177130)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 467)
    c_177131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'stypy_return_type', c_177131)
    # SSA branch for the else part of an if statement (line 466)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 471):
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'c' (line 471)
    c_177132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 14), 'c')
    # Assigning a type to the variable 'prd' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'prd', c_177132)
    
    
    # Call to range(...): (line 472)
    # Processing the call arguments (line 472)
    int_177134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 23), 'int')
    # Getting the type of 'power' (line 472)
    power_177135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 26), 'power', False)
    int_177136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 34), 'int')
    # Applying the binary operator '+' (line 472)
    result_add_177137 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 26), '+', power_177135, int_177136)
    
    # Processing the call keyword arguments (line 472)
    kwargs_177138 = {}
    # Getting the type of 'range' (line 472)
    range_177133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 17), 'range', False)
    # Calling range(args, kwargs) (line 472)
    range_call_result_177139 = invoke(stypy.reporting.localization.Localization(__file__, 472, 17), range_177133, *[int_177134, result_add_177137], **kwargs_177138)
    
    # Testing the type of a for loop iterable (line 472)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 472, 8), range_call_result_177139)
    # Getting the type of the for loop variable (line 472)
    for_loop_var_177140 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 472, 8), range_call_result_177139)
    # Assigning a type to the variable 'i' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'i', for_loop_var_177140)
    # SSA begins for a for statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to convolve(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'prd' (line 473)
    prd_177143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 30), 'prd', False)
    # Getting the type of 'c' (line 473)
    c_177144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 35), 'c', False)
    # Processing the call keyword arguments (line 473)
    kwargs_177145 = {}
    # Getting the type of 'np' (line 473)
    np_177141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 18), 'np', False)
    # Obtaining the member 'convolve' of a type (line 473)
    convolve_177142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 18), np_177141, 'convolve')
    # Calling convolve(args, kwargs) (line 473)
    convolve_call_result_177146 = invoke(stypy.reporting.localization.Localization(__file__, 473, 18), convolve_177142, *[prd_177143, c_177144], **kwargs_177145)
    
    # Assigning a type to the variable 'prd' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'prd', convolve_call_result_177146)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 474)
    prd_177147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'stypy_return_type', prd_177147)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 464)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polypow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polypow' in the type store
    # Getting the type of 'stypy_return_type' (line 426)
    stypy_return_type_177148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polypow'
    return stypy_return_type_177148

# Assigning a type to the variable 'polypow' (line 426)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'polypow', polypow)

@norecursion
def polyder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_177149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 17), 'int')
    int_177150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 24), 'int')
    int_177151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 32), 'int')
    defaults = [int_177149, int_177150, int_177151]
    # Create a new context for function 'polyder'
    module_type_store = module_type_store.open_function_context('polyder', 477, 0, False)
    
    # Passed parameters checking function
    polyder.stypy_localization = localization
    polyder.stypy_type_of_self = None
    polyder.stypy_type_store = module_type_store
    polyder.stypy_function_name = 'polyder'
    polyder.stypy_param_names_list = ['c', 'm', 'scl', 'axis']
    polyder.stypy_varargs_param_name = None
    polyder.stypy_kwargs_param_name = None
    polyder.stypy_call_defaults = defaults
    polyder.stypy_call_varargs = varargs
    polyder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyder', ['c', 'm', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyder', localization, ['c', 'm', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyder(...)' code ##################

    str_177152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, (-1)), 'str', '\n    Differentiate a polynomial.\n\n    Returns the polynomial coefficients `c` differentiated `m` times along\n    `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable).  The\n    argument `c` is an array of coefficients from low to high degree along\n    each axis, e.g., [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``\n    while [[1,2],[1,2]] represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is\n    ``x`` and axis=1 is ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of polynomial coefficients. If c is multidimensional the\n        different axis correspond to different variables with the degree\n        in each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change\n        of variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Polynomial coefficients of the derivative.\n\n    See Also\n    --------\n    polyint\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> c = (1,2,3,4) # 1 + 2x + 3x**2 + 4x**3\n    >>> P.polyder(c) # (d/dx)(c) = 2 + 6x + 12x**2\n    array([  2.,   6.,  12.])\n    >>> P.polyder(c,3) # (d**3/dx**3)(c) = 24\n    array([ 24.])\n    >>> P.polyder(c,scl=-1) # (d/d(-x))(c) = -2 - 6x - 12x**2\n    array([ -2.,  -6., -12.])\n    >>> P.polyder(c,2,-1) # (d**2/d(-x)**2)(c) = 6 + 24x\n    array([  6.,  24.])\n\n    ')
    
    # Assigning a Call to a Name (line 529):
    
    # Assigning a Call to a Name (line 529):
    
    # Call to array(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'c' (line 529)
    c_177155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 17), 'c', False)
    # Processing the call keyword arguments (line 529)
    int_177156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 26), 'int')
    keyword_177157 = int_177156
    int_177158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 34), 'int')
    keyword_177159 = int_177158
    kwargs_177160 = {'copy': keyword_177159, 'ndmin': keyword_177157}
    # Getting the type of 'np' (line 529)
    np_177153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 529)
    array_177154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 8), np_177153, 'array')
    # Calling array(args, kwargs) (line 529)
    array_call_result_177161 = invoke(stypy.reporting.localization.Localization(__file__, 529, 8), array_177154, *[c_177155], **kwargs_177160)
    
    # Assigning a type to the variable 'c' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'c', array_call_result_177161)
    
    
    # Getting the type of 'c' (line 530)
    c_177162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 530)
    dtype_177163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 7), c_177162, 'dtype')
    # Obtaining the member 'char' of a type (line 530)
    char_177164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 7), dtype_177163, 'char')
    str_177165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 530)
    result_contains_177166 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 7), 'in', char_177164, str_177165)
    
    # Testing the type of an if condition (line 530)
    if_condition_177167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 4), result_contains_177166)
    # Assigning a type to the variable 'if_condition_177167' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'if_condition_177167', if_condition_177167)
    # SSA begins for if statement (line 530)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 532):
    
    # Assigning a BinOp to a Name (line 532):
    # Getting the type of 'c' (line 532)
    c_177168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'c')
    float_177169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 16), 'float')
    # Applying the binary operator '+' (line 532)
    result_add_177170 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 12), '+', c_177168, float_177169)
    
    # Assigning a type to the variable 'c' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'c', result_add_177170)
    # SSA join for if statement (line 530)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 533):
    
    # Assigning a Attribute to a Name (line 533):
    # Getting the type of 'c' (line 533)
    c_177171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 10), 'c')
    # Obtaining the member 'dtype' of a type (line 533)
    dtype_177172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 10), c_177171, 'dtype')
    # Assigning a type to the variable 'cdt' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'cdt', dtype_177172)
    
    # Assigning a ListComp to a Tuple (line 534):
    
    # Assigning a Subscript to a Name (line 534):
    
    # Obtaining the type of the subscript
    int_177173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 534)
    list_177178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 534)
    # Adding element type (line 534)
    # Getting the type of 'm' (line 534)
    m_177179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 34), list_177178, m_177179)
    # Adding element type (line 534)
    # Getting the type of 'axis' (line 534)
    axis_177180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 34), list_177178, axis_177180)
    
    comprehension_177181 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_177178)
    # Assigning a type to the variable 't' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 't', comprehension_177181)
    
    # Call to int(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 't' (line 534)
    t_177175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 22), 't', False)
    # Processing the call keyword arguments (line 534)
    kwargs_177176 = {}
    # Getting the type of 'int' (line 534)
    int_177174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'int', False)
    # Calling int(args, kwargs) (line 534)
    int_call_result_177177 = invoke(stypy.reporting.localization.Localization(__file__, 534, 18), int_177174, *[t_177175], **kwargs_177176)
    
    list_177182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_177182, int_call_result_177177)
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___177183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 4), list_177182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_177184 = invoke(stypy.reporting.localization.Localization(__file__, 534, 4), getitem___177183, int_177173)
    
    # Assigning a type to the variable 'tuple_var_assignment_176501' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_176501', subscript_call_result_177184)
    
    # Assigning a Subscript to a Name (line 534):
    
    # Obtaining the type of the subscript
    int_177185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 534)
    list_177190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 534)
    # Adding element type (line 534)
    # Getting the type of 'm' (line 534)
    m_177191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 34), list_177190, m_177191)
    # Adding element type (line 534)
    # Getting the type of 'axis' (line 534)
    axis_177192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 34), list_177190, axis_177192)
    
    comprehension_177193 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_177190)
    # Assigning a type to the variable 't' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 't', comprehension_177193)
    
    # Call to int(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 't' (line 534)
    t_177187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 22), 't', False)
    # Processing the call keyword arguments (line 534)
    kwargs_177188 = {}
    # Getting the type of 'int' (line 534)
    int_177186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'int', False)
    # Calling int(args, kwargs) (line 534)
    int_call_result_177189 = invoke(stypy.reporting.localization.Localization(__file__, 534, 18), int_177186, *[t_177187], **kwargs_177188)
    
    list_177194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_177194, int_call_result_177189)
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___177195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 4), list_177194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_177196 = invoke(stypy.reporting.localization.Localization(__file__, 534, 4), getitem___177195, int_177185)
    
    # Assigning a type to the variable 'tuple_var_assignment_176502' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_176502', subscript_call_result_177196)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'tuple_var_assignment_176501' (line 534)
    tuple_var_assignment_176501_177197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_176501')
    # Assigning a type to the variable 'cnt' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'cnt', tuple_var_assignment_176501_177197)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'tuple_var_assignment_176502' (line 534)
    tuple_var_assignment_176502_177198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_176502')
    # Assigning a type to the variable 'iaxis' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 9), 'iaxis', tuple_var_assignment_176502_177198)
    
    
    # Getting the type of 'cnt' (line 536)
    cnt_177199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 7), 'cnt')
    # Getting the type of 'm' (line 536)
    m_177200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 14), 'm')
    # Applying the binary operator '!=' (line 536)
    result_ne_177201 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 7), '!=', cnt_177199, m_177200)
    
    # Testing the type of an if condition (line 536)
    if_condition_177202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 4), result_ne_177201)
    # Assigning a type to the variable 'if_condition_177202' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'if_condition_177202', if_condition_177202)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 537)
    # Processing the call arguments (line 537)
    str_177204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 25), 'str', 'The order of derivation must be integer')
    # Processing the call keyword arguments (line 537)
    kwargs_177205 = {}
    # Getting the type of 'ValueError' (line 537)
    ValueError_177203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 537)
    ValueError_call_result_177206 = invoke(stypy.reporting.localization.Localization(__file__, 537, 14), ValueError_177203, *[str_177204], **kwargs_177205)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 537, 8), ValueError_call_result_177206, 'raise parameter', BaseException)
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 538)
    cnt_177207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 7), 'cnt')
    int_177208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 13), 'int')
    # Applying the binary operator '<' (line 538)
    result_lt_177209 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 7), '<', cnt_177207, int_177208)
    
    # Testing the type of an if condition (line 538)
    if_condition_177210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 4), result_lt_177209)
    # Assigning a type to the variable 'if_condition_177210' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'if_condition_177210', if_condition_177210)
    # SSA begins for if statement (line 538)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 539)
    # Processing the call arguments (line 539)
    str_177212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 25), 'str', 'The order of derivation must be non-negative')
    # Processing the call keyword arguments (line 539)
    kwargs_177213 = {}
    # Getting the type of 'ValueError' (line 539)
    ValueError_177211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 539)
    ValueError_call_result_177214 = invoke(stypy.reporting.localization.Localization(__file__, 539, 14), ValueError_177211, *[str_177212], **kwargs_177213)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 539, 8), ValueError_call_result_177214, 'raise parameter', BaseException)
    # SSA join for if statement (line 538)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 540)
    iaxis_177215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 7), 'iaxis')
    # Getting the type of 'axis' (line 540)
    axis_177216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'axis')
    # Applying the binary operator '!=' (line 540)
    result_ne_177217 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 7), '!=', iaxis_177215, axis_177216)
    
    # Testing the type of an if condition (line 540)
    if_condition_177218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 4), result_ne_177217)
    # Assigning a type to the variable 'if_condition_177218' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'if_condition_177218', if_condition_177218)
    # SSA begins for if statement (line 540)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 541)
    # Processing the call arguments (line 541)
    str_177220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 541)
    kwargs_177221 = {}
    # Getting the type of 'ValueError' (line 541)
    ValueError_177219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 541)
    ValueError_call_result_177222 = invoke(stypy.reporting.localization.Localization(__file__, 541, 14), ValueError_177219, *[str_177220], **kwargs_177221)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 541, 8), ValueError_call_result_177222, 'raise parameter', BaseException)
    # SSA join for if statement (line 540)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 542)
    c_177223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 542)
    ndim_177224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 12), c_177223, 'ndim')
    # Applying the 'usub' unary operator (line 542)
    result___neg___177225 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 11), 'usub', ndim_177224)
    
    # Getting the type of 'iaxis' (line 542)
    iaxis_177226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 22), 'iaxis')
    # Applying the binary operator '<=' (line 542)
    result_le_177227 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 11), '<=', result___neg___177225, iaxis_177226)
    # Getting the type of 'c' (line 542)
    c_177228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 542)
    ndim_177229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 30), c_177228, 'ndim')
    # Applying the binary operator '<' (line 542)
    result_lt_177230 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 11), '<', iaxis_177226, ndim_177229)
    # Applying the binary operator '&' (line 542)
    result_and__177231 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 11), '&', result_le_177227, result_lt_177230)
    
    # Applying the 'not' unary operator (line 542)
    result_not__177232 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 7), 'not', result_and__177231)
    
    # Testing the type of an if condition (line 542)
    if_condition_177233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 4), result_not__177232)
    # Assigning a type to the variable 'if_condition_177233' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'if_condition_177233', if_condition_177233)
    # SSA begins for if statement (line 542)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 543)
    # Processing the call arguments (line 543)
    str_177235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 543)
    kwargs_177236 = {}
    # Getting the type of 'ValueError' (line 543)
    ValueError_177234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 543)
    ValueError_call_result_177237 = invoke(stypy.reporting.localization.Localization(__file__, 543, 14), ValueError_177234, *[str_177235], **kwargs_177236)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 543, 8), ValueError_call_result_177237, 'raise parameter', BaseException)
    # SSA join for if statement (line 542)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 544)
    iaxis_177238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 7), 'iaxis')
    int_177239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 15), 'int')
    # Applying the binary operator '<' (line 544)
    result_lt_177240 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 7), '<', iaxis_177238, int_177239)
    
    # Testing the type of an if condition (line 544)
    if_condition_177241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 4), result_lt_177240)
    # Assigning a type to the variable 'if_condition_177241' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'if_condition_177241', if_condition_177241)
    # SSA begins for if statement (line 544)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 545)
    iaxis_177242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'iaxis')
    # Getting the type of 'c' (line 545)
    c_177243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 545)
    ndim_177244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 17), c_177243, 'ndim')
    # Applying the binary operator '+=' (line 545)
    result_iadd_177245 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 8), '+=', iaxis_177242, ndim_177244)
    # Assigning a type to the variable 'iaxis' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'iaxis', result_iadd_177245)
    
    # SSA join for if statement (line 544)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 547)
    cnt_177246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 7), 'cnt')
    int_177247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 14), 'int')
    # Applying the binary operator '==' (line 547)
    result_eq_177248 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 7), '==', cnt_177246, int_177247)
    
    # Testing the type of an if condition (line 547)
    if_condition_177249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 4), result_eq_177248)
    # Assigning a type to the variable 'if_condition_177249' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'if_condition_177249', if_condition_177249)
    # SSA begins for if statement (line 547)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 548)
    c_177250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'stypy_return_type', c_177250)
    # SSA join for if statement (line 547)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 550):
    
    # Assigning a Call to a Name (line 550):
    
    # Call to rollaxis(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'c' (line 550)
    c_177253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'c', False)
    # Getting the type of 'iaxis' (line 550)
    iaxis_177254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 550)
    kwargs_177255 = {}
    # Getting the type of 'np' (line 550)
    np_177251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 550)
    rollaxis_177252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), np_177251, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 550)
    rollaxis_call_result_177256 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), rollaxis_177252, *[c_177253, iaxis_177254], **kwargs_177255)
    
    # Assigning a type to the variable 'c' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'c', rollaxis_call_result_177256)
    
    # Assigning a Call to a Name (line 551):
    
    # Assigning a Call to a Name (line 551):
    
    # Call to len(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'c' (line 551)
    c_177258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'c', False)
    # Processing the call keyword arguments (line 551)
    kwargs_177259 = {}
    # Getting the type of 'len' (line 551)
    len_177257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'len', False)
    # Calling len(args, kwargs) (line 551)
    len_call_result_177260 = invoke(stypy.reporting.localization.Localization(__file__, 551, 8), len_177257, *[c_177258], **kwargs_177259)
    
    # Assigning a type to the variable 'n' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'n', len_call_result_177260)
    
    
    # Getting the type of 'cnt' (line 552)
    cnt_177261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 7), 'cnt')
    # Getting the type of 'n' (line 552)
    n_177262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 14), 'n')
    # Applying the binary operator '>=' (line 552)
    result_ge_177263 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 7), '>=', cnt_177261, n_177262)
    
    # Testing the type of an if condition (line 552)
    if_condition_177264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 552, 4), result_ge_177263)
    # Assigning a type to the variable 'if_condition_177264' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'if_condition_177264', if_condition_177264)
    # SSA begins for if statement (line 552)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 553):
    
    # Assigning a BinOp to a Name (line 553):
    
    # Obtaining the type of the subscript
    int_177265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 15), 'int')
    slice_177266 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 553, 12), None, int_177265, None)
    # Getting the type of 'c' (line 553)
    c_177267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___177268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 12), c_177267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_177269 = invoke(stypy.reporting.localization.Localization(__file__, 553, 12), getitem___177268, slice_177266)
    
    int_177270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 18), 'int')
    # Applying the binary operator '*' (line 553)
    result_mul_177271 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 12), '*', subscript_call_result_177269, int_177270)
    
    # Assigning a type to the variable 'c' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'c', result_mul_177271)
    # SSA branch for the else part of an if statement (line 552)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'cnt' (line 555)
    cnt_177273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 23), 'cnt', False)
    # Processing the call keyword arguments (line 555)
    kwargs_177274 = {}
    # Getting the type of 'range' (line 555)
    range_177272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 17), 'range', False)
    # Calling range(args, kwargs) (line 555)
    range_call_result_177275 = invoke(stypy.reporting.localization.Localization(__file__, 555, 17), range_177272, *[cnt_177273], **kwargs_177274)
    
    # Testing the type of a for loop iterable (line 555)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 555, 8), range_call_result_177275)
    # Getting the type of the for loop variable (line 555)
    for_loop_var_177276 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 555, 8), range_call_result_177275)
    # Assigning a type to the variable 'i' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'i', for_loop_var_177276)
    # SSA begins for a for statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 556):
    
    # Assigning a BinOp to a Name (line 556):
    # Getting the type of 'n' (line 556)
    n_177277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'n')
    int_177278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 20), 'int')
    # Applying the binary operator '-' (line 556)
    result_sub_177279 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 16), '-', n_177277, int_177278)
    
    # Assigning a type to the variable 'n' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'n', result_sub_177279)
    
    # Getting the type of 'c' (line 557)
    c_177280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'c')
    # Getting the type of 'scl' (line 557)
    scl_177281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 17), 'scl')
    # Applying the binary operator '*=' (line 557)
    result_imul_177282 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 12), '*=', c_177280, scl_177281)
    # Assigning a type to the variable 'c' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'c', result_imul_177282)
    
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to empty(...): (line 558)
    # Processing the call arguments (line 558)
    
    # Obtaining an instance of the builtin type 'tuple' (line 558)
    tuple_177285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 558)
    # Adding element type (line 558)
    # Getting the type of 'n' (line 558)
    n_177286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 28), tuple_177285, n_177286)
    
    
    # Obtaining the type of the subscript
    int_177287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 42), 'int')
    slice_177288 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 558, 34), int_177287, None, None)
    # Getting the type of 'c' (line 558)
    c_177289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 558)
    shape_177290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 34), c_177289, 'shape')
    # Obtaining the member '__getitem__' of a type (line 558)
    getitem___177291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 34), shape_177290, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 558)
    subscript_call_result_177292 = invoke(stypy.reporting.localization.Localization(__file__, 558, 34), getitem___177291, slice_177288)
    
    # Applying the binary operator '+' (line 558)
    result_add_177293 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 27), '+', tuple_177285, subscript_call_result_177292)
    
    # Processing the call keyword arguments (line 558)
    # Getting the type of 'cdt' (line 558)
    cdt_177294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 53), 'cdt', False)
    keyword_177295 = cdt_177294
    kwargs_177296 = {'dtype': keyword_177295}
    # Getting the type of 'np' (line 558)
    np_177283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 558)
    empty_177284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 18), np_177283, 'empty')
    # Calling empty(args, kwargs) (line 558)
    empty_call_result_177297 = invoke(stypy.reporting.localization.Localization(__file__, 558, 18), empty_177284, *[result_add_177293], **kwargs_177296)
    
    # Assigning a type to the variable 'der' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'der', empty_call_result_177297)
    
    
    # Call to range(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'n' (line 559)
    n_177299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 27), 'n', False)
    int_177300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 30), 'int')
    int_177301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 33), 'int')
    # Processing the call keyword arguments (line 559)
    kwargs_177302 = {}
    # Getting the type of 'range' (line 559)
    range_177298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 21), 'range', False)
    # Calling range(args, kwargs) (line 559)
    range_call_result_177303 = invoke(stypy.reporting.localization.Localization(__file__, 559, 21), range_177298, *[n_177299, int_177300, int_177301], **kwargs_177302)
    
    # Testing the type of a for loop iterable (line 559)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 559, 12), range_call_result_177303)
    # Getting the type of the for loop variable (line 559)
    for_loop_var_177304 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 559, 12), range_call_result_177303)
    # Assigning a type to the variable 'j' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'j', for_loop_var_177304)
    # SSA begins for a for statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 560):
    
    # Assigning a BinOp to a Subscript (line 560):
    # Getting the type of 'j' (line 560)
    j_177305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 29), 'j')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 560)
    j_177306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 33), 'j')
    # Getting the type of 'c' (line 560)
    c_177307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 31), 'c')
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___177308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 31), c_177307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_177309 = invoke(stypy.reporting.localization.Localization(__file__, 560, 31), getitem___177308, j_177306)
    
    # Applying the binary operator '*' (line 560)
    result_mul_177310 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 29), '*', j_177305, subscript_call_result_177309)
    
    # Getting the type of 'der' (line 560)
    der_177311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'der')
    # Getting the type of 'j' (line 560)
    j_177312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'j')
    int_177313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 24), 'int')
    # Applying the binary operator '-' (line 560)
    result_sub_177314 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 20), '-', j_177312, int_177313)
    
    # Storing an element on a container (line 560)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 16), der_177311, (result_sub_177314, result_mul_177310))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 561):
    
    # Assigning a Name to a Name (line 561):
    # Getting the type of 'der' (line 561)
    der_177315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'der')
    # Assigning a type to the variable 'c' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'c', der_177315)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 552)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to rollaxis(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'c' (line 562)
    c_177318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'c', False)
    int_177319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 23), 'int')
    # Getting the type of 'iaxis' (line 562)
    iaxis_177320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 26), 'iaxis', False)
    int_177321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 34), 'int')
    # Applying the binary operator '+' (line 562)
    result_add_177322 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 26), '+', iaxis_177320, int_177321)
    
    # Processing the call keyword arguments (line 562)
    kwargs_177323 = {}
    # Getting the type of 'np' (line 562)
    np_177316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 562)
    rollaxis_177317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), np_177316, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 562)
    rollaxis_call_result_177324 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), rollaxis_177317, *[c_177318, int_177319, result_add_177322], **kwargs_177323)
    
    # Assigning a type to the variable 'c' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'c', rollaxis_call_result_177324)
    # Getting the type of 'c' (line 563)
    c_177325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type', c_177325)
    
    # ################# End of 'polyder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyder' in the type store
    # Getting the type of 'stypy_return_type' (line 477)
    stypy_return_type_177326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyder'
    return stypy_return_type_177326

# Assigning a type to the variable 'polyder' (line 477)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'polyder', polyder)

@norecursion
def polyint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_177327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 17), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 566)
    list_177328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 566)
    
    int_177329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 31), 'int')
    int_177330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 38), 'int')
    int_177331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 46), 'int')
    defaults = [int_177327, list_177328, int_177329, int_177330, int_177331]
    # Create a new context for function 'polyint'
    module_type_store = module_type_store.open_function_context('polyint', 566, 0, False)
    
    # Passed parameters checking function
    polyint.stypy_localization = localization
    polyint.stypy_type_of_self = None
    polyint.stypy_type_store = module_type_store
    polyint.stypy_function_name = 'polyint'
    polyint.stypy_param_names_list = ['c', 'm', 'k', 'lbnd', 'scl', 'axis']
    polyint.stypy_varargs_param_name = None
    polyint.stypy_kwargs_param_name = None
    polyint.stypy_call_defaults = defaults
    polyint.stypy_call_varargs = varargs
    polyint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyint', ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyint', localization, ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyint(...)' code ##################

    str_177332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, (-1)), 'str', '\n    Integrate a polynomial.\n\n    Returns the polynomial coefficients `c` integrated `m` times from\n    `lbnd` along `axis`.  At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.) The argument `c` is an array of\n    coefficients, from low to high degree along each axis, e.g., [1,2,3]\n    represents the polynomial ``1 + 2*x + 3*x**2`` while [[1,2],[1,2]]\n    represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is ``x`` and axis=1 is\n    ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of polynomial coefficients, ordered from low to high.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at zero\n        is the first value in the list, the value of the second integral\n        at zero is the second value, etc.  If ``k == []`` (the default),\n        all constants are set to zero.  If ``m == 1``, a single scalar can\n        be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        Coefficient array of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 1``, ``len(k) > m``.\n\n    See Also\n    --------\n    polyder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.  Why\n    is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`. Then\n    .. math::`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a` - perhaps not what one would have first thought.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> c = (1,2,3)\n    >>> P.polyint(c) # should return array([0, 1, 1, 1])\n    array([ 0.,  1.,  1.,  1.])\n    >>> P.polyint(c,3) # should return array([0, 0, 0, 1/6, 1/12, 1/20])\n    array([ 0.        ,  0.        ,  0.        ,  0.16666667,  0.08333333,\n            0.05      ])\n    >>> P.polyint(c,k=3) # should return array([3, 1, 1, 1])\n    array([ 3.,  1.,  1.,  1.])\n    >>> P.polyint(c,lbnd=-2) # should return array([6, 1, 1, 1])\n    array([ 6.,  1.,  1.,  1.])\n    >>> P.polyint(c,scl=-2) # should return array([0, -2, -2, -2])\n    array([ 0., -2., -2., -2.])\n\n    ')
    
    # Assigning a Call to a Name (line 643):
    
    # Assigning a Call to a Name (line 643):
    
    # Call to array(...): (line 643)
    # Processing the call arguments (line 643)
    # Getting the type of 'c' (line 643)
    c_177335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 17), 'c', False)
    # Processing the call keyword arguments (line 643)
    int_177336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 26), 'int')
    keyword_177337 = int_177336
    int_177338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 34), 'int')
    keyword_177339 = int_177338
    kwargs_177340 = {'copy': keyword_177339, 'ndmin': keyword_177337}
    # Getting the type of 'np' (line 643)
    np_177333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 643)
    array_177334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), np_177333, 'array')
    # Calling array(args, kwargs) (line 643)
    array_call_result_177341 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), array_177334, *[c_177335], **kwargs_177340)
    
    # Assigning a type to the variable 'c' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'c', array_call_result_177341)
    
    
    # Getting the type of 'c' (line 644)
    c_177342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 644)
    dtype_177343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 7), c_177342, 'dtype')
    # Obtaining the member 'char' of a type (line 644)
    char_177344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 7), dtype_177343, 'char')
    str_177345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 644)
    result_contains_177346 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 7), 'in', char_177344, str_177345)
    
    # Testing the type of an if condition (line 644)
    if_condition_177347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 4), result_contains_177346)
    # Assigning a type to the variable 'if_condition_177347' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'if_condition_177347', if_condition_177347)
    # SSA begins for if statement (line 644)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 646):
    
    # Assigning a BinOp to a Name (line 646):
    # Getting the type of 'c' (line 646)
    c_177348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'c')
    float_177349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 16), 'float')
    # Applying the binary operator '+' (line 646)
    result_add_177350 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 12), '+', c_177348, float_177349)
    
    # Assigning a type to the variable 'c' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'c', result_add_177350)
    # SSA join for if statement (line 644)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 647):
    
    # Assigning a Attribute to a Name (line 647):
    # Getting the type of 'c' (line 647)
    c_177351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 10), 'c')
    # Obtaining the member 'dtype' of a type (line 647)
    dtype_177352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 10), c_177351, 'dtype')
    # Assigning a type to the variable 'cdt' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'cdt', dtype_177352)
    
    
    
    # Call to iterable(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'k' (line 648)
    k_177355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 23), 'k', False)
    # Processing the call keyword arguments (line 648)
    kwargs_177356 = {}
    # Getting the type of 'np' (line 648)
    np_177353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 11), 'np', False)
    # Obtaining the member 'iterable' of a type (line 648)
    iterable_177354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 11), np_177353, 'iterable')
    # Calling iterable(args, kwargs) (line 648)
    iterable_call_result_177357 = invoke(stypy.reporting.localization.Localization(__file__, 648, 11), iterable_177354, *[k_177355], **kwargs_177356)
    
    # Applying the 'not' unary operator (line 648)
    result_not__177358 = python_operator(stypy.reporting.localization.Localization(__file__, 648, 7), 'not', iterable_call_result_177357)
    
    # Testing the type of an if condition (line 648)
    if_condition_177359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 648, 4), result_not__177358)
    # Assigning a type to the variable 'if_condition_177359' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'if_condition_177359', if_condition_177359)
    # SSA begins for if statement (line 648)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 649):
    
    # Assigning a List to a Name (line 649):
    
    # Obtaining an instance of the builtin type 'list' (line 649)
    list_177360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 649)
    # Adding element type (line 649)
    # Getting the type of 'k' (line 649)
    k_177361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 12), list_177360, k_177361)
    
    # Assigning a type to the variable 'k' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'k', list_177360)
    # SSA join for if statement (line 648)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 650):
    
    # Assigning a Subscript to a Name (line 650):
    
    # Obtaining the type of the subscript
    int_177362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 650)
    list_177367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 650)
    # Adding element type (line 650)
    # Getting the type of 'm' (line 650)
    m_177368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 34), list_177367, m_177368)
    # Adding element type (line 650)
    # Getting the type of 'axis' (line 650)
    axis_177369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 34), list_177367, axis_177369)
    
    comprehension_177370 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 18), list_177367)
    # Assigning a type to the variable 't' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 18), 't', comprehension_177370)
    
    # Call to int(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 't' (line 650)
    t_177364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 22), 't', False)
    # Processing the call keyword arguments (line 650)
    kwargs_177365 = {}
    # Getting the type of 'int' (line 650)
    int_177363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 18), 'int', False)
    # Calling int(args, kwargs) (line 650)
    int_call_result_177366 = invoke(stypy.reporting.localization.Localization(__file__, 650, 18), int_177363, *[t_177364], **kwargs_177365)
    
    list_177371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 18), list_177371, int_call_result_177366)
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___177372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 4), list_177371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_177373 = invoke(stypy.reporting.localization.Localization(__file__, 650, 4), getitem___177372, int_177362)
    
    # Assigning a type to the variable 'tuple_var_assignment_176503' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'tuple_var_assignment_176503', subscript_call_result_177373)
    
    # Assigning a Subscript to a Name (line 650):
    
    # Obtaining the type of the subscript
    int_177374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 650)
    list_177379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 650)
    # Adding element type (line 650)
    # Getting the type of 'm' (line 650)
    m_177380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 34), list_177379, m_177380)
    # Adding element type (line 650)
    # Getting the type of 'axis' (line 650)
    axis_177381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 34), list_177379, axis_177381)
    
    comprehension_177382 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 18), list_177379)
    # Assigning a type to the variable 't' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 18), 't', comprehension_177382)
    
    # Call to int(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 't' (line 650)
    t_177376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 22), 't', False)
    # Processing the call keyword arguments (line 650)
    kwargs_177377 = {}
    # Getting the type of 'int' (line 650)
    int_177375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 18), 'int', False)
    # Calling int(args, kwargs) (line 650)
    int_call_result_177378 = invoke(stypy.reporting.localization.Localization(__file__, 650, 18), int_177375, *[t_177376], **kwargs_177377)
    
    list_177383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 18), list_177383, int_call_result_177378)
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___177384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 4), list_177383, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_177385 = invoke(stypy.reporting.localization.Localization(__file__, 650, 4), getitem___177384, int_177374)
    
    # Assigning a type to the variable 'tuple_var_assignment_176504' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'tuple_var_assignment_176504', subscript_call_result_177385)
    
    # Assigning a Name to a Name (line 650):
    # Getting the type of 'tuple_var_assignment_176503' (line 650)
    tuple_var_assignment_176503_177386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'tuple_var_assignment_176503')
    # Assigning a type to the variable 'cnt' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'cnt', tuple_var_assignment_176503_177386)
    
    # Assigning a Name to a Name (line 650):
    # Getting the type of 'tuple_var_assignment_176504' (line 650)
    tuple_var_assignment_176504_177387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'tuple_var_assignment_176504')
    # Assigning a type to the variable 'iaxis' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 9), 'iaxis', tuple_var_assignment_176504_177387)
    
    
    # Getting the type of 'cnt' (line 652)
    cnt_177388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 7), 'cnt')
    # Getting the type of 'm' (line 652)
    m_177389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 14), 'm')
    # Applying the binary operator '!=' (line 652)
    result_ne_177390 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 7), '!=', cnt_177388, m_177389)
    
    # Testing the type of an if condition (line 652)
    if_condition_177391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 652, 4), result_ne_177390)
    # Assigning a type to the variable 'if_condition_177391' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'if_condition_177391', if_condition_177391)
    # SSA begins for if statement (line 652)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 653)
    # Processing the call arguments (line 653)
    str_177393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 25), 'str', 'The order of integration must be integer')
    # Processing the call keyword arguments (line 653)
    kwargs_177394 = {}
    # Getting the type of 'ValueError' (line 653)
    ValueError_177392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 653)
    ValueError_call_result_177395 = invoke(stypy.reporting.localization.Localization(__file__, 653, 14), ValueError_177392, *[str_177393], **kwargs_177394)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 653, 8), ValueError_call_result_177395, 'raise parameter', BaseException)
    # SSA join for if statement (line 652)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 654)
    cnt_177396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 7), 'cnt')
    int_177397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 13), 'int')
    # Applying the binary operator '<' (line 654)
    result_lt_177398 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 7), '<', cnt_177396, int_177397)
    
    # Testing the type of an if condition (line 654)
    if_condition_177399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 4), result_lt_177398)
    # Assigning a type to the variable 'if_condition_177399' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'if_condition_177399', if_condition_177399)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 655)
    # Processing the call arguments (line 655)
    str_177401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 25), 'str', 'The order of integration must be non-negative')
    # Processing the call keyword arguments (line 655)
    kwargs_177402 = {}
    # Getting the type of 'ValueError' (line 655)
    ValueError_177400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 655)
    ValueError_call_result_177403 = invoke(stypy.reporting.localization.Localization(__file__, 655, 14), ValueError_177400, *[str_177401], **kwargs_177402)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 655, 8), ValueError_call_result_177403, 'raise parameter', BaseException)
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'k' (line 656)
    k_177405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 11), 'k', False)
    # Processing the call keyword arguments (line 656)
    kwargs_177406 = {}
    # Getting the type of 'len' (line 656)
    len_177404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 7), 'len', False)
    # Calling len(args, kwargs) (line 656)
    len_call_result_177407 = invoke(stypy.reporting.localization.Localization(__file__, 656, 7), len_177404, *[k_177405], **kwargs_177406)
    
    # Getting the type of 'cnt' (line 656)
    cnt_177408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'cnt')
    # Applying the binary operator '>' (line 656)
    result_gt_177409 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 7), '>', len_call_result_177407, cnt_177408)
    
    # Testing the type of an if condition (line 656)
    if_condition_177410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 4), result_gt_177409)
    # Assigning a type to the variable 'if_condition_177410' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'if_condition_177410', if_condition_177410)
    # SSA begins for if statement (line 656)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 657)
    # Processing the call arguments (line 657)
    str_177412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 25), 'str', 'Too many integration constants')
    # Processing the call keyword arguments (line 657)
    kwargs_177413 = {}
    # Getting the type of 'ValueError' (line 657)
    ValueError_177411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 657)
    ValueError_call_result_177414 = invoke(stypy.reporting.localization.Localization(__file__, 657, 14), ValueError_177411, *[str_177412], **kwargs_177413)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 657, 8), ValueError_call_result_177414, 'raise parameter', BaseException)
    # SSA join for if statement (line 656)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 658)
    iaxis_177415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 7), 'iaxis')
    # Getting the type of 'axis' (line 658)
    axis_177416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'axis')
    # Applying the binary operator '!=' (line 658)
    result_ne_177417 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 7), '!=', iaxis_177415, axis_177416)
    
    # Testing the type of an if condition (line 658)
    if_condition_177418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 4), result_ne_177417)
    # Assigning a type to the variable 'if_condition_177418' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'if_condition_177418', if_condition_177418)
    # SSA begins for if statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 659)
    # Processing the call arguments (line 659)
    str_177420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 659)
    kwargs_177421 = {}
    # Getting the type of 'ValueError' (line 659)
    ValueError_177419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 659)
    ValueError_call_result_177422 = invoke(stypy.reporting.localization.Localization(__file__, 659, 14), ValueError_177419, *[str_177420], **kwargs_177421)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 659, 8), ValueError_call_result_177422, 'raise parameter', BaseException)
    # SSA join for if statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 660)
    c_177423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 660)
    ndim_177424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 12), c_177423, 'ndim')
    # Applying the 'usub' unary operator (line 660)
    result___neg___177425 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 11), 'usub', ndim_177424)
    
    # Getting the type of 'iaxis' (line 660)
    iaxis_177426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 22), 'iaxis')
    # Applying the binary operator '<=' (line 660)
    result_le_177427 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 11), '<=', result___neg___177425, iaxis_177426)
    # Getting the type of 'c' (line 660)
    c_177428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 660)
    ndim_177429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 30), c_177428, 'ndim')
    # Applying the binary operator '<' (line 660)
    result_lt_177430 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 11), '<', iaxis_177426, ndim_177429)
    # Applying the binary operator '&' (line 660)
    result_and__177431 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 11), '&', result_le_177427, result_lt_177430)
    
    # Applying the 'not' unary operator (line 660)
    result_not__177432 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 7), 'not', result_and__177431)
    
    # Testing the type of an if condition (line 660)
    if_condition_177433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 4), result_not__177432)
    # Assigning a type to the variable 'if_condition_177433' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'if_condition_177433', if_condition_177433)
    # SSA begins for if statement (line 660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 661)
    # Processing the call arguments (line 661)
    str_177435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 661)
    kwargs_177436 = {}
    # Getting the type of 'ValueError' (line 661)
    ValueError_177434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 661)
    ValueError_call_result_177437 = invoke(stypy.reporting.localization.Localization(__file__, 661, 14), ValueError_177434, *[str_177435], **kwargs_177436)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 661, 8), ValueError_call_result_177437, 'raise parameter', BaseException)
    # SSA join for if statement (line 660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 662)
    iaxis_177438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 7), 'iaxis')
    int_177439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 15), 'int')
    # Applying the binary operator '<' (line 662)
    result_lt_177440 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 7), '<', iaxis_177438, int_177439)
    
    # Testing the type of an if condition (line 662)
    if_condition_177441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 4), result_lt_177440)
    # Assigning a type to the variable 'if_condition_177441' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'if_condition_177441', if_condition_177441)
    # SSA begins for if statement (line 662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 663)
    iaxis_177442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'iaxis')
    # Getting the type of 'c' (line 663)
    c_177443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 663)
    ndim_177444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 17), c_177443, 'ndim')
    # Applying the binary operator '+=' (line 663)
    result_iadd_177445 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 8), '+=', iaxis_177442, ndim_177444)
    # Assigning a type to the variable 'iaxis' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'iaxis', result_iadd_177445)
    
    # SSA join for if statement (line 662)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 665)
    cnt_177446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 7), 'cnt')
    int_177447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 14), 'int')
    # Applying the binary operator '==' (line 665)
    result_eq_177448 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 7), '==', cnt_177446, int_177447)
    
    # Testing the type of an if condition (line 665)
    if_condition_177449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 665, 4), result_eq_177448)
    # Assigning a type to the variable 'if_condition_177449' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'if_condition_177449', if_condition_177449)
    # SSA begins for if statement (line 665)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 666)
    c_177450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'stypy_return_type', c_177450)
    # SSA join for if statement (line 665)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 668):
    
    # Assigning a BinOp to a Name (line 668):
    
    # Call to list(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'k' (line 668)
    k_177452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 13), 'k', False)
    # Processing the call keyword arguments (line 668)
    kwargs_177453 = {}
    # Getting the type of 'list' (line 668)
    list_177451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'list', False)
    # Calling list(args, kwargs) (line 668)
    list_call_result_177454 = invoke(stypy.reporting.localization.Localization(__file__, 668, 8), list_177451, *[k_177452], **kwargs_177453)
    
    
    # Obtaining an instance of the builtin type 'list' (line 668)
    list_177455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 668)
    # Adding element type (line 668)
    int_177456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 18), list_177455, int_177456)
    
    # Getting the type of 'cnt' (line 668)
    cnt_177457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'cnt')
    
    # Call to len(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'k' (line 668)
    k_177459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 33), 'k', False)
    # Processing the call keyword arguments (line 668)
    kwargs_177460 = {}
    # Getting the type of 'len' (line 668)
    len_177458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 29), 'len', False)
    # Calling len(args, kwargs) (line 668)
    len_call_result_177461 = invoke(stypy.reporting.localization.Localization(__file__, 668, 29), len_177458, *[k_177459], **kwargs_177460)
    
    # Applying the binary operator '-' (line 668)
    result_sub_177462 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 23), '-', cnt_177457, len_call_result_177461)
    
    # Applying the binary operator '*' (line 668)
    result_mul_177463 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 18), '*', list_177455, result_sub_177462)
    
    # Applying the binary operator '+' (line 668)
    result_add_177464 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 8), '+', list_call_result_177454, result_mul_177463)
    
    # Assigning a type to the variable 'k' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'k', result_add_177464)
    
    # Assigning a Call to a Name (line 669):
    
    # Assigning a Call to a Name (line 669):
    
    # Call to rollaxis(...): (line 669)
    # Processing the call arguments (line 669)
    # Getting the type of 'c' (line 669)
    c_177467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 20), 'c', False)
    # Getting the type of 'iaxis' (line 669)
    iaxis_177468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 669)
    kwargs_177469 = {}
    # Getting the type of 'np' (line 669)
    np_177465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 669)
    rollaxis_177466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 8), np_177465, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 669)
    rollaxis_call_result_177470 = invoke(stypy.reporting.localization.Localization(__file__, 669, 8), rollaxis_177466, *[c_177467, iaxis_177468], **kwargs_177469)
    
    # Assigning a type to the variable 'c' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'c', rollaxis_call_result_177470)
    
    
    # Call to range(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'cnt' (line 670)
    cnt_177472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 19), 'cnt', False)
    # Processing the call keyword arguments (line 670)
    kwargs_177473 = {}
    # Getting the type of 'range' (line 670)
    range_177471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 13), 'range', False)
    # Calling range(args, kwargs) (line 670)
    range_call_result_177474 = invoke(stypy.reporting.localization.Localization(__file__, 670, 13), range_177471, *[cnt_177472], **kwargs_177473)
    
    # Testing the type of a for loop iterable (line 670)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 670, 4), range_call_result_177474)
    # Getting the type of the for loop variable (line 670)
    for_loop_var_177475 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 670, 4), range_call_result_177474)
    # Assigning a type to the variable 'i' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'i', for_loop_var_177475)
    # SSA begins for a for statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 671):
    
    # Assigning a Call to a Name (line 671):
    
    # Call to len(...): (line 671)
    # Processing the call arguments (line 671)
    # Getting the type of 'c' (line 671)
    c_177477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'c', False)
    # Processing the call keyword arguments (line 671)
    kwargs_177478 = {}
    # Getting the type of 'len' (line 671)
    len_177476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'len', False)
    # Calling len(args, kwargs) (line 671)
    len_call_result_177479 = invoke(stypy.reporting.localization.Localization(__file__, 671, 12), len_177476, *[c_177477], **kwargs_177478)
    
    # Assigning a type to the variable 'n' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'n', len_call_result_177479)
    
    # Getting the type of 'c' (line 672)
    c_177480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'c')
    # Getting the type of 'scl' (line 672)
    scl_177481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 13), 'scl')
    # Applying the binary operator '*=' (line 672)
    result_imul_177482 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 8), '*=', c_177480, scl_177481)
    # Assigning a type to the variable 'c' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'c', result_imul_177482)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 673)
    n_177483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), 'n')
    int_177484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 16), 'int')
    # Applying the binary operator '==' (line 673)
    result_eq_177485 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 11), '==', n_177483, int_177484)
    
    
    # Call to all(...): (line 673)
    # Processing the call arguments (line 673)
    
    
    # Obtaining the type of the subscript
    int_177488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 31), 'int')
    # Getting the type of 'c' (line 673)
    c_177489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 673)
    getitem___177490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 29), c_177489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 673)
    subscript_call_result_177491 = invoke(stypy.reporting.localization.Localization(__file__, 673, 29), getitem___177490, int_177488)
    
    int_177492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 37), 'int')
    # Applying the binary operator '==' (line 673)
    result_eq_177493 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 29), '==', subscript_call_result_177491, int_177492)
    
    # Processing the call keyword arguments (line 673)
    kwargs_177494 = {}
    # Getting the type of 'np' (line 673)
    np_177486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 22), 'np', False)
    # Obtaining the member 'all' of a type (line 673)
    all_177487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 22), np_177486, 'all')
    # Calling all(args, kwargs) (line 673)
    all_call_result_177495 = invoke(stypy.reporting.localization.Localization(__file__, 673, 22), all_177487, *[result_eq_177493], **kwargs_177494)
    
    # Applying the binary operator 'and' (line 673)
    result_and_keyword_177496 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 11), 'and', result_eq_177485, all_call_result_177495)
    
    # Testing the type of an if condition (line 673)
    if_condition_177497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 673, 8), result_and_keyword_177496)
    # Assigning a type to the variable 'if_condition_177497' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'if_condition_177497', if_condition_177497)
    # SSA begins for if statement (line 673)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 674)
    c_177498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'c')
    
    # Obtaining the type of the subscript
    int_177499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 14), 'int')
    # Getting the type of 'c' (line 674)
    c_177500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 674)
    getitem___177501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 12), c_177500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 674)
    subscript_call_result_177502 = invoke(stypy.reporting.localization.Localization(__file__, 674, 12), getitem___177501, int_177499)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 674)
    i_177503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 22), 'i')
    # Getting the type of 'k' (line 674)
    k_177504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 20), 'k')
    # Obtaining the member '__getitem__' of a type (line 674)
    getitem___177505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 20), k_177504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 674)
    subscript_call_result_177506 = invoke(stypy.reporting.localization.Localization(__file__, 674, 20), getitem___177505, i_177503)
    
    # Applying the binary operator '+=' (line 674)
    result_iadd_177507 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 12), '+=', subscript_call_result_177502, subscript_call_result_177506)
    # Getting the type of 'c' (line 674)
    c_177508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'c')
    int_177509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 14), 'int')
    # Storing an element on a container (line 674)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 12), c_177508, (int_177509, result_iadd_177507))
    
    # SSA branch for the else part of an if statement (line 673)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 676):
    
    # Assigning a Call to a Name (line 676):
    
    # Call to empty(...): (line 676)
    # Processing the call arguments (line 676)
    
    # Obtaining an instance of the builtin type 'tuple' (line 676)
    tuple_177512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 676)
    # Adding element type (line 676)
    # Getting the type of 'n' (line 676)
    n_177513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 28), 'n', False)
    int_177514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 32), 'int')
    # Applying the binary operator '+' (line 676)
    result_add_177515 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 28), '+', n_177513, int_177514)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 28), tuple_177512, result_add_177515)
    
    
    # Obtaining the type of the subscript
    int_177516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 46), 'int')
    slice_177517 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 38), int_177516, None, None)
    # Getting the type of 'c' (line 676)
    c_177518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 676)
    shape_177519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 38), c_177518, 'shape')
    # Obtaining the member '__getitem__' of a type (line 676)
    getitem___177520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 38), shape_177519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 676)
    subscript_call_result_177521 = invoke(stypy.reporting.localization.Localization(__file__, 676, 38), getitem___177520, slice_177517)
    
    # Applying the binary operator '+' (line 676)
    result_add_177522 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 27), '+', tuple_177512, subscript_call_result_177521)
    
    # Processing the call keyword arguments (line 676)
    # Getting the type of 'cdt' (line 676)
    cdt_177523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 57), 'cdt', False)
    keyword_177524 = cdt_177523
    kwargs_177525 = {'dtype': keyword_177524}
    # Getting the type of 'np' (line 676)
    np_177510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 676)
    empty_177511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 18), np_177510, 'empty')
    # Calling empty(args, kwargs) (line 676)
    empty_call_result_177526 = invoke(stypy.reporting.localization.Localization(__file__, 676, 18), empty_177511, *[result_add_177522], **kwargs_177525)
    
    # Assigning a type to the variable 'tmp' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'tmp', empty_call_result_177526)
    
    # Assigning a BinOp to a Subscript (line 677):
    
    # Assigning a BinOp to a Subscript (line 677):
    
    # Obtaining the type of the subscript
    int_177527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 23), 'int')
    # Getting the type of 'c' (line 677)
    c_177528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___177529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 21), c_177528, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_177530 = invoke(stypy.reporting.localization.Localization(__file__, 677, 21), getitem___177529, int_177527)
    
    int_177531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 26), 'int')
    # Applying the binary operator '*' (line 677)
    result_mul_177532 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 21), '*', subscript_call_result_177530, int_177531)
    
    # Getting the type of 'tmp' (line 677)
    tmp_177533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'tmp')
    int_177534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 16), 'int')
    # Storing an element on a container (line 677)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 12), tmp_177533, (int_177534, result_mul_177532))
    
    # Assigning a Subscript to a Subscript (line 678):
    
    # Assigning a Subscript to a Subscript (line 678):
    
    # Obtaining the type of the subscript
    int_177535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 23), 'int')
    # Getting the type of 'c' (line 678)
    c_177536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___177537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 21), c_177536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_177538 = invoke(stypy.reporting.localization.Localization(__file__, 678, 21), getitem___177537, int_177535)
    
    # Getting the type of 'tmp' (line 678)
    tmp_177539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'tmp')
    int_177540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 16), 'int')
    # Storing an element on a container (line 678)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 12), tmp_177539, (int_177540, subscript_call_result_177538))
    
    
    # Call to range(...): (line 679)
    # Processing the call arguments (line 679)
    int_177542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 27), 'int')
    # Getting the type of 'n' (line 679)
    n_177543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 30), 'n', False)
    # Processing the call keyword arguments (line 679)
    kwargs_177544 = {}
    # Getting the type of 'range' (line 679)
    range_177541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 21), 'range', False)
    # Calling range(args, kwargs) (line 679)
    range_call_result_177545 = invoke(stypy.reporting.localization.Localization(__file__, 679, 21), range_177541, *[int_177542, n_177543], **kwargs_177544)
    
    # Testing the type of a for loop iterable (line 679)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 679, 12), range_call_result_177545)
    # Getting the type of the for loop variable (line 679)
    for_loop_var_177546 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 679, 12), range_call_result_177545)
    # Assigning a type to the variable 'j' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'j', for_loop_var_177546)
    # SSA begins for a for statement (line 679)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 680):
    
    # Assigning a BinOp to a Subscript (line 680):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 680)
    j_177547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 31), 'j')
    # Getting the type of 'c' (line 680)
    c_177548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 29), 'c')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___177549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 29), c_177548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_177550 = invoke(stypy.reporting.localization.Localization(__file__, 680, 29), getitem___177549, j_177547)
    
    # Getting the type of 'j' (line 680)
    j_177551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 35), 'j')
    int_177552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 39), 'int')
    # Applying the binary operator '+' (line 680)
    result_add_177553 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 35), '+', j_177551, int_177552)
    
    # Applying the binary operator 'div' (line 680)
    result_div_177554 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 29), 'div', subscript_call_result_177550, result_add_177553)
    
    # Getting the type of 'tmp' (line 680)
    tmp_177555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'tmp')
    # Getting the type of 'j' (line 680)
    j_177556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 20), 'j')
    int_177557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 24), 'int')
    # Applying the binary operator '+' (line 680)
    result_add_177558 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 20), '+', j_177556, int_177557)
    
    # Storing an element on a container (line 680)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 16), tmp_177555, (result_add_177558, result_div_177554))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tmp' (line 681)
    tmp_177559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), 'tmp')
    
    # Obtaining the type of the subscript
    int_177560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 16), 'int')
    # Getting the type of 'tmp' (line 681)
    tmp_177561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___177562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 12), tmp_177561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_177563 = invoke(stypy.reporting.localization.Localization(__file__, 681, 12), getitem___177562, int_177560)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 681)
    i_177564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 24), 'i')
    # Getting the type of 'k' (line 681)
    k_177565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 22), 'k')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___177566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 22), k_177565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_177567 = invoke(stypy.reporting.localization.Localization(__file__, 681, 22), getitem___177566, i_177564)
    
    
    # Call to polyval(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'lbnd' (line 681)
    lbnd_177569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 37), 'lbnd', False)
    # Getting the type of 'tmp' (line 681)
    tmp_177570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 43), 'tmp', False)
    # Processing the call keyword arguments (line 681)
    kwargs_177571 = {}
    # Getting the type of 'polyval' (line 681)
    polyval_177568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 29), 'polyval', False)
    # Calling polyval(args, kwargs) (line 681)
    polyval_call_result_177572 = invoke(stypy.reporting.localization.Localization(__file__, 681, 29), polyval_177568, *[lbnd_177569, tmp_177570], **kwargs_177571)
    
    # Applying the binary operator '-' (line 681)
    result_sub_177573 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 22), '-', subscript_call_result_177567, polyval_call_result_177572)
    
    # Applying the binary operator '+=' (line 681)
    result_iadd_177574 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 12), '+=', subscript_call_result_177563, result_sub_177573)
    # Getting the type of 'tmp' (line 681)
    tmp_177575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), 'tmp')
    int_177576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 16), 'int')
    # Storing an element on a container (line 681)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 12), tmp_177575, (int_177576, result_iadd_177574))
    
    
    # Assigning a Name to a Name (line 682):
    
    # Assigning a Name to a Name (line 682):
    # Getting the type of 'tmp' (line 682)
    tmp_177577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 16), 'tmp')
    # Assigning a type to the variable 'c' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'c', tmp_177577)
    # SSA join for if statement (line 673)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 683):
    
    # Assigning a Call to a Name (line 683):
    
    # Call to rollaxis(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'c' (line 683)
    c_177580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'c', False)
    int_177581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 23), 'int')
    # Getting the type of 'iaxis' (line 683)
    iaxis_177582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 26), 'iaxis', False)
    int_177583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 34), 'int')
    # Applying the binary operator '+' (line 683)
    result_add_177584 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 26), '+', iaxis_177582, int_177583)
    
    # Processing the call keyword arguments (line 683)
    kwargs_177585 = {}
    # Getting the type of 'np' (line 683)
    np_177578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 683)
    rollaxis_177579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 8), np_177578, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 683)
    rollaxis_call_result_177586 = invoke(stypy.reporting.localization.Localization(__file__, 683, 8), rollaxis_177579, *[c_177580, int_177581, result_add_177584], **kwargs_177585)
    
    # Assigning a type to the variable 'c' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'c', rollaxis_call_result_177586)
    # Getting the type of 'c' (line 684)
    c_177587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'stypy_return_type', c_177587)
    
    # ################# End of 'polyint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyint' in the type store
    # Getting the type of 'stypy_return_type' (line 566)
    stypy_return_type_177588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177588)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyint'
    return stypy_return_type_177588

# Assigning a type to the variable 'polyint' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'polyint', polyint)

@norecursion
def polyval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 687)
    True_177589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 25), 'True')
    defaults = [True_177589]
    # Create a new context for function 'polyval'
    module_type_store = module_type_store.open_function_context('polyval', 687, 0, False)
    
    # Passed parameters checking function
    polyval.stypy_localization = localization
    polyval.stypy_type_of_self = None
    polyval.stypy_type_store = module_type_store
    polyval.stypy_function_name = 'polyval'
    polyval.stypy_param_names_list = ['x', 'c', 'tensor']
    polyval.stypy_varargs_param_name = None
    polyval.stypy_kwargs_param_name = None
    polyval.stypy_call_defaults = defaults
    polyval.stypy_call_varargs = varargs
    polyval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyval', ['x', 'c', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyval', localization, ['x', 'c', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyval(...)' code ##################

    str_177590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, (-1)), 'str', "\n    Evaluate a polynomial at points x.\n\n    If `c` is of length `n + 1`, this function returns the value\n\n    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The shape of the returned array is described above.\n\n    See Also\n    --------\n    polyval2d, polygrid2d, polyval3d, polygrid3d\n\n    Notes\n    -----\n    The evaluation uses Horner's method.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.polynomial import polyval\n    >>> polyval(1, [1,2,3])\n    6.0\n    >>> a = np.arange(4).reshape(2,2)\n    >>> a\n    array([[0, 1],\n           [2, 3]])\n    >>> polyval(a, [1,2,3])\n    array([[  1.,   6.],\n           [ 17.,  34.]])\n    >>> coef = np.arange(4).reshape(2,2) # multidimensional coefficients\n    >>> coef\n    array([[0, 1],\n           [2, 3]])\n    >>> polyval([1,2], coef, tensor=True)\n    array([[ 2.,  4.],\n           [ 4.,  7.]])\n    >>> polyval([1,2], coef, tensor=False)\n    array([ 2.,  7.])\n\n    ")
    
    # Assigning a Call to a Name (line 768):
    
    # Assigning a Call to a Name (line 768):
    
    # Call to array(...): (line 768)
    # Processing the call arguments (line 768)
    # Getting the type of 'c' (line 768)
    c_177593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 17), 'c', False)
    # Processing the call keyword arguments (line 768)
    int_177594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 26), 'int')
    keyword_177595 = int_177594
    int_177596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 34), 'int')
    keyword_177597 = int_177596
    kwargs_177598 = {'copy': keyword_177597, 'ndmin': keyword_177595}
    # Getting the type of 'np' (line 768)
    np_177591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 768)
    array_177592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 8), np_177591, 'array')
    # Calling array(args, kwargs) (line 768)
    array_call_result_177599 = invoke(stypy.reporting.localization.Localization(__file__, 768, 8), array_177592, *[c_177593], **kwargs_177598)
    
    # Assigning a type to the variable 'c' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'c', array_call_result_177599)
    
    
    # Getting the type of 'c' (line 769)
    c_177600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 769)
    dtype_177601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 7), c_177600, 'dtype')
    # Obtaining the member 'char' of a type (line 769)
    char_177602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 7), dtype_177601, 'char')
    str_177603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 769)
    result_contains_177604 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 7), 'in', char_177602, str_177603)
    
    # Testing the type of an if condition (line 769)
    if_condition_177605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 769, 4), result_contains_177604)
    # Assigning a type to the variable 'if_condition_177605' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'if_condition_177605', if_condition_177605)
    # SSA begins for if statement (line 769)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 771):
    
    # Assigning a BinOp to a Name (line 771):
    # Getting the type of 'c' (line 771)
    c_177606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'c')
    float_177607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 16), 'float')
    # Applying the binary operator '+' (line 771)
    result_add_177608 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 12), '+', c_177606, float_177607)
    
    # Assigning a type to the variable 'c' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'c', result_add_177608)
    # SSA join for if statement (line 769)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 772)
    # Processing the call arguments (line 772)
    # Getting the type of 'x' (line 772)
    x_177610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 18), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 772)
    tuple_177611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 772)
    # Adding element type (line 772)
    # Getting the type of 'tuple' (line 772)
    tuple_177612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 22), tuple_177611, tuple_177612)
    # Adding element type (line 772)
    # Getting the type of 'list' (line 772)
    list_177613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 22), tuple_177611, list_177613)
    
    # Processing the call keyword arguments (line 772)
    kwargs_177614 = {}
    # Getting the type of 'isinstance' (line 772)
    isinstance_177609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 772)
    isinstance_call_result_177615 = invoke(stypy.reporting.localization.Localization(__file__, 772, 7), isinstance_177609, *[x_177610, tuple_177611], **kwargs_177614)
    
    # Testing the type of an if condition (line 772)
    if_condition_177616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 4), isinstance_call_result_177615)
    # Assigning a type to the variable 'if_condition_177616' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'if_condition_177616', if_condition_177616)
    # SSA begins for if statement (line 772)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 773):
    
    # Assigning a Call to a Name (line 773):
    
    # Call to asarray(...): (line 773)
    # Processing the call arguments (line 773)
    # Getting the type of 'x' (line 773)
    x_177619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 23), 'x', False)
    # Processing the call keyword arguments (line 773)
    kwargs_177620 = {}
    # Getting the type of 'np' (line 773)
    np_177617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 773)
    asarray_177618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 12), np_177617, 'asarray')
    # Calling asarray(args, kwargs) (line 773)
    asarray_call_result_177621 = invoke(stypy.reporting.localization.Localization(__file__, 773, 12), asarray_177618, *[x_177619], **kwargs_177620)
    
    # Assigning a type to the variable 'x' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'x', asarray_call_result_177621)
    # SSA join for if statement (line 772)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'x' (line 774)
    x_177623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 18), 'x', False)
    # Getting the type of 'np' (line 774)
    np_177624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 774)
    ndarray_177625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 21), np_177624, 'ndarray')
    # Processing the call keyword arguments (line 774)
    kwargs_177626 = {}
    # Getting the type of 'isinstance' (line 774)
    isinstance_177622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 774)
    isinstance_call_result_177627 = invoke(stypy.reporting.localization.Localization(__file__, 774, 7), isinstance_177622, *[x_177623, ndarray_177625], **kwargs_177626)
    
    # Getting the type of 'tensor' (line 774)
    tensor_177628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 37), 'tensor')
    # Applying the binary operator 'and' (line 774)
    result_and_keyword_177629 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 7), 'and', isinstance_call_result_177627, tensor_177628)
    
    # Testing the type of an if condition (line 774)
    if_condition_177630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 774, 4), result_and_keyword_177629)
    # Assigning a type to the variable 'if_condition_177630' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'if_condition_177630', if_condition_177630)
    # SSA begins for if statement (line 774)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 775):
    
    # Assigning a Call to a Name (line 775):
    
    # Call to reshape(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'c' (line 775)
    c_177633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 22), 'c', False)
    # Obtaining the member 'shape' of a type (line 775)
    shape_177634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 22), c_177633, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 775)
    tuple_177635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 775)
    # Adding element type (line 775)
    int_177636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 33), tuple_177635, int_177636)
    
    # Getting the type of 'x' (line 775)
    x_177637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 37), 'x', False)
    # Obtaining the member 'ndim' of a type (line 775)
    ndim_177638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 37), x_177637, 'ndim')
    # Applying the binary operator '*' (line 775)
    result_mul_177639 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 32), '*', tuple_177635, ndim_177638)
    
    # Applying the binary operator '+' (line 775)
    result_add_177640 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 22), '+', shape_177634, result_mul_177639)
    
    # Processing the call keyword arguments (line 775)
    kwargs_177641 = {}
    # Getting the type of 'c' (line 775)
    c_177631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 775)
    reshape_177632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 12), c_177631, 'reshape')
    # Calling reshape(args, kwargs) (line 775)
    reshape_call_result_177642 = invoke(stypy.reporting.localization.Localization(__file__, 775, 12), reshape_177632, *[result_add_177640], **kwargs_177641)
    
    # Assigning a type to the variable 'c' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'c', reshape_call_result_177642)
    # SSA join for if statement (line 774)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 777):
    
    # Assigning a BinOp to a Name (line 777):
    
    # Obtaining the type of the subscript
    int_177643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 11), 'int')
    # Getting the type of 'c' (line 777)
    c_177644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 9), 'c')
    # Obtaining the member '__getitem__' of a type (line 777)
    getitem___177645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 9), c_177644, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 777)
    subscript_call_result_177646 = invoke(stypy.reporting.localization.Localization(__file__, 777, 9), getitem___177645, int_177643)
    
    # Getting the type of 'x' (line 777)
    x_177647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 17), 'x')
    int_177648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 19), 'int')
    # Applying the binary operator '*' (line 777)
    result_mul_177649 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 17), '*', x_177647, int_177648)
    
    # Applying the binary operator '+' (line 777)
    result_add_177650 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 9), '+', subscript_call_result_177646, result_mul_177649)
    
    # Assigning a type to the variable 'c0' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'c0', result_add_177650)
    
    
    # Call to range(...): (line 778)
    # Processing the call arguments (line 778)
    int_177652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 19), 'int')
    
    # Call to len(...): (line 778)
    # Processing the call arguments (line 778)
    # Getting the type of 'c' (line 778)
    c_177654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 26), 'c', False)
    # Processing the call keyword arguments (line 778)
    kwargs_177655 = {}
    # Getting the type of 'len' (line 778)
    len_177653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 22), 'len', False)
    # Calling len(args, kwargs) (line 778)
    len_call_result_177656 = invoke(stypy.reporting.localization.Localization(__file__, 778, 22), len_177653, *[c_177654], **kwargs_177655)
    
    int_177657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 31), 'int')
    # Applying the binary operator '+' (line 778)
    result_add_177658 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 22), '+', len_call_result_177656, int_177657)
    
    # Processing the call keyword arguments (line 778)
    kwargs_177659 = {}
    # Getting the type of 'range' (line 778)
    range_177651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 13), 'range', False)
    # Calling range(args, kwargs) (line 778)
    range_call_result_177660 = invoke(stypy.reporting.localization.Localization(__file__, 778, 13), range_177651, *[int_177652, result_add_177658], **kwargs_177659)
    
    # Testing the type of a for loop iterable (line 778)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 778, 4), range_call_result_177660)
    # Getting the type of the for loop variable (line 778)
    for_loop_var_177661 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 778, 4), range_call_result_177660)
    # Assigning a type to the variable 'i' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'i', for_loop_var_177661)
    # SSA begins for a for statement (line 778)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 779):
    
    # Assigning a BinOp to a Name (line 779):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 779)
    i_177662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 16), 'i')
    # Applying the 'usub' unary operator (line 779)
    result___neg___177663 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 15), 'usub', i_177662)
    
    # Getting the type of 'c' (line 779)
    c_177664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 779)
    getitem___177665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 13), c_177664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 779)
    subscript_call_result_177666 = invoke(stypy.reporting.localization.Localization(__file__, 779, 13), getitem___177665, result___neg___177663)
    
    # Getting the type of 'c0' (line 779)
    c0_177667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 21), 'c0')
    # Getting the type of 'x' (line 779)
    x_177668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), 'x')
    # Applying the binary operator '*' (line 779)
    result_mul_177669 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 21), '*', c0_177667, x_177668)
    
    # Applying the binary operator '+' (line 779)
    result_add_177670 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 13), '+', subscript_call_result_177666, result_mul_177669)
    
    # Assigning a type to the variable 'c0' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'c0', result_add_177670)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 780)
    c0_177671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 11), 'c0')
    # Assigning a type to the variable 'stypy_return_type' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'stypy_return_type', c0_177671)
    
    # ################# End of 'polyval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyval' in the type store
    # Getting the type of 'stypy_return_type' (line 687)
    stypy_return_type_177672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177672)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyval'
    return stypy_return_type_177672

# Assigning a type to the variable 'polyval' (line 687)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 0), 'polyval', polyval)

@norecursion
def polyval2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyval2d'
    module_type_store = module_type_store.open_function_context('polyval2d', 783, 0, False)
    
    # Passed parameters checking function
    polyval2d.stypy_localization = localization
    polyval2d.stypy_type_of_self = None
    polyval2d.stypy_type_store = module_type_store
    polyval2d.stypy_function_name = 'polyval2d'
    polyval2d.stypy_param_names_list = ['x', 'y', 'c']
    polyval2d.stypy_varargs_param_name = None
    polyval2d.stypy_kwargs_param_name = None
    polyval2d.stypy_call_defaults = defaults
    polyval2d.stypy_call_varargs = varargs
    polyval2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyval2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyval2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyval2d(...)' code ##################

    str_177673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, (-1)), 'str', "\n    Evaluate a 2-D polynomial at points (x, y).\n\n    This function returns the value\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in `c[i,j]`. If `c` has\n        dimension greater than two the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points formed with\n        pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    polyval, polygrid2d, polyval3d, polygrid3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 831):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 831)
    # Processing the call arguments (line 831)
    
    # Obtaining an instance of the builtin type 'tuple' (line 831)
    tuple_177676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 831)
    # Adding element type (line 831)
    # Getting the type of 'x' (line 831)
    x_177677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 25), tuple_177676, x_177677)
    # Adding element type (line 831)
    # Getting the type of 'y' (line 831)
    y_177678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 28), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 25), tuple_177676, y_177678)
    
    # Processing the call keyword arguments (line 831)
    int_177679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 37), 'int')
    keyword_177680 = int_177679
    kwargs_177681 = {'copy': keyword_177680}
    # Getting the type of 'np' (line 831)
    np_177674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 831)
    array_177675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 15), np_177674, 'array')
    # Calling array(args, kwargs) (line 831)
    array_call_result_177682 = invoke(stypy.reporting.localization.Localization(__file__, 831, 15), array_177675, *[tuple_177676], **kwargs_177681)
    
    # Assigning a type to the variable 'call_assignment_176505' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176505', array_call_result_177682)
    
    # Assigning a Call to a Name (line 831):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_177685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 8), 'int')
    # Processing the call keyword arguments
    kwargs_177686 = {}
    # Getting the type of 'call_assignment_176505' (line 831)
    call_assignment_176505_177683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176505', False)
    # Obtaining the member '__getitem__' of a type (line 831)
    getitem___177684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 8), call_assignment_176505_177683, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_177687 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___177684, *[int_177685], **kwargs_177686)
    
    # Assigning a type to the variable 'call_assignment_176506' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176506', getitem___call_result_177687)
    
    # Assigning a Name to a Name (line 831):
    # Getting the type of 'call_assignment_176506' (line 831)
    call_assignment_176506_177688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176506')
    # Assigning a type to the variable 'x' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'x', call_assignment_176506_177688)
    
    # Assigning a Call to a Name (line 831):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_177691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 8), 'int')
    # Processing the call keyword arguments
    kwargs_177692 = {}
    # Getting the type of 'call_assignment_176505' (line 831)
    call_assignment_176505_177689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176505', False)
    # Obtaining the member '__getitem__' of a type (line 831)
    getitem___177690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 8), call_assignment_176505_177689, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_177693 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___177690, *[int_177691], **kwargs_177692)
    
    # Assigning a type to the variable 'call_assignment_176507' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176507', getitem___call_result_177693)
    
    # Assigning a Name to a Name (line 831):
    # Getting the type of 'call_assignment_176507' (line 831)
    call_assignment_176507_177694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'call_assignment_176507')
    # Assigning a type to the variable 'y' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 11), 'y', call_assignment_176507_177694)
    # SSA branch for the except part of a try statement (line 830)
    # SSA branch for the except '<any exception>' branch of a try statement (line 830)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 833)
    # Processing the call arguments (line 833)
    str_177696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 25), 'str', 'x, y are incompatible')
    # Processing the call keyword arguments (line 833)
    kwargs_177697 = {}
    # Getting the type of 'ValueError' (line 833)
    ValueError_177695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 833)
    ValueError_call_result_177698 = invoke(stypy.reporting.localization.Localization(__file__, 833, 14), ValueError_177695, *[str_177696], **kwargs_177697)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 833, 8), ValueError_call_result_177698, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 835):
    
    # Assigning a Call to a Name (line 835):
    
    # Call to polyval(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'x' (line 835)
    x_177700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 16), 'x', False)
    # Getting the type of 'c' (line 835)
    c_177701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 19), 'c', False)
    # Processing the call keyword arguments (line 835)
    kwargs_177702 = {}
    # Getting the type of 'polyval' (line 835)
    polyval_177699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 835)
    polyval_call_result_177703 = invoke(stypy.reporting.localization.Localization(__file__, 835, 8), polyval_177699, *[x_177700, c_177701], **kwargs_177702)
    
    # Assigning a type to the variable 'c' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 4), 'c', polyval_call_result_177703)
    
    # Assigning a Call to a Name (line 836):
    
    # Assigning a Call to a Name (line 836):
    
    # Call to polyval(...): (line 836)
    # Processing the call arguments (line 836)
    # Getting the type of 'y' (line 836)
    y_177705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'y', False)
    # Getting the type of 'c' (line 836)
    c_177706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 19), 'c', False)
    # Processing the call keyword arguments (line 836)
    # Getting the type of 'False' (line 836)
    False_177707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 29), 'False', False)
    keyword_177708 = False_177707
    kwargs_177709 = {'tensor': keyword_177708}
    # Getting the type of 'polyval' (line 836)
    polyval_177704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 836)
    polyval_call_result_177710 = invoke(stypy.reporting.localization.Localization(__file__, 836, 8), polyval_177704, *[y_177705, c_177706], **kwargs_177709)
    
    # Assigning a type to the variable 'c' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 4), 'c', polyval_call_result_177710)
    # Getting the type of 'c' (line 837)
    c_177711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'stypy_return_type', c_177711)
    
    # ################# End of 'polyval2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyval2d' in the type store
    # Getting the type of 'stypy_return_type' (line 783)
    stypy_return_type_177712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyval2d'
    return stypy_return_type_177712

# Assigning a type to the variable 'polyval2d' (line 783)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 0), 'polyval2d', polyval2d)

@norecursion
def polygrid2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polygrid2d'
    module_type_store = module_type_store.open_function_context('polygrid2d', 840, 0, False)
    
    # Passed parameters checking function
    polygrid2d.stypy_localization = localization
    polygrid2d.stypy_type_of_self = None
    polygrid2d.stypy_type_store = module_type_store
    polygrid2d.stypy_function_name = 'polygrid2d'
    polygrid2d.stypy_param_names_list = ['x', 'y', 'c']
    polygrid2d.stypy_varargs_param_name = None
    polygrid2d.stypy_kwargs_param_name = None
    polygrid2d.stypy_call_defaults = defaults
    polygrid2d.stypy_call_varargs = varargs
    polygrid2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polygrid2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polygrid2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polygrid2d(...)' code ##################

    str_177713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, (-1)), 'str', "\n    Evaluate a 2-D polynomial on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * a^i * b^j\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape + y.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    polyval, polyval2d, polyval3d, polygrid3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 890):
    
    # Assigning a Call to a Name (line 890):
    
    # Call to polyval(...): (line 890)
    # Processing the call arguments (line 890)
    # Getting the type of 'x' (line 890)
    x_177715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 16), 'x', False)
    # Getting the type of 'c' (line 890)
    c_177716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 19), 'c', False)
    # Processing the call keyword arguments (line 890)
    kwargs_177717 = {}
    # Getting the type of 'polyval' (line 890)
    polyval_177714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 890)
    polyval_call_result_177718 = invoke(stypy.reporting.localization.Localization(__file__, 890, 8), polyval_177714, *[x_177715, c_177716], **kwargs_177717)
    
    # Assigning a type to the variable 'c' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'c', polyval_call_result_177718)
    
    # Assigning a Call to a Name (line 891):
    
    # Assigning a Call to a Name (line 891):
    
    # Call to polyval(...): (line 891)
    # Processing the call arguments (line 891)
    # Getting the type of 'y' (line 891)
    y_177720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'y', False)
    # Getting the type of 'c' (line 891)
    c_177721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 19), 'c', False)
    # Processing the call keyword arguments (line 891)
    kwargs_177722 = {}
    # Getting the type of 'polyval' (line 891)
    polyval_177719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 891)
    polyval_call_result_177723 = invoke(stypy.reporting.localization.Localization(__file__, 891, 8), polyval_177719, *[y_177720, c_177721], **kwargs_177722)
    
    # Assigning a type to the variable 'c' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'c', polyval_call_result_177723)
    # Getting the type of 'c' (line 892)
    c_177724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 4), 'stypy_return_type', c_177724)
    
    # ################# End of 'polygrid2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polygrid2d' in the type store
    # Getting the type of 'stypy_return_type' (line 840)
    stypy_return_type_177725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177725)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polygrid2d'
    return stypy_return_type_177725

# Assigning a type to the variable 'polygrid2d' (line 840)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 0), 'polygrid2d', polygrid2d)

@norecursion
def polyval3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyval3d'
    module_type_store = module_type_store.open_function_context('polyval3d', 895, 0, False)
    
    # Passed parameters checking function
    polyval3d.stypy_localization = localization
    polyval3d.stypy_type_of_self = None
    polyval3d.stypy_type_store = module_type_store
    polyval3d.stypy_function_name = 'polyval3d'
    polyval3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    polyval3d.stypy_varargs_param_name = None
    polyval3d.stypy_kwargs_param_name = None
    polyval3d.stypy_call_defaults = defaults
    polyval3d.stypy_call_varargs = varargs
    polyval3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyval3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyval3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyval3d(...)' code ##################

    str_177726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, (-1)), 'str', "\n    Evaluate a 3-D polynomial at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * x^i * y^j * z^k\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimensional polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    polyval, polyval2d, polygrid2d, polygrid3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 943)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 944):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 944)
    # Processing the call arguments (line 944)
    
    # Obtaining an instance of the builtin type 'tuple' (line 944)
    tuple_177729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 944)
    # Adding element type (line 944)
    # Getting the type of 'x' (line 944)
    x_177730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 944, 28), tuple_177729, x_177730)
    # Adding element type (line 944)
    # Getting the type of 'y' (line 944)
    y_177731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 944, 28), tuple_177729, y_177731)
    # Adding element type (line 944)
    # Getting the type of 'z' (line 944)
    z_177732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 34), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 944, 28), tuple_177729, z_177732)
    
    # Processing the call keyword arguments (line 944)
    int_177733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 43), 'int')
    keyword_177734 = int_177733
    kwargs_177735 = {'copy': keyword_177734}
    # Getting the type of 'np' (line 944)
    np_177727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 944)
    array_177728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 18), np_177727, 'array')
    # Calling array(args, kwargs) (line 944)
    array_call_result_177736 = invoke(stypy.reporting.localization.Localization(__file__, 944, 18), array_177728, *[tuple_177729], **kwargs_177735)
    
    # Assigning a type to the variable 'call_assignment_176508' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176508', array_call_result_177736)
    
    # Assigning a Call to a Name (line 944):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_177739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 8), 'int')
    # Processing the call keyword arguments
    kwargs_177740 = {}
    # Getting the type of 'call_assignment_176508' (line 944)
    call_assignment_176508_177737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176508', False)
    # Obtaining the member '__getitem__' of a type (line 944)
    getitem___177738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 8), call_assignment_176508_177737, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_177741 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___177738, *[int_177739], **kwargs_177740)
    
    # Assigning a type to the variable 'call_assignment_176509' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176509', getitem___call_result_177741)
    
    # Assigning a Name to a Name (line 944):
    # Getting the type of 'call_assignment_176509' (line 944)
    call_assignment_176509_177742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176509')
    # Assigning a type to the variable 'x' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'x', call_assignment_176509_177742)
    
    # Assigning a Call to a Name (line 944):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_177745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 8), 'int')
    # Processing the call keyword arguments
    kwargs_177746 = {}
    # Getting the type of 'call_assignment_176508' (line 944)
    call_assignment_176508_177743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176508', False)
    # Obtaining the member '__getitem__' of a type (line 944)
    getitem___177744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 8), call_assignment_176508_177743, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_177747 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___177744, *[int_177745], **kwargs_177746)
    
    # Assigning a type to the variable 'call_assignment_176510' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176510', getitem___call_result_177747)
    
    # Assigning a Name to a Name (line 944):
    # Getting the type of 'call_assignment_176510' (line 944)
    call_assignment_176510_177748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176510')
    # Assigning a type to the variable 'y' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 11), 'y', call_assignment_176510_177748)
    
    # Assigning a Call to a Name (line 944):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_177751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 8), 'int')
    # Processing the call keyword arguments
    kwargs_177752 = {}
    # Getting the type of 'call_assignment_176508' (line 944)
    call_assignment_176508_177749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176508', False)
    # Obtaining the member '__getitem__' of a type (line 944)
    getitem___177750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 8), call_assignment_176508_177749, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_177753 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___177750, *[int_177751], **kwargs_177752)
    
    # Assigning a type to the variable 'call_assignment_176511' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176511', getitem___call_result_177753)
    
    # Assigning a Name to a Name (line 944):
    # Getting the type of 'call_assignment_176511' (line 944)
    call_assignment_176511_177754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'call_assignment_176511')
    # Assigning a type to the variable 'z' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 14), 'z', call_assignment_176511_177754)
    # SSA branch for the except part of a try statement (line 943)
    # SSA branch for the except '<any exception>' branch of a try statement (line 943)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 946)
    # Processing the call arguments (line 946)
    str_177756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 25), 'str', 'x, y, z are incompatible')
    # Processing the call keyword arguments (line 946)
    kwargs_177757 = {}
    # Getting the type of 'ValueError' (line 946)
    ValueError_177755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 946)
    ValueError_call_result_177758 = invoke(stypy.reporting.localization.Localization(__file__, 946, 14), ValueError_177755, *[str_177756], **kwargs_177757)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 946, 8), ValueError_call_result_177758, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 943)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 948):
    
    # Assigning a Call to a Name (line 948):
    
    # Call to polyval(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'x' (line 948)
    x_177760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 16), 'x', False)
    # Getting the type of 'c' (line 948)
    c_177761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 19), 'c', False)
    # Processing the call keyword arguments (line 948)
    kwargs_177762 = {}
    # Getting the type of 'polyval' (line 948)
    polyval_177759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 948)
    polyval_call_result_177763 = invoke(stypy.reporting.localization.Localization(__file__, 948, 8), polyval_177759, *[x_177760, c_177761], **kwargs_177762)
    
    # Assigning a type to the variable 'c' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'c', polyval_call_result_177763)
    
    # Assigning a Call to a Name (line 949):
    
    # Assigning a Call to a Name (line 949):
    
    # Call to polyval(...): (line 949)
    # Processing the call arguments (line 949)
    # Getting the type of 'y' (line 949)
    y_177765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 16), 'y', False)
    # Getting the type of 'c' (line 949)
    c_177766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 19), 'c', False)
    # Processing the call keyword arguments (line 949)
    # Getting the type of 'False' (line 949)
    False_177767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 29), 'False', False)
    keyword_177768 = False_177767
    kwargs_177769 = {'tensor': keyword_177768}
    # Getting the type of 'polyval' (line 949)
    polyval_177764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 949)
    polyval_call_result_177770 = invoke(stypy.reporting.localization.Localization(__file__, 949, 8), polyval_177764, *[y_177765, c_177766], **kwargs_177769)
    
    # Assigning a type to the variable 'c' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 4), 'c', polyval_call_result_177770)
    
    # Assigning a Call to a Name (line 950):
    
    # Assigning a Call to a Name (line 950):
    
    # Call to polyval(...): (line 950)
    # Processing the call arguments (line 950)
    # Getting the type of 'z' (line 950)
    z_177772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 16), 'z', False)
    # Getting the type of 'c' (line 950)
    c_177773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 19), 'c', False)
    # Processing the call keyword arguments (line 950)
    # Getting the type of 'False' (line 950)
    False_177774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 29), 'False', False)
    keyword_177775 = False_177774
    kwargs_177776 = {'tensor': keyword_177775}
    # Getting the type of 'polyval' (line 950)
    polyval_177771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 950)
    polyval_call_result_177777 = invoke(stypy.reporting.localization.Localization(__file__, 950, 8), polyval_177771, *[z_177772, c_177773], **kwargs_177776)
    
    # Assigning a type to the variable 'c' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'c', polyval_call_result_177777)
    # Getting the type of 'c' (line 951)
    c_177778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'stypy_return_type', c_177778)
    
    # ################# End of 'polyval3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyval3d' in the type store
    # Getting the type of 'stypy_return_type' (line 895)
    stypy_return_type_177779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyval3d'
    return stypy_return_type_177779

# Assigning a type to the variable 'polyval3d' (line 895)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 0), 'polyval3d', polyval3d)

@norecursion
def polygrid3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polygrid3d'
    module_type_store = module_type_store.open_function_context('polygrid3d', 954, 0, False)
    
    # Passed parameters checking function
    polygrid3d.stypy_localization = localization
    polygrid3d.stypy_type_of_self = None
    polygrid3d.stypy_type_store = module_type_store
    polygrid3d.stypy_function_name = 'polygrid3d'
    polygrid3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    polygrid3d.stypy_varargs_param_name = None
    polygrid3d.stypy_kwargs_param_name = None
    polygrid3d.stypy_call_defaults = defaults
    polygrid3d.stypy_call_varargs = varargs
    polygrid3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polygrid3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polygrid3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polygrid3d(...)' code ##################

    str_177780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, (-1)), 'str', "\n    Evaluate a 3-D polynomial on the Cartesian product of x, y and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * a^i * b^j * c^k\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    polyval, polyval2d, polygrid2d, polyval3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1007):
    
    # Assigning a Call to a Name (line 1007):
    
    # Call to polyval(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'x' (line 1007)
    x_177782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 16), 'x', False)
    # Getting the type of 'c' (line 1007)
    c_177783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 19), 'c', False)
    # Processing the call keyword arguments (line 1007)
    kwargs_177784 = {}
    # Getting the type of 'polyval' (line 1007)
    polyval_177781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 1007)
    polyval_call_result_177785 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 8), polyval_177781, *[x_177782, c_177783], **kwargs_177784)
    
    # Assigning a type to the variable 'c' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'c', polyval_call_result_177785)
    
    # Assigning a Call to a Name (line 1008):
    
    # Assigning a Call to a Name (line 1008):
    
    # Call to polyval(...): (line 1008)
    # Processing the call arguments (line 1008)
    # Getting the type of 'y' (line 1008)
    y_177787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'y', False)
    # Getting the type of 'c' (line 1008)
    c_177788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 19), 'c', False)
    # Processing the call keyword arguments (line 1008)
    kwargs_177789 = {}
    # Getting the type of 'polyval' (line 1008)
    polyval_177786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 1008)
    polyval_call_result_177790 = invoke(stypy.reporting.localization.Localization(__file__, 1008, 8), polyval_177786, *[y_177787, c_177788], **kwargs_177789)
    
    # Assigning a type to the variable 'c' (line 1008)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 4), 'c', polyval_call_result_177790)
    
    # Assigning a Call to a Name (line 1009):
    
    # Assigning a Call to a Name (line 1009):
    
    # Call to polyval(...): (line 1009)
    # Processing the call arguments (line 1009)
    # Getting the type of 'z' (line 1009)
    z_177792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 16), 'z', False)
    # Getting the type of 'c' (line 1009)
    c_177793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 19), 'c', False)
    # Processing the call keyword arguments (line 1009)
    kwargs_177794 = {}
    # Getting the type of 'polyval' (line 1009)
    polyval_177791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'polyval', False)
    # Calling polyval(args, kwargs) (line 1009)
    polyval_call_result_177795 = invoke(stypy.reporting.localization.Localization(__file__, 1009, 8), polyval_177791, *[z_177792, c_177793], **kwargs_177794)
    
    # Assigning a type to the variable 'c' (line 1009)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 4), 'c', polyval_call_result_177795)
    # Getting the type of 'c' (line 1010)
    c_177796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1010)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 4), 'stypy_return_type', c_177796)
    
    # ################# End of 'polygrid3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polygrid3d' in the type store
    # Getting the type of 'stypy_return_type' (line 954)
    stypy_return_type_177797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177797)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polygrid3d'
    return stypy_return_type_177797

# Assigning a type to the variable 'polygrid3d' (line 954)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 0), 'polygrid3d', polygrid3d)

@norecursion
def polyvander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyvander'
    module_type_store = module_type_store.open_function_context('polyvander', 1013, 0, False)
    
    # Passed parameters checking function
    polyvander.stypy_localization = localization
    polyvander.stypy_type_of_self = None
    polyvander.stypy_type_store = module_type_store
    polyvander.stypy_function_name = 'polyvander'
    polyvander.stypy_param_names_list = ['x', 'deg']
    polyvander.stypy_varargs_param_name = None
    polyvander.stypy_kwargs_param_name = None
    polyvander.stypy_call_defaults = defaults
    polyvander.stypy_call_varargs = varargs
    polyvander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyvander', ['x', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyvander', localization, ['x', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyvander(...)' code ##################

    str_177798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, (-1)), 'str', 'Vandermonde matrix of given degree.\n\n    Returns the Vandermonde matrix of degree `deg` and sample points\n    `x`. The Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = x^i,\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the power of `x`.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    matrix ``V = polyvander(x, n)``, then ``np.dot(V, c)`` and\n    ``polyval(x, c)`` are the same up to roundoff. This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of polynomials of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray.\n        The Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where the last index is the power of `x`.\n        The dtype will be the same as the converted `x`.\n\n    See Also\n    --------\n    polyvander2d, polyvander3d\n\n    ')
    
    # Assigning a Call to a Name (line 1051):
    
    # Assigning a Call to a Name (line 1051):
    
    # Call to int(...): (line 1051)
    # Processing the call arguments (line 1051)
    # Getting the type of 'deg' (line 1051)
    deg_177800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 15), 'deg', False)
    # Processing the call keyword arguments (line 1051)
    kwargs_177801 = {}
    # Getting the type of 'int' (line 1051)
    int_177799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 11), 'int', False)
    # Calling int(args, kwargs) (line 1051)
    int_call_result_177802 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 11), int_177799, *[deg_177800], **kwargs_177801)
    
    # Assigning a type to the variable 'ideg' (line 1051)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 4), 'ideg', int_call_result_177802)
    
    
    # Getting the type of 'ideg' (line 1052)
    ideg_177803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 7), 'ideg')
    # Getting the type of 'deg' (line 1052)
    deg_177804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 15), 'deg')
    # Applying the binary operator '!=' (line 1052)
    result_ne_177805 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 7), '!=', ideg_177803, deg_177804)
    
    # Testing the type of an if condition (line 1052)
    if_condition_177806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1052, 4), result_ne_177805)
    # Assigning a type to the variable 'if_condition_177806' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'if_condition_177806', if_condition_177806)
    # SSA begins for if statement (line 1052)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1053)
    # Processing the call arguments (line 1053)
    str_177808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 25), 'str', 'deg must be integer')
    # Processing the call keyword arguments (line 1053)
    kwargs_177809 = {}
    # Getting the type of 'ValueError' (line 1053)
    ValueError_177807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1053)
    ValueError_call_result_177810 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 14), ValueError_177807, *[str_177808], **kwargs_177809)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1053, 8), ValueError_call_result_177810, 'raise parameter', BaseException)
    # SSA join for if statement (line 1052)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ideg' (line 1054)
    ideg_177811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 7), 'ideg')
    int_177812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 14), 'int')
    # Applying the binary operator '<' (line 1054)
    result_lt_177813 = python_operator(stypy.reporting.localization.Localization(__file__, 1054, 7), '<', ideg_177811, int_177812)
    
    # Testing the type of an if condition (line 1054)
    if_condition_177814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1054, 4), result_lt_177813)
    # Assigning a type to the variable 'if_condition_177814' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 4), 'if_condition_177814', if_condition_177814)
    # SSA begins for if statement (line 1054)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1055)
    # Processing the call arguments (line 1055)
    str_177816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 25), 'str', 'deg must be non-negative')
    # Processing the call keyword arguments (line 1055)
    kwargs_177817 = {}
    # Getting the type of 'ValueError' (line 1055)
    ValueError_177815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1055)
    ValueError_call_result_177818 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 14), ValueError_177815, *[str_177816], **kwargs_177817)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1055, 8), ValueError_call_result_177818, 'raise parameter', BaseException)
    # SSA join for if statement (line 1054)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1057):
    
    # Assigning a BinOp to a Name (line 1057):
    
    # Call to array(...): (line 1057)
    # Processing the call arguments (line 1057)
    # Getting the type of 'x' (line 1057)
    x_177821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 17), 'x', False)
    # Processing the call keyword arguments (line 1057)
    int_177822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 25), 'int')
    keyword_177823 = int_177822
    int_177824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 34), 'int')
    keyword_177825 = int_177824
    kwargs_177826 = {'copy': keyword_177823, 'ndmin': keyword_177825}
    # Getting the type of 'np' (line 1057)
    np_177819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1057)
    array_177820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 8), np_177819, 'array')
    # Calling array(args, kwargs) (line 1057)
    array_call_result_177827 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 8), array_177820, *[x_177821], **kwargs_177826)
    
    float_177828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 39), 'float')
    # Applying the binary operator '+' (line 1057)
    result_add_177829 = python_operator(stypy.reporting.localization.Localization(__file__, 1057, 8), '+', array_call_result_177827, float_177828)
    
    # Assigning a type to the variable 'x' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'x', result_add_177829)
    
    # Assigning a BinOp to a Name (line 1058):
    
    # Assigning a BinOp to a Name (line 1058):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1058)
    tuple_177830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1058)
    # Adding element type (line 1058)
    # Getting the type of 'ideg' (line 1058)
    ideg_177831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 12), 'ideg')
    int_177832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 19), 'int')
    # Applying the binary operator '+' (line 1058)
    result_add_177833 = python_operator(stypy.reporting.localization.Localization(__file__, 1058, 12), '+', ideg_177831, int_177832)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1058, 12), tuple_177830, result_add_177833)
    
    # Getting the type of 'x' (line 1058)
    x_177834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 25), 'x')
    # Obtaining the member 'shape' of a type (line 1058)
    shape_177835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1058, 25), x_177834, 'shape')
    # Applying the binary operator '+' (line 1058)
    result_add_177836 = python_operator(stypy.reporting.localization.Localization(__file__, 1058, 11), '+', tuple_177830, shape_177835)
    
    # Assigning a type to the variable 'dims' (line 1058)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 4), 'dims', result_add_177836)
    
    # Assigning a Attribute to a Name (line 1059):
    
    # Assigning a Attribute to a Name (line 1059):
    # Getting the type of 'x' (line 1059)
    x_177837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 1059)
    dtype_177838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1059, 11), x_177837, 'dtype')
    # Assigning a type to the variable 'dtyp' (line 1059)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 4), 'dtyp', dtype_177838)
    
    # Assigning a Call to a Name (line 1060):
    
    # Assigning a Call to a Name (line 1060):
    
    # Call to empty(...): (line 1060)
    # Processing the call arguments (line 1060)
    # Getting the type of 'dims' (line 1060)
    dims_177841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 17), 'dims', False)
    # Processing the call keyword arguments (line 1060)
    # Getting the type of 'dtyp' (line 1060)
    dtyp_177842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 29), 'dtyp', False)
    keyword_177843 = dtyp_177842
    kwargs_177844 = {'dtype': keyword_177843}
    # Getting the type of 'np' (line 1060)
    np_177839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1060)
    empty_177840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 8), np_177839, 'empty')
    # Calling empty(args, kwargs) (line 1060)
    empty_call_result_177845 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 8), empty_177840, *[dims_177841], **kwargs_177844)
    
    # Assigning a type to the variable 'v' (line 1060)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'v', empty_call_result_177845)
    
    # Assigning a BinOp to a Subscript (line 1061):
    
    # Assigning a BinOp to a Subscript (line 1061):
    # Getting the type of 'x' (line 1061)
    x_177846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 11), 'x')
    int_177847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 13), 'int')
    # Applying the binary operator '*' (line 1061)
    result_mul_177848 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 11), '*', x_177846, int_177847)
    
    int_177849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 17), 'int')
    # Applying the binary operator '+' (line 1061)
    result_add_177850 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 11), '+', result_mul_177848, int_177849)
    
    # Getting the type of 'v' (line 1061)
    v_177851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 4), 'v')
    int_177852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 6), 'int')
    # Storing an element on a container (line 1061)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 4), v_177851, (int_177852, result_add_177850))
    
    
    # Getting the type of 'ideg' (line 1062)
    ideg_177853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 7), 'ideg')
    int_177854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 14), 'int')
    # Applying the binary operator '>' (line 1062)
    result_gt_177855 = python_operator(stypy.reporting.localization.Localization(__file__, 1062, 7), '>', ideg_177853, int_177854)
    
    # Testing the type of an if condition (line 1062)
    if_condition_177856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1062, 4), result_gt_177855)
    # Assigning a type to the variable 'if_condition_177856' (line 1062)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1062, 4), 'if_condition_177856', if_condition_177856)
    # SSA begins for if statement (line 1062)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 1063):
    
    # Assigning a Name to a Subscript (line 1063):
    # Getting the type of 'x' (line 1063)
    x_177857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 15), 'x')
    # Getting the type of 'v' (line 1063)
    v_177858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 8), 'v')
    int_177859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 10), 'int')
    # Storing an element on a container (line 1063)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1063, 8), v_177858, (int_177859, x_177857))
    
    
    # Call to range(...): (line 1064)
    # Processing the call arguments (line 1064)
    int_177861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 23), 'int')
    # Getting the type of 'ideg' (line 1064)
    ideg_177862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 26), 'ideg', False)
    int_177863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 33), 'int')
    # Applying the binary operator '+' (line 1064)
    result_add_177864 = python_operator(stypy.reporting.localization.Localization(__file__, 1064, 26), '+', ideg_177862, int_177863)
    
    # Processing the call keyword arguments (line 1064)
    kwargs_177865 = {}
    # Getting the type of 'range' (line 1064)
    range_177860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 17), 'range', False)
    # Calling range(args, kwargs) (line 1064)
    range_call_result_177866 = invoke(stypy.reporting.localization.Localization(__file__, 1064, 17), range_177860, *[int_177861, result_add_177864], **kwargs_177865)
    
    # Testing the type of a for loop iterable (line 1064)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1064, 8), range_call_result_177866)
    # Getting the type of the for loop variable (line 1064)
    for_loop_var_177867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1064, 8), range_call_result_177866)
    # Assigning a type to the variable 'i' (line 1064)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 8), 'i', for_loop_var_177867)
    # SSA begins for a for statement (line 1064)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 1065):
    
    # Assigning a BinOp to a Subscript (line 1065):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1065)
    i_177868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 21), 'i')
    int_177869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 23), 'int')
    # Applying the binary operator '-' (line 1065)
    result_sub_177870 = python_operator(stypy.reporting.localization.Localization(__file__, 1065, 21), '-', i_177868, int_177869)
    
    # Getting the type of 'v' (line 1065)
    v_177871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 19), 'v')
    # Obtaining the member '__getitem__' of a type (line 1065)
    getitem___177872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 19), v_177871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1065)
    subscript_call_result_177873 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 19), getitem___177872, result_sub_177870)
    
    # Getting the type of 'x' (line 1065)
    x_177874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 26), 'x')
    # Applying the binary operator '*' (line 1065)
    result_mul_177875 = python_operator(stypy.reporting.localization.Localization(__file__, 1065, 19), '*', subscript_call_result_177873, x_177874)
    
    # Getting the type of 'v' (line 1065)
    v_177876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 12), 'v')
    # Getting the type of 'i' (line 1065)
    i_177877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 14), 'i')
    # Storing an element on a container (line 1065)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1065, 12), v_177876, (i_177877, result_mul_177875))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1062)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rollaxis(...): (line 1066)
    # Processing the call arguments (line 1066)
    # Getting the type of 'v' (line 1066)
    v_177880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 23), 'v', False)
    int_177881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 26), 'int')
    # Getting the type of 'v' (line 1066)
    v_177882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 29), 'v', False)
    # Obtaining the member 'ndim' of a type (line 1066)
    ndim_177883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 29), v_177882, 'ndim')
    # Processing the call keyword arguments (line 1066)
    kwargs_177884 = {}
    # Getting the type of 'np' (line 1066)
    np_177878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 11), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1066)
    rollaxis_177879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 11), np_177878, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1066)
    rollaxis_call_result_177885 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 11), rollaxis_177879, *[v_177880, int_177881, ndim_177883], **kwargs_177884)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 4), 'stypy_return_type', rollaxis_call_result_177885)
    
    # ################# End of 'polyvander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyvander' in the type store
    # Getting the type of 'stypy_return_type' (line 1013)
    stypy_return_type_177886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyvander'
    return stypy_return_type_177886

# Assigning a type to the variable 'polyvander' (line 1013)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1013, 0), 'polyvander', polyvander)

@norecursion
def polyvander2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyvander2d'
    module_type_store = module_type_store.open_function_context('polyvander2d', 1069, 0, False)
    
    # Passed parameters checking function
    polyvander2d.stypy_localization = localization
    polyvander2d.stypy_type_of_self = None
    polyvander2d.stypy_type_store = module_type_store
    polyvander2d.stypy_function_name = 'polyvander2d'
    polyvander2d.stypy_param_names_list = ['x', 'y', 'deg']
    polyvander2d.stypy_varargs_param_name = None
    polyvander2d.stypy_kwargs_param_name = None
    polyvander2d.stypy_call_defaults = defaults
    polyvander2d.stypy_call_varargs = varargs
    polyvander2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyvander2d', ['x', 'y', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyvander2d', localization, ['x', 'y', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyvander2d(...)' code ##################

    str_177887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., deg[1]*i + j] = x^i * y^j,\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the powers of\n    `x` and `y`.\n\n    If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D polynomials\n    of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to\n        1-D arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    polyvander, polyvander3d. polyval2d, polyval3d\n\n    ')
    
    # Assigning a ListComp to a Name (line 1114):
    
    # Assigning a ListComp to a Name (line 1114):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1114)
    deg_177892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 28), 'deg')
    comprehension_177893 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1114, 12), deg_177892)
    # Assigning a type to the variable 'd' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 12), 'd', comprehension_177893)
    
    # Call to int(...): (line 1114)
    # Processing the call arguments (line 1114)
    # Getting the type of 'd' (line 1114)
    d_177889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 16), 'd', False)
    # Processing the call keyword arguments (line 1114)
    kwargs_177890 = {}
    # Getting the type of 'int' (line 1114)
    int_177888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 12), 'int', False)
    # Calling int(args, kwargs) (line 1114)
    int_call_result_177891 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 12), int_177888, *[d_177889], **kwargs_177890)
    
    list_177894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1114, 12), list_177894, int_call_result_177891)
    # Assigning a type to the variable 'ideg' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'ideg', list_177894)
    
    # Assigning a ListComp to a Name (line 1115):
    
    # Assigning a ListComp to a Name (line 1115):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1115)
    # Processing the call arguments (line 1115)
    # Getting the type of 'ideg' (line 1115)
    ideg_177903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1115)
    deg_177904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 59), 'deg', False)
    # Processing the call keyword arguments (line 1115)
    kwargs_177905 = {}
    # Getting the type of 'zip' (line 1115)
    zip_177902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1115)
    zip_call_result_177906 = invoke(stypy.reporting.localization.Localization(__file__, 1115, 49), zip_177902, *[ideg_177903, deg_177904], **kwargs_177905)
    
    comprehension_177907 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 16), zip_call_result_177906)
    # Assigning a type to the variable 'id' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 16), comprehension_177907))
    # Assigning a type to the variable 'd' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 16), comprehension_177907))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1115)
    id_177895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 16), 'id')
    # Getting the type of 'd' (line 1115)
    d_177896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 22), 'd')
    # Applying the binary operator '==' (line 1115)
    result_eq_177897 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 16), '==', id_177895, d_177896)
    
    
    # Getting the type of 'id' (line 1115)
    id_177898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 28), 'id')
    int_177899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 34), 'int')
    # Applying the binary operator '>=' (line 1115)
    result_ge_177900 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 28), '>=', id_177898, int_177899)
    
    # Applying the binary operator 'and' (line 1115)
    result_and_keyword_177901 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 16), 'and', result_eq_177897, result_ge_177900)
    
    list_177908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 16), list_177908, result_and_keyword_177901)
    # Assigning a type to the variable 'is_valid' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 4), 'is_valid', list_177908)
    
    
    # Getting the type of 'is_valid' (line 1116)
    is_valid_177909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1116)
    list_177910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1116)
    # Adding element type (line 1116)
    int_177911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 19), list_177910, int_177911)
    # Adding element type (line 1116)
    int_177912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 19), list_177910, int_177912)
    
    # Applying the binary operator '!=' (line 1116)
    result_ne_177913 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 7), '!=', is_valid_177909, list_177910)
    
    # Testing the type of an if condition (line 1116)
    if_condition_177914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1116, 4), result_ne_177913)
    # Assigning a type to the variable 'if_condition_177914' (line 1116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 4), 'if_condition_177914', if_condition_177914)
    # SSA begins for if statement (line 1116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1117)
    # Processing the call arguments (line 1117)
    str_177916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1117)
    kwargs_177917 = {}
    # Getting the type of 'ValueError' (line 1117)
    ValueError_177915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1117)
    ValueError_call_result_177918 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 14), ValueError_177915, *[str_177916], **kwargs_177917)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1117, 8), ValueError_call_result_177918, 'raise parameter', BaseException)
    # SSA join for if statement (line 1116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1118):
    
    # Assigning a Subscript to a Name (line 1118):
    
    # Obtaining the type of the subscript
    int_177919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 4), 'int')
    # Getting the type of 'ideg' (line 1118)
    ideg_177920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1118)
    getitem___177921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 4), ideg_177920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1118)
    subscript_call_result_177922 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 4), getitem___177921, int_177919)
    
    # Assigning a type to the variable 'tuple_var_assignment_176512' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'tuple_var_assignment_176512', subscript_call_result_177922)
    
    # Assigning a Subscript to a Name (line 1118):
    
    # Obtaining the type of the subscript
    int_177923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 4), 'int')
    # Getting the type of 'ideg' (line 1118)
    ideg_177924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1118)
    getitem___177925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 4), ideg_177924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1118)
    subscript_call_result_177926 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 4), getitem___177925, int_177923)
    
    # Assigning a type to the variable 'tuple_var_assignment_176513' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'tuple_var_assignment_176513', subscript_call_result_177926)
    
    # Assigning a Name to a Name (line 1118):
    # Getting the type of 'tuple_var_assignment_176512' (line 1118)
    tuple_var_assignment_176512_177927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'tuple_var_assignment_176512')
    # Assigning a type to the variable 'degx' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'degx', tuple_var_assignment_176512_177927)
    
    # Assigning a Name to a Name (line 1118):
    # Getting the type of 'tuple_var_assignment_176513' (line 1118)
    tuple_var_assignment_176513_177928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'tuple_var_assignment_176513')
    # Assigning a type to the variable 'degy' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 10), 'degy', tuple_var_assignment_176513_177928)
    
    # Assigning a BinOp to a Tuple (line 1119):
    
    # Assigning a Subscript to a Name (line 1119):
    
    # Obtaining the type of the subscript
    int_177929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 4), 'int')
    
    # Call to array(...): (line 1119)
    # Processing the call arguments (line 1119)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1119)
    tuple_177932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1119)
    # Adding element type (line 1119)
    # Getting the type of 'x' (line 1119)
    x_177933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1119, 21), tuple_177932, x_177933)
    # Adding element type (line 1119)
    # Getting the type of 'y' (line 1119)
    y_177934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1119, 21), tuple_177932, y_177934)
    
    # Processing the call keyword arguments (line 1119)
    int_177935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 33), 'int')
    keyword_177936 = int_177935
    kwargs_177937 = {'copy': keyword_177936}
    # Getting the type of 'np' (line 1119)
    np_177930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1119)
    array_177931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 11), np_177930, 'array')
    # Calling array(args, kwargs) (line 1119)
    array_call_result_177938 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 11), array_177931, *[tuple_177932], **kwargs_177937)
    
    float_177939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 38), 'float')
    # Applying the binary operator '+' (line 1119)
    result_add_177940 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 11), '+', array_call_result_177938, float_177939)
    
    # Obtaining the member '__getitem__' of a type (line 1119)
    getitem___177941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 4), result_add_177940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1119)
    subscript_call_result_177942 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 4), getitem___177941, int_177929)
    
    # Assigning a type to the variable 'tuple_var_assignment_176514' (line 1119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'tuple_var_assignment_176514', subscript_call_result_177942)
    
    # Assigning a Subscript to a Name (line 1119):
    
    # Obtaining the type of the subscript
    int_177943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 4), 'int')
    
    # Call to array(...): (line 1119)
    # Processing the call arguments (line 1119)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1119)
    tuple_177946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1119)
    # Adding element type (line 1119)
    # Getting the type of 'x' (line 1119)
    x_177947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1119, 21), tuple_177946, x_177947)
    # Adding element type (line 1119)
    # Getting the type of 'y' (line 1119)
    y_177948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1119, 21), tuple_177946, y_177948)
    
    # Processing the call keyword arguments (line 1119)
    int_177949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 33), 'int')
    keyword_177950 = int_177949
    kwargs_177951 = {'copy': keyword_177950}
    # Getting the type of 'np' (line 1119)
    np_177944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1119)
    array_177945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 11), np_177944, 'array')
    # Calling array(args, kwargs) (line 1119)
    array_call_result_177952 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 11), array_177945, *[tuple_177946], **kwargs_177951)
    
    float_177953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 38), 'float')
    # Applying the binary operator '+' (line 1119)
    result_add_177954 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 11), '+', array_call_result_177952, float_177953)
    
    # Obtaining the member '__getitem__' of a type (line 1119)
    getitem___177955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 4), result_add_177954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1119)
    subscript_call_result_177956 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 4), getitem___177955, int_177943)
    
    # Assigning a type to the variable 'tuple_var_assignment_176515' (line 1119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'tuple_var_assignment_176515', subscript_call_result_177956)
    
    # Assigning a Name to a Name (line 1119):
    # Getting the type of 'tuple_var_assignment_176514' (line 1119)
    tuple_var_assignment_176514_177957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'tuple_var_assignment_176514')
    # Assigning a type to the variable 'x' (line 1119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'x', tuple_var_assignment_176514_177957)
    
    # Assigning a Name to a Name (line 1119):
    # Getting the type of 'tuple_var_assignment_176515' (line 1119)
    tuple_var_assignment_176515_177958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'tuple_var_assignment_176515')
    # Assigning a type to the variable 'y' (line 1119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 7), 'y', tuple_var_assignment_176515_177958)
    
    # Assigning a Call to a Name (line 1121):
    
    # Assigning a Call to a Name (line 1121):
    
    # Call to polyvander(...): (line 1121)
    # Processing the call arguments (line 1121)
    # Getting the type of 'x' (line 1121)
    x_177960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 20), 'x', False)
    # Getting the type of 'degx' (line 1121)
    degx_177961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 23), 'degx', False)
    # Processing the call keyword arguments (line 1121)
    kwargs_177962 = {}
    # Getting the type of 'polyvander' (line 1121)
    polyvander_177959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 9), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1121)
    polyvander_call_result_177963 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 9), polyvander_177959, *[x_177960, degx_177961], **kwargs_177962)
    
    # Assigning a type to the variable 'vx' (line 1121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 4), 'vx', polyvander_call_result_177963)
    
    # Assigning a Call to a Name (line 1122):
    
    # Assigning a Call to a Name (line 1122):
    
    # Call to polyvander(...): (line 1122)
    # Processing the call arguments (line 1122)
    # Getting the type of 'y' (line 1122)
    y_177965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 20), 'y', False)
    # Getting the type of 'degy' (line 1122)
    degy_177966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 23), 'degy', False)
    # Processing the call keyword arguments (line 1122)
    kwargs_177967 = {}
    # Getting the type of 'polyvander' (line 1122)
    polyvander_177964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 9), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1122)
    polyvander_call_result_177968 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 9), polyvander_177964, *[y_177965, degy_177966], **kwargs_177967)
    
    # Assigning a type to the variable 'vy' (line 1122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 4), 'vy', polyvander_call_result_177968)
    
    # Assigning a BinOp to a Name (line 1123):
    
    # Assigning a BinOp to a Name (line 1123):
    
    # Obtaining the type of the subscript
    Ellipsis_177969 = Ellipsis
    # Getting the type of 'None' (line 1123)
    None_177970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 16), 'None')
    # Getting the type of 'vx' (line 1123)
    vx_177971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1123)
    getitem___177972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1123, 8), vx_177971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1123)
    subscript_call_result_177973 = invoke(stypy.reporting.localization.Localization(__file__, 1123, 8), getitem___177972, (Ellipsis_177969, None_177970))
    
    
    # Obtaining the type of the subscript
    Ellipsis_177974 = Ellipsis
    # Getting the type of 'None' (line 1123)
    None_177975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 30), 'None')
    slice_177976 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1123, 22), None, None, None)
    # Getting the type of 'vy' (line 1123)
    vy_177977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 22), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1123)
    getitem___177978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1123, 22), vy_177977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1123)
    subscript_call_result_177979 = invoke(stypy.reporting.localization.Localization(__file__, 1123, 22), getitem___177978, (Ellipsis_177974, None_177975, slice_177976))
    
    # Applying the binary operator '*' (line 1123)
    result_mul_177980 = python_operator(stypy.reporting.localization.Localization(__file__, 1123, 8), '*', subscript_call_result_177973, subscript_call_result_177979)
    
    # Assigning a type to the variable 'v' (line 1123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1123, 4), 'v', result_mul_177980)
    
    # Call to reshape(...): (line 1126)
    # Processing the call arguments (line 1126)
    
    # Obtaining the type of the subscript
    int_177983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 30), 'int')
    slice_177984 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1126, 21), None, int_177983, None)
    # Getting the type of 'v' (line 1126)
    v_177985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1126)
    shape_177986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 21), v_177985, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___177987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 21), shape_177986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_177988 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 21), getitem___177987, slice_177984)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1126)
    tuple_177989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1126)
    # Adding element type (line 1126)
    int_177990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1126, 37), tuple_177989, int_177990)
    
    # Applying the binary operator '+' (line 1126)
    result_add_177991 = python_operator(stypy.reporting.localization.Localization(__file__, 1126, 21), '+', subscript_call_result_177988, tuple_177989)
    
    # Processing the call keyword arguments (line 1126)
    kwargs_177992 = {}
    # Getting the type of 'v' (line 1126)
    v_177981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1126)
    reshape_177982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 11), v_177981, 'reshape')
    # Calling reshape(args, kwargs) (line 1126)
    reshape_call_result_177993 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 11), reshape_177982, *[result_add_177991], **kwargs_177992)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'stypy_return_type', reshape_call_result_177993)
    
    # ################# End of 'polyvander2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyvander2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1069)
    stypy_return_type_177994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177994)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyvander2d'
    return stypy_return_type_177994

# Assigning a type to the variable 'polyvander2d' (line 1069)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 0), 'polyvander2d', polyvander2d)

@norecursion
def polyvander3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyvander3d'
    module_type_store = module_type_store.open_function_context('polyvander3d', 1129, 0, False)
    
    # Passed parameters checking function
    polyvander3d.stypy_localization = localization
    polyvander3d.stypy_type_of_self = None
    polyvander3d.stypy_type_store = module_type_store
    polyvander3d.stypy_function_name = 'polyvander3d'
    polyvander3d.stypy_param_names_list = ['x', 'y', 'z', 'deg']
    polyvander3d.stypy_varargs_param_name = None
    polyvander3d.stypy_kwargs_param_name = None
    polyvander3d.stypy_call_defaults = defaults
    polyvander3d.stypy_call_varargs = varargs
    polyvander3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyvander3d', ['x', 'y', 'z', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyvander3d', localization, ['x', 'y', 'z', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyvander3d(...)' code ##################

    str_177995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1179, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the powers of `x`, `y`, and `z`.\n\n    If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D polynomials\n    of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    polyvander, polyvander3d. polyval2d, polyval3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1180):
    
    # Assigning a ListComp to a Name (line 1180):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1180)
    deg_178000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 28), 'deg')
    comprehension_178001 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1180, 12), deg_178000)
    # Assigning a type to the variable 'd' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 12), 'd', comprehension_178001)
    
    # Call to int(...): (line 1180)
    # Processing the call arguments (line 1180)
    # Getting the type of 'd' (line 1180)
    d_177997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 16), 'd', False)
    # Processing the call keyword arguments (line 1180)
    kwargs_177998 = {}
    # Getting the type of 'int' (line 1180)
    int_177996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 12), 'int', False)
    # Calling int(args, kwargs) (line 1180)
    int_call_result_177999 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 12), int_177996, *[d_177997], **kwargs_177998)
    
    list_178002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1180, 12), list_178002, int_call_result_177999)
    # Assigning a type to the variable 'ideg' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'ideg', list_178002)
    
    # Assigning a ListComp to a Name (line 1181):
    
    # Assigning a ListComp to a Name (line 1181):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1181)
    # Processing the call arguments (line 1181)
    # Getting the type of 'ideg' (line 1181)
    ideg_178011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1181)
    deg_178012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 59), 'deg', False)
    # Processing the call keyword arguments (line 1181)
    kwargs_178013 = {}
    # Getting the type of 'zip' (line 1181)
    zip_178010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1181)
    zip_call_result_178014 = invoke(stypy.reporting.localization.Localization(__file__, 1181, 49), zip_178010, *[ideg_178011, deg_178012], **kwargs_178013)
    
    comprehension_178015 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1181, 16), zip_call_result_178014)
    # Assigning a type to the variable 'id' (line 1181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1181, 16), comprehension_178015))
    # Assigning a type to the variable 'd' (line 1181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1181, 16), comprehension_178015))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1181)
    id_178003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 16), 'id')
    # Getting the type of 'd' (line 1181)
    d_178004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 22), 'd')
    # Applying the binary operator '==' (line 1181)
    result_eq_178005 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 16), '==', id_178003, d_178004)
    
    
    # Getting the type of 'id' (line 1181)
    id_178006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 28), 'id')
    int_178007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, 34), 'int')
    # Applying the binary operator '>=' (line 1181)
    result_ge_178008 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 28), '>=', id_178006, int_178007)
    
    # Applying the binary operator 'and' (line 1181)
    result_and_keyword_178009 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 16), 'and', result_eq_178005, result_ge_178008)
    
    list_178016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1181, 16), list_178016, result_and_keyword_178009)
    # Assigning a type to the variable 'is_valid' (line 1181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 4), 'is_valid', list_178016)
    
    
    # Getting the type of 'is_valid' (line 1182)
    is_valid_178017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1182)
    list_178018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1182)
    # Adding element type (line 1182)
    int_178019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 19), list_178018, int_178019)
    # Adding element type (line 1182)
    int_178020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 19), list_178018, int_178020)
    # Adding element type (line 1182)
    int_178021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 19), list_178018, int_178021)
    
    # Applying the binary operator '!=' (line 1182)
    result_ne_178022 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 7), '!=', is_valid_178017, list_178018)
    
    # Testing the type of an if condition (line 1182)
    if_condition_178023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1182, 4), result_ne_178022)
    # Assigning a type to the variable 'if_condition_178023' (line 1182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 4), 'if_condition_178023', if_condition_178023)
    # SSA begins for if statement (line 1182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1183)
    # Processing the call arguments (line 1183)
    str_178025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1183)
    kwargs_178026 = {}
    # Getting the type of 'ValueError' (line 1183)
    ValueError_178024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1183)
    ValueError_call_result_178027 = invoke(stypy.reporting.localization.Localization(__file__, 1183, 14), ValueError_178024, *[str_178025], **kwargs_178026)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1183, 8), ValueError_call_result_178027, 'raise parameter', BaseException)
    # SSA join for if statement (line 1182)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1184):
    
    # Assigning a Subscript to a Name (line 1184):
    
    # Obtaining the type of the subscript
    int_178028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1184, 4), 'int')
    # Getting the type of 'ideg' (line 1184)
    ideg_178029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1184)
    getitem___178030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1184, 4), ideg_178029, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1184)
    subscript_call_result_178031 = invoke(stypy.reporting.localization.Localization(__file__, 1184, 4), getitem___178030, int_178028)
    
    # Assigning a type to the variable 'tuple_var_assignment_176516' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'tuple_var_assignment_176516', subscript_call_result_178031)
    
    # Assigning a Subscript to a Name (line 1184):
    
    # Obtaining the type of the subscript
    int_178032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1184, 4), 'int')
    # Getting the type of 'ideg' (line 1184)
    ideg_178033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1184)
    getitem___178034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1184, 4), ideg_178033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1184)
    subscript_call_result_178035 = invoke(stypy.reporting.localization.Localization(__file__, 1184, 4), getitem___178034, int_178032)
    
    # Assigning a type to the variable 'tuple_var_assignment_176517' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'tuple_var_assignment_176517', subscript_call_result_178035)
    
    # Assigning a Subscript to a Name (line 1184):
    
    # Obtaining the type of the subscript
    int_178036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1184, 4), 'int')
    # Getting the type of 'ideg' (line 1184)
    ideg_178037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1184)
    getitem___178038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1184, 4), ideg_178037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1184)
    subscript_call_result_178039 = invoke(stypy.reporting.localization.Localization(__file__, 1184, 4), getitem___178038, int_178036)
    
    # Assigning a type to the variable 'tuple_var_assignment_176518' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'tuple_var_assignment_176518', subscript_call_result_178039)
    
    # Assigning a Name to a Name (line 1184):
    # Getting the type of 'tuple_var_assignment_176516' (line 1184)
    tuple_var_assignment_176516_178040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'tuple_var_assignment_176516')
    # Assigning a type to the variable 'degx' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'degx', tuple_var_assignment_176516_178040)
    
    # Assigning a Name to a Name (line 1184):
    # Getting the type of 'tuple_var_assignment_176517' (line 1184)
    tuple_var_assignment_176517_178041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'tuple_var_assignment_176517')
    # Assigning a type to the variable 'degy' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 10), 'degy', tuple_var_assignment_176517_178041)
    
    # Assigning a Name to a Name (line 1184):
    # Getting the type of 'tuple_var_assignment_176518' (line 1184)
    tuple_var_assignment_176518_178042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'tuple_var_assignment_176518')
    # Assigning a type to the variable 'degz' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 16), 'degz', tuple_var_assignment_176518_178042)
    
    # Assigning a BinOp to a Tuple (line 1185):
    
    # Assigning a Subscript to a Name (line 1185):
    
    # Obtaining the type of the subscript
    int_178043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 4), 'int')
    
    # Call to array(...): (line 1185)
    # Processing the call arguments (line 1185)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1185)
    tuple_178046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1185)
    # Adding element type (line 1185)
    # Getting the type of 'x' (line 1185)
    x_178047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178046, x_178047)
    # Adding element type (line 1185)
    # Getting the type of 'y' (line 1185)
    y_178048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178046, y_178048)
    # Adding element type (line 1185)
    # Getting the type of 'z' (line 1185)
    z_178049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178046, z_178049)
    
    # Processing the call keyword arguments (line 1185)
    int_178050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 39), 'int')
    keyword_178051 = int_178050
    kwargs_178052 = {'copy': keyword_178051}
    # Getting the type of 'np' (line 1185)
    np_178044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1185)
    array_178045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 14), np_178044, 'array')
    # Calling array(args, kwargs) (line 1185)
    array_call_result_178053 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 14), array_178045, *[tuple_178046], **kwargs_178052)
    
    float_178054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 44), 'float')
    # Applying the binary operator '+' (line 1185)
    result_add_178055 = python_operator(stypy.reporting.localization.Localization(__file__, 1185, 14), '+', array_call_result_178053, float_178054)
    
    # Obtaining the member '__getitem__' of a type (line 1185)
    getitem___178056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 4), result_add_178055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1185)
    subscript_call_result_178057 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 4), getitem___178056, int_178043)
    
    # Assigning a type to the variable 'tuple_var_assignment_176519' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'tuple_var_assignment_176519', subscript_call_result_178057)
    
    # Assigning a Subscript to a Name (line 1185):
    
    # Obtaining the type of the subscript
    int_178058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 4), 'int')
    
    # Call to array(...): (line 1185)
    # Processing the call arguments (line 1185)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1185)
    tuple_178061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1185)
    # Adding element type (line 1185)
    # Getting the type of 'x' (line 1185)
    x_178062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178061, x_178062)
    # Adding element type (line 1185)
    # Getting the type of 'y' (line 1185)
    y_178063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178061, y_178063)
    # Adding element type (line 1185)
    # Getting the type of 'z' (line 1185)
    z_178064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178061, z_178064)
    
    # Processing the call keyword arguments (line 1185)
    int_178065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 39), 'int')
    keyword_178066 = int_178065
    kwargs_178067 = {'copy': keyword_178066}
    # Getting the type of 'np' (line 1185)
    np_178059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1185)
    array_178060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 14), np_178059, 'array')
    # Calling array(args, kwargs) (line 1185)
    array_call_result_178068 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 14), array_178060, *[tuple_178061], **kwargs_178067)
    
    float_178069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 44), 'float')
    # Applying the binary operator '+' (line 1185)
    result_add_178070 = python_operator(stypy.reporting.localization.Localization(__file__, 1185, 14), '+', array_call_result_178068, float_178069)
    
    # Obtaining the member '__getitem__' of a type (line 1185)
    getitem___178071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 4), result_add_178070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1185)
    subscript_call_result_178072 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 4), getitem___178071, int_178058)
    
    # Assigning a type to the variable 'tuple_var_assignment_176520' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'tuple_var_assignment_176520', subscript_call_result_178072)
    
    # Assigning a Subscript to a Name (line 1185):
    
    # Obtaining the type of the subscript
    int_178073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 4), 'int')
    
    # Call to array(...): (line 1185)
    # Processing the call arguments (line 1185)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1185)
    tuple_178076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1185)
    # Adding element type (line 1185)
    # Getting the type of 'x' (line 1185)
    x_178077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178076, x_178077)
    # Adding element type (line 1185)
    # Getting the type of 'y' (line 1185)
    y_178078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178076, y_178078)
    # Adding element type (line 1185)
    # Getting the type of 'z' (line 1185)
    z_178079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 24), tuple_178076, z_178079)
    
    # Processing the call keyword arguments (line 1185)
    int_178080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 39), 'int')
    keyword_178081 = int_178080
    kwargs_178082 = {'copy': keyword_178081}
    # Getting the type of 'np' (line 1185)
    np_178074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1185)
    array_178075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 14), np_178074, 'array')
    # Calling array(args, kwargs) (line 1185)
    array_call_result_178083 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 14), array_178075, *[tuple_178076], **kwargs_178082)
    
    float_178084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 44), 'float')
    # Applying the binary operator '+' (line 1185)
    result_add_178085 = python_operator(stypy.reporting.localization.Localization(__file__, 1185, 14), '+', array_call_result_178083, float_178084)
    
    # Obtaining the member '__getitem__' of a type (line 1185)
    getitem___178086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 4), result_add_178085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1185)
    subscript_call_result_178087 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 4), getitem___178086, int_178073)
    
    # Assigning a type to the variable 'tuple_var_assignment_176521' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'tuple_var_assignment_176521', subscript_call_result_178087)
    
    # Assigning a Name to a Name (line 1185):
    # Getting the type of 'tuple_var_assignment_176519' (line 1185)
    tuple_var_assignment_176519_178088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'tuple_var_assignment_176519')
    # Assigning a type to the variable 'x' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'x', tuple_var_assignment_176519_178088)
    
    # Assigning a Name to a Name (line 1185):
    # Getting the type of 'tuple_var_assignment_176520' (line 1185)
    tuple_var_assignment_176520_178089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'tuple_var_assignment_176520')
    # Assigning a type to the variable 'y' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 7), 'y', tuple_var_assignment_176520_178089)
    
    # Assigning a Name to a Name (line 1185):
    # Getting the type of 'tuple_var_assignment_176521' (line 1185)
    tuple_var_assignment_176521_178090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'tuple_var_assignment_176521')
    # Assigning a type to the variable 'z' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 10), 'z', tuple_var_assignment_176521_178090)
    
    # Assigning a Call to a Name (line 1187):
    
    # Assigning a Call to a Name (line 1187):
    
    # Call to polyvander(...): (line 1187)
    # Processing the call arguments (line 1187)
    # Getting the type of 'x' (line 1187)
    x_178092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 20), 'x', False)
    # Getting the type of 'degx' (line 1187)
    degx_178093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 23), 'degx', False)
    # Processing the call keyword arguments (line 1187)
    kwargs_178094 = {}
    # Getting the type of 'polyvander' (line 1187)
    polyvander_178091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 9), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1187)
    polyvander_call_result_178095 = invoke(stypy.reporting.localization.Localization(__file__, 1187, 9), polyvander_178091, *[x_178092, degx_178093], **kwargs_178094)
    
    # Assigning a type to the variable 'vx' (line 1187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1187, 4), 'vx', polyvander_call_result_178095)
    
    # Assigning a Call to a Name (line 1188):
    
    # Assigning a Call to a Name (line 1188):
    
    # Call to polyvander(...): (line 1188)
    # Processing the call arguments (line 1188)
    # Getting the type of 'y' (line 1188)
    y_178097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 20), 'y', False)
    # Getting the type of 'degy' (line 1188)
    degy_178098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 23), 'degy', False)
    # Processing the call keyword arguments (line 1188)
    kwargs_178099 = {}
    # Getting the type of 'polyvander' (line 1188)
    polyvander_178096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 9), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1188)
    polyvander_call_result_178100 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 9), polyvander_178096, *[y_178097, degy_178098], **kwargs_178099)
    
    # Assigning a type to the variable 'vy' (line 1188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 4), 'vy', polyvander_call_result_178100)
    
    # Assigning a Call to a Name (line 1189):
    
    # Assigning a Call to a Name (line 1189):
    
    # Call to polyvander(...): (line 1189)
    # Processing the call arguments (line 1189)
    # Getting the type of 'z' (line 1189)
    z_178102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 20), 'z', False)
    # Getting the type of 'degz' (line 1189)
    degz_178103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 23), 'degz', False)
    # Processing the call keyword arguments (line 1189)
    kwargs_178104 = {}
    # Getting the type of 'polyvander' (line 1189)
    polyvander_178101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 9), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1189)
    polyvander_call_result_178105 = invoke(stypy.reporting.localization.Localization(__file__, 1189, 9), polyvander_178101, *[z_178102, degz_178103], **kwargs_178104)
    
    # Assigning a type to the variable 'vz' (line 1189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 4), 'vz', polyvander_call_result_178105)
    
    # Assigning a BinOp to a Name (line 1190):
    
    # Assigning a BinOp to a Name (line 1190):
    
    # Obtaining the type of the subscript
    Ellipsis_178106 = Ellipsis
    # Getting the type of 'None' (line 1190)
    None_178107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 16), 'None')
    # Getting the type of 'None' (line 1190)
    None_178108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 22), 'None')
    # Getting the type of 'vx' (line 1190)
    vx_178109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1190)
    getitem___178110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1190, 8), vx_178109, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1190)
    subscript_call_result_178111 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 8), getitem___178110, (Ellipsis_178106, None_178107, None_178108))
    
    
    # Obtaining the type of the subscript
    Ellipsis_178112 = Ellipsis
    # Getting the type of 'None' (line 1190)
    None_178113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 36), 'None')
    slice_178114 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1190, 28), None, None, None)
    # Getting the type of 'None' (line 1190)
    None_178115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 44), 'None')
    # Getting the type of 'vy' (line 1190)
    vy_178116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 28), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1190)
    getitem___178117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1190, 28), vy_178116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1190)
    subscript_call_result_178118 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 28), getitem___178117, (Ellipsis_178112, None_178113, slice_178114, None_178115))
    
    # Applying the binary operator '*' (line 1190)
    result_mul_178119 = python_operator(stypy.reporting.localization.Localization(__file__, 1190, 8), '*', subscript_call_result_178111, subscript_call_result_178118)
    
    
    # Obtaining the type of the subscript
    Ellipsis_178120 = Ellipsis
    # Getting the type of 'None' (line 1190)
    None_178121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 58), 'None')
    # Getting the type of 'None' (line 1190)
    None_178122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 64), 'None')
    slice_178123 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1190, 50), None, None, None)
    # Getting the type of 'vz' (line 1190)
    vz_178124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 50), 'vz')
    # Obtaining the member '__getitem__' of a type (line 1190)
    getitem___178125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1190, 50), vz_178124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1190)
    subscript_call_result_178126 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 50), getitem___178125, (Ellipsis_178120, None_178121, None_178122, slice_178123))
    
    # Applying the binary operator '*' (line 1190)
    result_mul_178127 = python_operator(stypy.reporting.localization.Localization(__file__, 1190, 49), '*', result_mul_178119, subscript_call_result_178126)
    
    # Assigning a type to the variable 'v' (line 1190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 4), 'v', result_mul_178127)
    
    # Call to reshape(...): (line 1193)
    # Processing the call arguments (line 1193)
    
    # Obtaining the type of the subscript
    int_178130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1193, 30), 'int')
    slice_178131 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1193, 21), None, int_178130, None)
    # Getting the type of 'v' (line 1193)
    v_178132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1193)
    shape_178133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1193, 21), v_178132, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1193)
    getitem___178134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1193, 21), shape_178133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1193)
    subscript_call_result_178135 = invoke(stypy.reporting.localization.Localization(__file__, 1193, 21), getitem___178134, slice_178131)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1193)
    tuple_178136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1193, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1193)
    # Adding element type (line 1193)
    int_178137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1193, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1193, 37), tuple_178136, int_178137)
    
    # Applying the binary operator '+' (line 1193)
    result_add_178138 = python_operator(stypy.reporting.localization.Localization(__file__, 1193, 21), '+', subscript_call_result_178135, tuple_178136)
    
    # Processing the call keyword arguments (line 1193)
    kwargs_178139 = {}
    # Getting the type of 'v' (line 1193)
    v_178128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1193)
    reshape_178129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1193, 11), v_178128, 'reshape')
    # Calling reshape(args, kwargs) (line 1193)
    reshape_call_result_178140 = invoke(stypy.reporting.localization.Localization(__file__, 1193, 11), reshape_178129, *[result_add_178138], **kwargs_178139)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1193, 4), 'stypy_return_type', reshape_call_result_178140)
    
    # ################# End of 'polyvander3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyvander3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1129)
    stypy_return_type_178141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_178141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyvander3d'
    return stypy_return_type_178141

# Assigning a type to the variable 'polyvander3d' (line 1129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 0), 'polyvander3d', polyvander3d)

@norecursion
def polyfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1196)
    None_178142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 29), 'None')
    # Getting the type of 'False' (line 1196)
    False_178143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 40), 'False')
    # Getting the type of 'None' (line 1196)
    None_178144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 49), 'None')
    defaults = [None_178142, False_178143, None_178144]
    # Create a new context for function 'polyfit'
    module_type_store = module_type_store.open_function_context('polyfit', 1196, 0, False)
    
    # Passed parameters checking function
    polyfit.stypy_localization = localization
    polyfit.stypy_type_of_self = None
    polyfit.stypy_type_store = module_type_store
    polyfit.stypy_function_name = 'polyfit'
    polyfit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w']
    polyfit.stypy_varargs_param_name = None
    polyfit.stypy_kwargs_param_name = None
    polyfit.stypy_call_defaults = defaults
    polyfit.stypy_call_varargs = varargs
    polyfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyfit', ['x', 'y', 'deg', 'rcond', 'full', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyfit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyfit(...)' code ##################

    str_178145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1337, (-1)), 'str', '\n    Least-squares fit of a polynomial to data.\n\n    Return the coefficients of a polynomial of degree `deg` that is the\n    least squares fit to the data values `y` given at points `x`. If `y` is\n    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple\n    fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,\n\n    where `n` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (`M`,)\n        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.\n    y : array_like, shape (`M`,) or (`M`, `K`)\n        y-coordinates of the sample points.  Several sets of sample points\n        sharing the same x-coordinates can be (independently) fit with one\n        call to `polyfit` by passing in for `y` a 2-D array that contains\n        one data set per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For Numpy versions >= 1.11 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit.  Singular values smaller\n        than `rcond`, relative to the largest singular value, will be\n        ignored.  The default value is ``len(x)*eps``, where `eps` is the\n        relative precision of the platform\'s float type, about 2e-16 in\n        most cases.\n    full : bool, optional\n        Switch determining the nature of the return value.  When ``False``\n        (the default) just the coefficients are returned; when ``True``,\n        diagnostic information from the singular value decomposition (used\n        to solve the fit\'s matrix equation) is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the contribution of each point\n        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n        weights are chosen so that the errors of the products ``w[i]*y[i]``\n        all have the same variance.  The default value is None.\n\n        .. versionadded:: 1.5.0\n\n    Returns\n    -------\n    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)\n        Polynomial coefficients ordered from low to high.  If `y` was 2-D,\n        the coefficients in column `k` of `coef` represent the polynomial\n        fit to the data in `y`\'s `k`-th column.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if `full` = True\n\n        resid -- sum of squared residuals of the least squares fit\n        rank -- the numerical rank of the scaled Vandermonde matrix\n        sv -- singular values of the scaled Vandermonde matrix\n        rcond -- value of `rcond`.\n\n        For more details, see `linalg.lstsq`.\n\n    Raises\n    ------\n    RankWarning\n        Raised if the matrix in the least-squares fit is rank deficient.\n        The warning is only raised if `full` == False.  The warnings can\n        be turned off by:\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', RankWarning)\n\n    See Also\n    --------\n    chebfit, legfit, lagfit, hermfit, hermefit\n    polyval : Evaluates a polynomial.\n    polyvander : Vandermonde matrix for powers.\n    linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the polynomial `p` that minimizes\n    the sum of the weighted squared errors\n\n    .. math :: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where the :math:`w_j` are the weights. This problem is solved by\n    setting up the (typically) over-determined matrix equation:\n\n    .. math :: V(x) * c = w * y,\n\n    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the\n    coefficients to be solved for, `w` are the weights, and `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of `V`.\n\n    If some of the singular values of `V` are so small that they are\n    neglected (and `full` == ``False``), a `RankWarning` will be raised.\n    This means that the coefficient values may be poorly determined.\n    Fitting to a lower order polynomial will usually get rid of the warning\n    (but may not be what you want, of course; if you have independent\n    reason(s) for choosing the degree which isn\'t working, you may have to:\n    a) reconsider those reasons, and/or b) reconsider the quality of your\n    data).  The `rcond` parameter can also be set to a value smaller than\n    its default, but the resulting fit may be spurious and have large\n    contributions from roundoff error.\n\n    Polynomial fits using double precision tend to "fail" at about\n    (polynomial) degree 20. Fits using Chebyshev or Legendre series are\n    generally better conditioned, but much can still depend on the\n    distribution of the sample points and the smoothness of the data.  If\n    the quality of the fit is inadequate, splines may be a good\n    alternative.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import polynomial as P\n    >>> x = np.linspace(-1,1,51) # x "data": [-1, -0.96, ..., 0.96, 1]\n    >>> y = x**3 - x + np.random.randn(len(x)) # x^3 - x + N(0,1) "noise"\n    >>> c, stats = P.polyfit(x,y,3,full=True)\n    >>> c # c[0], c[2] should be approx. 0, c[1] approx. -1, c[3] approx. 1\n    array([ 0.01909725, -1.30598256, -0.00577963,  1.02644286])\n    >>> stats # note the large SSR, explaining the rather poor results\n    [array([ 38.06116253]), 4, array([ 1.38446749,  1.32119158,  0.50443316,\n    0.28853036]), 1.1324274851176597e-014]\n\n    Same thing without the added noise\n\n    >>> y = x**3 - x\n    >>> c, stats = P.polyfit(x,y,3,full=True)\n    >>> c # c[0], c[2] should be "very close to 0", c[1] ~= -1, c[3] ~= 1\n    array([ -1.73362882e-17,  -1.00000000e+00,  -2.67471909e-16,\n             1.00000000e+00])\n    >>> stats # note the minuscule SSR\n    [array([  7.46346754e-31]), 4, array([ 1.38446749,  1.32119158,\n    0.50443316,  0.28853036]), 1.1324274851176597e-014]\n\n    ')
    
    # Assigning a BinOp to a Name (line 1338):
    
    # Assigning a BinOp to a Name (line 1338):
    
    # Call to asarray(...): (line 1338)
    # Processing the call arguments (line 1338)
    # Getting the type of 'x' (line 1338)
    x_178148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1338, 19), 'x', False)
    # Processing the call keyword arguments (line 1338)
    kwargs_178149 = {}
    # Getting the type of 'np' (line 1338)
    np_178146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1338, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1338)
    asarray_178147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1338, 8), np_178146, 'asarray')
    # Calling asarray(args, kwargs) (line 1338)
    asarray_call_result_178150 = invoke(stypy.reporting.localization.Localization(__file__, 1338, 8), asarray_178147, *[x_178148], **kwargs_178149)
    
    float_178151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1338, 24), 'float')
    # Applying the binary operator '+' (line 1338)
    result_add_178152 = python_operator(stypy.reporting.localization.Localization(__file__, 1338, 8), '+', asarray_call_result_178150, float_178151)
    
    # Assigning a type to the variable 'x' (line 1338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1338, 4), 'x', result_add_178152)
    
    # Assigning a BinOp to a Name (line 1339):
    
    # Assigning a BinOp to a Name (line 1339):
    
    # Call to asarray(...): (line 1339)
    # Processing the call arguments (line 1339)
    # Getting the type of 'y' (line 1339)
    y_178155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1339, 19), 'y', False)
    # Processing the call keyword arguments (line 1339)
    kwargs_178156 = {}
    # Getting the type of 'np' (line 1339)
    np_178153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1339, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1339)
    asarray_178154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1339, 8), np_178153, 'asarray')
    # Calling asarray(args, kwargs) (line 1339)
    asarray_call_result_178157 = invoke(stypy.reporting.localization.Localization(__file__, 1339, 8), asarray_178154, *[y_178155], **kwargs_178156)
    
    float_178158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1339, 24), 'float')
    # Applying the binary operator '+' (line 1339)
    result_add_178159 = python_operator(stypy.reporting.localization.Localization(__file__, 1339, 8), '+', asarray_call_result_178157, float_178158)
    
    # Assigning a type to the variable 'y' (line 1339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1339, 4), 'y', result_add_178159)
    
    # Assigning a Call to a Name (line 1340):
    
    # Assigning a Call to a Name (line 1340):
    
    # Call to asarray(...): (line 1340)
    # Processing the call arguments (line 1340)
    # Getting the type of 'deg' (line 1340)
    deg_178162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 21), 'deg', False)
    # Processing the call keyword arguments (line 1340)
    kwargs_178163 = {}
    # Getting the type of 'np' (line 1340)
    np_178160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1340)
    asarray_178161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1340, 10), np_178160, 'asarray')
    # Calling asarray(args, kwargs) (line 1340)
    asarray_call_result_178164 = invoke(stypy.reporting.localization.Localization(__file__, 1340, 10), asarray_178161, *[deg_178162], **kwargs_178163)
    
    # Assigning a type to the variable 'deg' (line 1340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1340, 4), 'deg', asarray_call_result_178164)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deg' (line 1343)
    deg_178165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1343)
    ndim_178166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1343, 7), deg_178165, 'ndim')
    int_178167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1343, 18), 'int')
    # Applying the binary operator '>' (line 1343)
    result_gt_178168 = python_operator(stypy.reporting.localization.Localization(__file__, 1343, 7), '>', ndim_178166, int_178167)
    
    
    # Getting the type of 'deg' (line 1343)
    deg_178169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 23), 'deg')
    # Obtaining the member 'dtype' of a type (line 1343)
    dtype_178170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1343, 23), deg_178169, 'dtype')
    # Obtaining the member 'kind' of a type (line 1343)
    kind_178171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1343, 23), dtype_178170, 'kind')
    str_178172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1343, 45), 'str', 'iu')
    # Applying the binary operator 'notin' (line 1343)
    result_contains_178173 = python_operator(stypy.reporting.localization.Localization(__file__, 1343, 23), 'notin', kind_178171, str_178172)
    
    # Applying the binary operator 'or' (line 1343)
    result_or_keyword_178174 = python_operator(stypy.reporting.localization.Localization(__file__, 1343, 7), 'or', result_gt_178168, result_contains_178173)
    
    # Getting the type of 'deg' (line 1343)
    deg_178175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 53), 'deg')
    # Obtaining the member 'size' of a type (line 1343)
    size_178176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1343, 53), deg_178175, 'size')
    int_178177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1343, 65), 'int')
    # Applying the binary operator '==' (line 1343)
    result_eq_178178 = python_operator(stypy.reporting.localization.Localization(__file__, 1343, 53), '==', size_178176, int_178177)
    
    # Applying the binary operator 'or' (line 1343)
    result_or_keyword_178179 = python_operator(stypy.reporting.localization.Localization(__file__, 1343, 7), 'or', result_or_keyword_178174, result_eq_178178)
    
    # Testing the type of an if condition (line 1343)
    if_condition_178180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1343, 4), result_or_keyword_178179)
    # Assigning a type to the variable 'if_condition_178180' (line 1343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1343, 4), 'if_condition_178180', if_condition_178180)
    # SSA begins for if statement (line 1343)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1344)
    # Processing the call arguments (line 1344)
    str_178182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1344, 24), 'str', 'deg must be an int or non-empty 1-D array of int')
    # Processing the call keyword arguments (line 1344)
    kwargs_178183 = {}
    # Getting the type of 'TypeError' (line 1344)
    TypeError_178181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1344, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1344)
    TypeError_call_result_178184 = invoke(stypy.reporting.localization.Localization(__file__, 1344, 14), TypeError_178181, *[str_178182], **kwargs_178183)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1344, 8), TypeError_call_result_178184, 'raise parameter', BaseException)
    # SSA join for if statement (line 1343)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to min(...): (line 1345)
    # Processing the call keyword arguments (line 1345)
    kwargs_178187 = {}
    # Getting the type of 'deg' (line 1345)
    deg_178185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 7), 'deg', False)
    # Obtaining the member 'min' of a type (line 1345)
    min_178186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 7), deg_178185, 'min')
    # Calling min(args, kwargs) (line 1345)
    min_call_result_178188 = invoke(stypy.reporting.localization.Localization(__file__, 1345, 7), min_178186, *[], **kwargs_178187)
    
    int_178189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 19), 'int')
    # Applying the binary operator '<' (line 1345)
    result_lt_178190 = python_operator(stypy.reporting.localization.Localization(__file__, 1345, 7), '<', min_call_result_178188, int_178189)
    
    # Testing the type of an if condition (line 1345)
    if_condition_178191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1345, 4), result_lt_178190)
    # Assigning a type to the variable 'if_condition_178191' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 4), 'if_condition_178191', if_condition_178191)
    # SSA begins for if statement (line 1345)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1346)
    # Processing the call arguments (line 1346)
    str_178193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1346, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 1346)
    kwargs_178194 = {}
    # Getting the type of 'ValueError' (line 1346)
    ValueError_178192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1346, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1346)
    ValueError_call_result_178195 = invoke(stypy.reporting.localization.Localization(__file__, 1346, 14), ValueError_178192, *[str_178193], **kwargs_178194)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1346, 8), ValueError_call_result_178195, 'raise parameter', BaseException)
    # SSA join for if statement (line 1345)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1347)
    x_178196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1347, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1347)
    ndim_178197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1347, 7), x_178196, 'ndim')
    int_178198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1347, 17), 'int')
    # Applying the binary operator '!=' (line 1347)
    result_ne_178199 = python_operator(stypy.reporting.localization.Localization(__file__, 1347, 7), '!=', ndim_178197, int_178198)
    
    # Testing the type of an if condition (line 1347)
    if_condition_178200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1347, 4), result_ne_178199)
    # Assigning a type to the variable 'if_condition_178200' (line 1347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1347, 4), 'if_condition_178200', if_condition_178200)
    # SSA begins for if statement (line 1347)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1348)
    # Processing the call arguments (line 1348)
    str_178202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1348, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 1348)
    kwargs_178203 = {}
    # Getting the type of 'TypeError' (line 1348)
    TypeError_178201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1348)
    TypeError_call_result_178204 = invoke(stypy.reporting.localization.Localization(__file__, 1348, 14), TypeError_178201, *[str_178202], **kwargs_178203)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1348, 8), TypeError_call_result_178204, 'raise parameter', BaseException)
    # SSA join for if statement (line 1347)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1349)
    x_178205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 7), 'x')
    # Obtaining the member 'size' of a type (line 1349)
    size_178206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1349, 7), x_178205, 'size')
    int_178207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1349, 17), 'int')
    # Applying the binary operator '==' (line 1349)
    result_eq_178208 = python_operator(stypy.reporting.localization.Localization(__file__, 1349, 7), '==', size_178206, int_178207)
    
    # Testing the type of an if condition (line 1349)
    if_condition_178209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1349, 4), result_eq_178208)
    # Assigning a type to the variable 'if_condition_178209' (line 1349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1349, 4), 'if_condition_178209', if_condition_178209)
    # SSA begins for if statement (line 1349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1350)
    # Processing the call arguments (line 1350)
    str_178211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1350, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 1350)
    kwargs_178212 = {}
    # Getting the type of 'TypeError' (line 1350)
    TypeError_178210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1350)
    TypeError_call_result_178213 = invoke(stypy.reporting.localization.Localization(__file__, 1350, 14), TypeError_178210, *[str_178211], **kwargs_178212)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1350, 8), TypeError_call_result_178213, 'raise parameter', BaseException)
    # SSA join for if statement (line 1349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 1351)
    y_178214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1351)
    ndim_178215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1351, 7), y_178214, 'ndim')
    int_178216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1351, 16), 'int')
    # Applying the binary operator '<' (line 1351)
    result_lt_178217 = python_operator(stypy.reporting.localization.Localization(__file__, 1351, 7), '<', ndim_178215, int_178216)
    
    
    # Getting the type of 'y' (line 1351)
    y_178218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 1351)
    ndim_178219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1351, 21), y_178218, 'ndim')
    int_178220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1351, 30), 'int')
    # Applying the binary operator '>' (line 1351)
    result_gt_178221 = python_operator(stypy.reporting.localization.Localization(__file__, 1351, 21), '>', ndim_178219, int_178220)
    
    # Applying the binary operator 'or' (line 1351)
    result_or_keyword_178222 = python_operator(stypy.reporting.localization.Localization(__file__, 1351, 7), 'or', result_lt_178217, result_gt_178221)
    
    # Testing the type of an if condition (line 1351)
    if_condition_178223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1351, 4), result_or_keyword_178222)
    # Assigning a type to the variable 'if_condition_178223' (line 1351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1351, 4), 'if_condition_178223', if_condition_178223)
    # SSA begins for if statement (line 1351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1352)
    # Processing the call arguments (line 1352)
    str_178225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1352, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 1352)
    kwargs_178226 = {}
    # Getting the type of 'TypeError' (line 1352)
    TypeError_178224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1352)
    TypeError_call_result_178227 = invoke(stypy.reporting.localization.Localization(__file__, 1352, 14), TypeError_178224, *[str_178225], **kwargs_178226)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1352, 8), TypeError_call_result_178227, 'raise parameter', BaseException)
    # SSA join for if statement (line 1351)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1353)
    # Processing the call arguments (line 1353)
    # Getting the type of 'x' (line 1353)
    x_178229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 11), 'x', False)
    # Processing the call keyword arguments (line 1353)
    kwargs_178230 = {}
    # Getting the type of 'len' (line 1353)
    len_178228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 7), 'len', False)
    # Calling len(args, kwargs) (line 1353)
    len_call_result_178231 = invoke(stypy.reporting.localization.Localization(__file__, 1353, 7), len_178228, *[x_178229], **kwargs_178230)
    
    
    # Call to len(...): (line 1353)
    # Processing the call arguments (line 1353)
    # Getting the type of 'y' (line 1353)
    y_178233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 21), 'y', False)
    # Processing the call keyword arguments (line 1353)
    kwargs_178234 = {}
    # Getting the type of 'len' (line 1353)
    len_178232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 17), 'len', False)
    # Calling len(args, kwargs) (line 1353)
    len_call_result_178235 = invoke(stypy.reporting.localization.Localization(__file__, 1353, 17), len_178232, *[y_178233], **kwargs_178234)
    
    # Applying the binary operator '!=' (line 1353)
    result_ne_178236 = python_operator(stypy.reporting.localization.Localization(__file__, 1353, 7), '!=', len_call_result_178231, len_call_result_178235)
    
    # Testing the type of an if condition (line 1353)
    if_condition_178237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1353, 4), result_ne_178236)
    # Assigning a type to the variable 'if_condition_178237' (line 1353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1353, 4), 'if_condition_178237', if_condition_178237)
    # SSA begins for if statement (line 1353)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1354)
    # Processing the call arguments (line 1354)
    str_178239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 1354)
    kwargs_178240 = {}
    # Getting the type of 'TypeError' (line 1354)
    TypeError_178238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1354)
    TypeError_call_result_178241 = invoke(stypy.reporting.localization.Localization(__file__, 1354, 14), TypeError_178238, *[str_178239], **kwargs_178240)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1354, 8), TypeError_call_result_178241, 'raise parameter', BaseException)
    # SSA join for if statement (line 1353)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'deg' (line 1356)
    deg_178242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1356)
    ndim_178243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 7), deg_178242, 'ndim')
    int_178244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 19), 'int')
    # Applying the binary operator '==' (line 1356)
    result_eq_178245 = python_operator(stypy.reporting.localization.Localization(__file__, 1356, 7), '==', ndim_178243, int_178244)
    
    # Testing the type of an if condition (line 1356)
    if_condition_178246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1356, 4), result_eq_178245)
    # Assigning a type to the variable 'if_condition_178246' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'if_condition_178246', if_condition_178246)
    # SSA begins for if statement (line 1356)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1357):
    
    # Assigning a Name to a Name (line 1357):
    # Getting the type of 'deg' (line 1357)
    deg_178247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 15), 'deg')
    # Assigning a type to the variable 'lmax' (line 1357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 8), 'lmax', deg_178247)
    
    # Assigning a BinOp to a Name (line 1358):
    
    # Assigning a BinOp to a Name (line 1358):
    # Getting the type of 'lmax' (line 1358)
    lmax_178248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 16), 'lmax')
    int_178249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 23), 'int')
    # Applying the binary operator '+' (line 1358)
    result_add_178250 = python_operator(stypy.reporting.localization.Localization(__file__, 1358, 16), '+', lmax_178248, int_178249)
    
    # Assigning a type to the variable 'order' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 8), 'order', result_add_178250)
    
    # Assigning a Call to a Name (line 1359):
    
    # Assigning a Call to a Name (line 1359):
    
    # Call to polyvander(...): (line 1359)
    # Processing the call arguments (line 1359)
    # Getting the type of 'x' (line 1359)
    x_178252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 25), 'x', False)
    # Getting the type of 'lmax' (line 1359)
    lmax_178253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 28), 'lmax', False)
    # Processing the call keyword arguments (line 1359)
    kwargs_178254 = {}
    # Getting the type of 'polyvander' (line 1359)
    polyvander_178251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 14), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1359)
    polyvander_call_result_178255 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 14), polyvander_178251, *[x_178252, lmax_178253], **kwargs_178254)
    
    # Assigning a type to the variable 'van' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 8), 'van', polyvander_call_result_178255)
    # SSA branch for the else part of an if statement (line 1356)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1361):
    
    # Assigning a Call to a Name (line 1361):
    
    # Call to sort(...): (line 1361)
    # Processing the call arguments (line 1361)
    # Getting the type of 'deg' (line 1361)
    deg_178258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 22), 'deg', False)
    # Processing the call keyword arguments (line 1361)
    kwargs_178259 = {}
    # Getting the type of 'np' (line 1361)
    np_178256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 14), 'np', False)
    # Obtaining the member 'sort' of a type (line 1361)
    sort_178257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1361, 14), np_178256, 'sort')
    # Calling sort(args, kwargs) (line 1361)
    sort_call_result_178260 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 14), sort_178257, *[deg_178258], **kwargs_178259)
    
    # Assigning a type to the variable 'deg' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 8), 'deg', sort_call_result_178260)
    
    # Assigning a Subscript to a Name (line 1362):
    
    # Assigning a Subscript to a Name (line 1362):
    
    # Obtaining the type of the subscript
    int_178261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 19), 'int')
    # Getting the type of 'deg' (line 1362)
    deg_178262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 15), 'deg')
    # Obtaining the member '__getitem__' of a type (line 1362)
    getitem___178263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 15), deg_178262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1362)
    subscript_call_result_178264 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 15), getitem___178263, int_178261)
    
    # Assigning a type to the variable 'lmax' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 8), 'lmax', subscript_call_result_178264)
    
    # Assigning a Call to a Name (line 1363):
    
    # Assigning a Call to a Name (line 1363):
    
    # Call to len(...): (line 1363)
    # Processing the call arguments (line 1363)
    # Getting the type of 'deg' (line 1363)
    deg_178266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 20), 'deg', False)
    # Processing the call keyword arguments (line 1363)
    kwargs_178267 = {}
    # Getting the type of 'len' (line 1363)
    len_178265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 16), 'len', False)
    # Calling len(args, kwargs) (line 1363)
    len_call_result_178268 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 16), len_178265, *[deg_178266], **kwargs_178267)
    
    # Assigning a type to the variable 'order' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 8), 'order', len_call_result_178268)
    
    # Assigning a Subscript to a Name (line 1364):
    
    # Assigning a Subscript to a Name (line 1364):
    
    # Obtaining the type of the subscript
    slice_178269 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1364, 14), None, None, None)
    # Getting the type of 'deg' (line 1364)
    deg_178270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 37), 'deg')
    
    # Call to polyvander(...): (line 1364)
    # Processing the call arguments (line 1364)
    # Getting the type of 'x' (line 1364)
    x_178272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 25), 'x', False)
    # Getting the type of 'lmax' (line 1364)
    lmax_178273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 28), 'lmax', False)
    # Processing the call keyword arguments (line 1364)
    kwargs_178274 = {}
    # Getting the type of 'polyvander' (line 1364)
    polyvander_178271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 14), 'polyvander', False)
    # Calling polyvander(args, kwargs) (line 1364)
    polyvander_call_result_178275 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 14), polyvander_178271, *[x_178272, lmax_178273], **kwargs_178274)
    
    # Obtaining the member '__getitem__' of a type (line 1364)
    getitem___178276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 14), polyvander_call_result_178275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1364)
    subscript_call_result_178277 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 14), getitem___178276, (slice_178269, deg_178270))
    
    # Assigning a type to the variable 'van' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 8), 'van', subscript_call_result_178277)
    # SSA join for if statement (line 1356)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1367):
    
    # Assigning a Attribute to a Name (line 1367):
    # Getting the type of 'van' (line 1367)
    van_178278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 10), 'van')
    # Obtaining the member 'T' of a type (line 1367)
    T_178279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 10), van_178278, 'T')
    # Assigning a type to the variable 'lhs' (line 1367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1367, 4), 'lhs', T_178279)
    
    # Assigning a Attribute to a Name (line 1368):
    
    # Assigning a Attribute to a Name (line 1368):
    # Getting the type of 'y' (line 1368)
    y_178280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 10), 'y')
    # Obtaining the member 'T' of a type (line 1368)
    T_178281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 10), y_178280, 'T')
    # Assigning a type to the variable 'rhs' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'rhs', T_178281)
    
    # Type idiom detected: calculating its left and rigth part (line 1369)
    # Getting the type of 'w' (line 1369)
    w_178282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 4), 'w')
    # Getting the type of 'None' (line 1369)
    None_178283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 16), 'None')
    
    (may_be_178284, more_types_in_union_178285) = may_not_be_none(w_178282, None_178283)

    if may_be_178284:

        if more_types_in_union_178285:
            # Runtime conditional SSA (line 1369)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1370):
        
        # Assigning a BinOp to a Name (line 1370):
        
        # Call to asarray(...): (line 1370)
        # Processing the call arguments (line 1370)
        # Getting the type of 'w' (line 1370)
        w_178288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 23), 'w', False)
        # Processing the call keyword arguments (line 1370)
        kwargs_178289 = {}
        # Getting the type of 'np' (line 1370)
        np_178286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1370)
        asarray_178287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1370, 12), np_178286, 'asarray')
        # Calling asarray(args, kwargs) (line 1370)
        asarray_call_result_178290 = invoke(stypy.reporting.localization.Localization(__file__, 1370, 12), asarray_178287, *[w_178288], **kwargs_178289)
        
        float_178291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1370, 28), 'float')
        # Applying the binary operator '+' (line 1370)
        result_add_178292 = python_operator(stypy.reporting.localization.Localization(__file__, 1370, 12), '+', asarray_call_result_178290, float_178291)
        
        # Assigning a type to the variable 'w' (line 1370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1370, 8), 'w', result_add_178292)
        
        
        # Getting the type of 'w' (line 1371)
        w_178293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 1371)
        ndim_178294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1371, 11), w_178293, 'ndim')
        int_178295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1371, 21), 'int')
        # Applying the binary operator '!=' (line 1371)
        result_ne_178296 = python_operator(stypy.reporting.localization.Localization(__file__, 1371, 11), '!=', ndim_178294, int_178295)
        
        # Testing the type of an if condition (line 1371)
        if_condition_178297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1371, 8), result_ne_178296)
        # Assigning a type to the variable 'if_condition_178297' (line 1371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1371, 8), 'if_condition_178297', if_condition_178297)
        # SSA begins for if statement (line 1371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1372)
        # Processing the call arguments (line 1372)
        str_178299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 28), 'str', 'expected 1D vector for w')
        # Processing the call keyword arguments (line 1372)
        kwargs_178300 = {}
        # Getting the type of 'TypeError' (line 1372)
        TypeError_178298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1372)
        TypeError_call_result_178301 = invoke(stypy.reporting.localization.Localization(__file__, 1372, 18), TypeError_178298, *[str_178299], **kwargs_178300)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1372, 12), TypeError_call_result_178301, 'raise parameter', BaseException)
        # SSA join for if statement (line 1371)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 1373)
        # Processing the call arguments (line 1373)
        # Getting the type of 'x' (line 1373)
        x_178303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 15), 'x', False)
        # Processing the call keyword arguments (line 1373)
        kwargs_178304 = {}
        # Getting the type of 'len' (line 1373)
        len_178302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 11), 'len', False)
        # Calling len(args, kwargs) (line 1373)
        len_call_result_178305 = invoke(stypy.reporting.localization.Localization(__file__, 1373, 11), len_178302, *[x_178303], **kwargs_178304)
        
        
        # Call to len(...): (line 1373)
        # Processing the call arguments (line 1373)
        # Getting the type of 'w' (line 1373)
        w_178307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 25), 'w', False)
        # Processing the call keyword arguments (line 1373)
        kwargs_178308 = {}
        # Getting the type of 'len' (line 1373)
        len_178306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 21), 'len', False)
        # Calling len(args, kwargs) (line 1373)
        len_call_result_178309 = invoke(stypy.reporting.localization.Localization(__file__, 1373, 21), len_178306, *[w_178307], **kwargs_178308)
        
        # Applying the binary operator '!=' (line 1373)
        result_ne_178310 = python_operator(stypy.reporting.localization.Localization(__file__, 1373, 11), '!=', len_call_result_178305, len_call_result_178309)
        
        # Testing the type of an if condition (line 1373)
        if_condition_178311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1373, 8), result_ne_178310)
        # Assigning a type to the variable 'if_condition_178311' (line 1373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1373, 8), 'if_condition_178311', if_condition_178311)
        # SSA begins for if statement (line 1373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1374)
        # Processing the call arguments (line 1374)
        str_178313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1374, 28), 'str', 'expected x and w to have same length')
        # Processing the call keyword arguments (line 1374)
        kwargs_178314 = {}
        # Getting the type of 'TypeError' (line 1374)
        TypeError_178312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1374)
        TypeError_call_result_178315 = invoke(stypy.reporting.localization.Localization(__file__, 1374, 18), TypeError_178312, *[str_178313], **kwargs_178314)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1374, 12), TypeError_call_result_178315, 'raise parameter', BaseException)
        # SSA join for if statement (line 1373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1377):
        
        # Assigning a BinOp to a Name (line 1377):
        # Getting the type of 'lhs' (line 1377)
        lhs_178316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 14), 'lhs')
        # Getting the type of 'w' (line 1377)
        w_178317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 20), 'w')
        # Applying the binary operator '*' (line 1377)
        result_mul_178318 = python_operator(stypy.reporting.localization.Localization(__file__, 1377, 14), '*', lhs_178316, w_178317)
        
        # Assigning a type to the variable 'lhs' (line 1377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1377, 8), 'lhs', result_mul_178318)
        
        # Assigning a BinOp to a Name (line 1378):
        
        # Assigning a BinOp to a Name (line 1378):
        # Getting the type of 'rhs' (line 1378)
        rhs_178319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 14), 'rhs')
        # Getting the type of 'w' (line 1378)
        w_178320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 20), 'w')
        # Applying the binary operator '*' (line 1378)
        result_mul_178321 = python_operator(stypy.reporting.localization.Localization(__file__, 1378, 14), '*', rhs_178319, w_178320)
        
        # Assigning a type to the variable 'rhs' (line 1378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 8), 'rhs', result_mul_178321)

        if more_types_in_union_178285:
            # SSA join for if statement (line 1369)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1381)
    # Getting the type of 'rcond' (line 1381)
    rcond_178322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 7), 'rcond')
    # Getting the type of 'None' (line 1381)
    None_178323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 16), 'None')
    
    (may_be_178324, more_types_in_union_178325) = may_be_none(rcond_178322, None_178323)

    if may_be_178324:

        if more_types_in_union_178325:
            # Runtime conditional SSA (line 1381)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1382):
        
        # Assigning a BinOp to a Name (line 1382):
        
        # Call to len(...): (line 1382)
        # Processing the call arguments (line 1382)
        # Getting the type of 'x' (line 1382)
        x_178327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 20), 'x', False)
        # Processing the call keyword arguments (line 1382)
        kwargs_178328 = {}
        # Getting the type of 'len' (line 1382)
        len_178326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 16), 'len', False)
        # Calling len(args, kwargs) (line 1382)
        len_call_result_178329 = invoke(stypy.reporting.localization.Localization(__file__, 1382, 16), len_178326, *[x_178327], **kwargs_178328)
        
        
        # Call to finfo(...): (line 1382)
        # Processing the call arguments (line 1382)
        # Getting the type of 'x' (line 1382)
        x_178332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 32), 'x', False)
        # Obtaining the member 'dtype' of a type (line 1382)
        dtype_178333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1382, 32), x_178332, 'dtype')
        # Processing the call keyword arguments (line 1382)
        kwargs_178334 = {}
        # Getting the type of 'np' (line 1382)
        np_178330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 23), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1382)
        finfo_178331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1382, 23), np_178330, 'finfo')
        # Calling finfo(args, kwargs) (line 1382)
        finfo_call_result_178335 = invoke(stypy.reporting.localization.Localization(__file__, 1382, 23), finfo_178331, *[dtype_178333], **kwargs_178334)
        
        # Obtaining the member 'eps' of a type (line 1382)
        eps_178336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1382, 23), finfo_call_result_178335, 'eps')
        # Applying the binary operator '*' (line 1382)
        result_mul_178337 = python_operator(stypy.reporting.localization.Localization(__file__, 1382, 16), '*', len_call_result_178329, eps_178336)
        
        # Assigning a type to the variable 'rcond' (line 1382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1382, 8), 'rcond', result_mul_178337)

        if more_types_in_union_178325:
            # SSA join for if statement (line 1381)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issubclass(...): (line 1385)
    # Processing the call arguments (line 1385)
    # Getting the type of 'lhs' (line 1385)
    lhs_178339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 18), 'lhs', False)
    # Obtaining the member 'dtype' of a type (line 1385)
    dtype_178340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1385, 18), lhs_178339, 'dtype')
    # Obtaining the member 'type' of a type (line 1385)
    type_178341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1385, 18), dtype_178340, 'type')
    # Getting the type of 'np' (line 1385)
    np_178342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 34), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1385)
    complexfloating_178343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1385, 34), np_178342, 'complexfloating')
    # Processing the call keyword arguments (line 1385)
    kwargs_178344 = {}
    # Getting the type of 'issubclass' (line 1385)
    issubclass_178338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1385)
    issubclass_call_result_178345 = invoke(stypy.reporting.localization.Localization(__file__, 1385, 7), issubclass_178338, *[type_178341, complexfloating_178343], **kwargs_178344)
    
    # Testing the type of an if condition (line 1385)
    if_condition_178346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1385, 4), issubclass_call_result_178345)
    # Assigning a type to the variable 'if_condition_178346' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 4), 'if_condition_178346', if_condition_178346)
    # SSA begins for if statement (line 1385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1386):
    
    # Assigning a Call to a Name (line 1386):
    
    # Call to sqrt(...): (line 1386)
    # Processing the call arguments (line 1386)
    
    # Call to sum(...): (line 1386)
    # Processing the call arguments (line 1386)
    int_178363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 70), 'int')
    # Processing the call keyword arguments (line 1386)
    kwargs_178364 = {}
    
    # Call to square(...): (line 1386)
    # Processing the call arguments (line 1386)
    # Getting the type of 'lhs' (line 1386)
    lhs_178351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 33), 'lhs', False)
    # Obtaining the member 'real' of a type (line 1386)
    real_178352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 33), lhs_178351, 'real')
    # Processing the call keyword arguments (line 1386)
    kwargs_178353 = {}
    # Getting the type of 'np' (line 1386)
    np_178349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 1386)
    square_178350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 23), np_178349, 'square')
    # Calling square(args, kwargs) (line 1386)
    square_call_result_178354 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 23), square_178350, *[real_178352], **kwargs_178353)
    
    
    # Call to square(...): (line 1386)
    # Processing the call arguments (line 1386)
    # Getting the type of 'lhs' (line 1386)
    lhs_178357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 55), 'lhs', False)
    # Obtaining the member 'imag' of a type (line 1386)
    imag_178358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 55), lhs_178357, 'imag')
    # Processing the call keyword arguments (line 1386)
    kwargs_178359 = {}
    # Getting the type of 'np' (line 1386)
    np_178355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 45), 'np', False)
    # Obtaining the member 'square' of a type (line 1386)
    square_178356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 45), np_178355, 'square')
    # Calling square(args, kwargs) (line 1386)
    square_call_result_178360 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 45), square_178356, *[imag_178358], **kwargs_178359)
    
    # Applying the binary operator '+' (line 1386)
    result_add_178361 = python_operator(stypy.reporting.localization.Localization(__file__, 1386, 23), '+', square_call_result_178354, square_call_result_178360)
    
    # Obtaining the member 'sum' of a type (line 1386)
    sum_178362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 23), result_add_178361, 'sum')
    # Calling sum(args, kwargs) (line 1386)
    sum_call_result_178365 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 23), sum_178362, *[int_178363], **kwargs_178364)
    
    # Processing the call keyword arguments (line 1386)
    kwargs_178366 = {}
    # Getting the type of 'np' (line 1386)
    np_178347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1386)
    sqrt_178348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 14), np_178347, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1386)
    sqrt_call_result_178367 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 14), sqrt_178348, *[sum_call_result_178365], **kwargs_178366)
    
    # Assigning a type to the variable 'scl' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 8), 'scl', sqrt_call_result_178367)
    # SSA branch for the else part of an if statement (line 1385)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1388):
    
    # Assigning a Call to a Name (line 1388):
    
    # Call to sqrt(...): (line 1388)
    # Processing the call arguments (line 1388)
    
    # Call to sum(...): (line 1388)
    # Processing the call arguments (line 1388)
    int_178376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, 41), 'int')
    # Processing the call keyword arguments (line 1388)
    kwargs_178377 = {}
    
    # Call to square(...): (line 1388)
    # Processing the call arguments (line 1388)
    # Getting the type of 'lhs' (line 1388)
    lhs_178372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 32), 'lhs', False)
    # Processing the call keyword arguments (line 1388)
    kwargs_178373 = {}
    # Getting the type of 'np' (line 1388)
    np_178370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 22), 'np', False)
    # Obtaining the member 'square' of a type (line 1388)
    square_178371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 22), np_178370, 'square')
    # Calling square(args, kwargs) (line 1388)
    square_call_result_178374 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 22), square_178371, *[lhs_178372], **kwargs_178373)
    
    # Obtaining the member 'sum' of a type (line 1388)
    sum_178375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 22), square_call_result_178374, 'sum')
    # Calling sum(args, kwargs) (line 1388)
    sum_call_result_178378 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 22), sum_178375, *[int_178376], **kwargs_178377)
    
    # Processing the call keyword arguments (line 1388)
    kwargs_178379 = {}
    # Getting the type of 'np' (line 1388)
    np_178368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1388)
    sqrt_178369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 14), np_178368, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1388)
    sqrt_call_result_178380 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 14), sqrt_178369, *[sum_call_result_178378], **kwargs_178379)
    
    # Assigning a type to the variable 'scl' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 8), 'scl', sqrt_call_result_178380)
    # SSA join for if statement (line 1385)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 1389):
    
    # Assigning a Num to a Subscript (line 1389):
    int_178381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 20), 'int')
    # Getting the type of 'scl' (line 1389)
    scl_178382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'scl')
    
    # Getting the type of 'scl' (line 1389)
    scl_178383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 8), 'scl')
    int_178384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 15), 'int')
    # Applying the binary operator '==' (line 1389)
    result_eq_178385 = python_operator(stypy.reporting.localization.Localization(__file__, 1389, 8), '==', scl_178383, int_178384)
    
    # Storing an element on a container (line 1389)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1389, 4), scl_178382, (result_eq_178385, int_178381))
    
    # Assigning a Call to a Tuple (line 1392):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 1392)
    # Processing the call arguments (line 1392)
    # Getting the type of 'lhs' (line 1392)
    lhs_178388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 34), 'lhs', False)
    # Obtaining the member 'T' of a type (line 1392)
    T_178389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 34), lhs_178388, 'T')
    # Getting the type of 'scl' (line 1392)
    scl_178390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 40), 'scl', False)
    # Applying the binary operator 'div' (line 1392)
    result_div_178391 = python_operator(stypy.reporting.localization.Localization(__file__, 1392, 34), 'div', T_178389, scl_178390)
    
    # Getting the type of 'rhs' (line 1392)
    rhs_178392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 45), 'rhs', False)
    # Obtaining the member 'T' of a type (line 1392)
    T_178393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 45), rhs_178392, 'T')
    # Getting the type of 'rcond' (line 1392)
    rcond_178394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 52), 'rcond', False)
    # Processing the call keyword arguments (line 1392)
    kwargs_178395 = {}
    # Getting the type of 'la' (line 1392)
    la_178386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 25), 'la', False)
    # Obtaining the member 'lstsq' of a type (line 1392)
    lstsq_178387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 25), la_178386, 'lstsq')
    # Calling lstsq(args, kwargs) (line 1392)
    lstsq_call_result_178396 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 25), lstsq_178387, *[result_div_178391, T_178393, rcond_178394], **kwargs_178395)
    
    # Assigning a type to the variable 'call_assignment_176522' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176522', lstsq_call_result_178396)
    
    # Assigning a Call to a Name (line 1392):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178400 = {}
    # Getting the type of 'call_assignment_176522' (line 1392)
    call_assignment_176522_178397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176522', False)
    # Obtaining the member '__getitem__' of a type (line 1392)
    getitem___178398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 4), call_assignment_176522_178397, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178401 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178398, *[int_178399], **kwargs_178400)
    
    # Assigning a type to the variable 'call_assignment_176523' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176523', getitem___call_result_178401)
    
    # Assigning a Name to a Name (line 1392):
    # Getting the type of 'call_assignment_176523' (line 1392)
    call_assignment_176523_178402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176523')
    # Assigning a type to the variable 'c' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'c', call_assignment_176523_178402)
    
    # Assigning a Call to a Name (line 1392):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178406 = {}
    # Getting the type of 'call_assignment_176522' (line 1392)
    call_assignment_176522_178403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176522', False)
    # Obtaining the member '__getitem__' of a type (line 1392)
    getitem___178404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 4), call_assignment_176522_178403, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178407 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178404, *[int_178405], **kwargs_178406)
    
    # Assigning a type to the variable 'call_assignment_176524' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176524', getitem___call_result_178407)
    
    # Assigning a Name to a Name (line 1392):
    # Getting the type of 'call_assignment_176524' (line 1392)
    call_assignment_176524_178408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176524')
    # Assigning a type to the variable 'resids' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 7), 'resids', call_assignment_176524_178408)
    
    # Assigning a Call to a Name (line 1392):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178412 = {}
    # Getting the type of 'call_assignment_176522' (line 1392)
    call_assignment_176522_178409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176522', False)
    # Obtaining the member '__getitem__' of a type (line 1392)
    getitem___178410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 4), call_assignment_176522_178409, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178413 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178410, *[int_178411], **kwargs_178412)
    
    # Assigning a type to the variable 'call_assignment_176525' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176525', getitem___call_result_178413)
    
    # Assigning a Name to a Name (line 1392):
    # Getting the type of 'call_assignment_176525' (line 1392)
    call_assignment_176525_178414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176525')
    # Assigning a type to the variable 'rank' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 15), 'rank', call_assignment_176525_178414)
    
    # Assigning a Call to a Name (line 1392):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178418 = {}
    # Getting the type of 'call_assignment_176522' (line 1392)
    call_assignment_176522_178415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176522', False)
    # Obtaining the member '__getitem__' of a type (line 1392)
    getitem___178416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 4), call_assignment_176522_178415, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178419 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178416, *[int_178417], **kwargs_178418)
    
    # Assigning a type to the variable 'call_assignment_176526' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176526', getitem___call_result_178419)
    
    # Assigning a Name to a Name (line 1392):
    # Getting the type of 'call_assignment_176526' (line 1392)
    call_assignment_176526_178420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'call_assignment_176526')
    # Assigning a type to the variable 's' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 21), 's', call_assignment_176526_178420)
    
    # Assigning a Attribute to a Name (line 1393):
    
    # Assigning a Attribute to a Name (line 1393):
    # Getting the type of 'c' (line 1393)
    c_178421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 9), 'c')
    # Obtaining the member 'T' of a type (line 1393)
    T_178422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1393, 9), c_178421, 'T')
    # Getting the type of 'scl' (line 1393)
    scl_178423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 13), 'scl')
    # Applying the binary operator 'div' (line 1393)
    result_div_178424 = python_operator(stypy.reporting.localization.Localization(__file__, 1393, 9), 'div', T_178422, scl_178423)
    
    # Obtaining the member 'T' of a type (line 1393)
    T_178425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1393, 9), result_div_178424, 'T')
    # Assigning a type to the variable 'c' (line 1393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1393, 4), 'c', T_178425)
    
    
    # Getting the type of 'deg' (line 1396)
    deg_178426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1396, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1396)
    ndim_178427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1396, 7), deg_178426, 'ndim')
    int_178428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1396, 19), 'int')
    # Applying the binary operator '==' (line 1396)
    result_eq_178429 = python_operator(stypy.reporting.localization.Localization(__file__, 1396, 7), '==', ndim_178427, int_178428)
    
    # Testing the type of an if condition (line 1396)
    if_condition_178430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1396, 4), result_eq_178429)
    # Assigning a type to the variable 'if_condition_178430' (line 1396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1396, 4), 'if_condition_178430', if_condition_178430)
    # SSA begins for if statement (line 1396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'c' (line 1397)
    c_178431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 11), 'c')
    # Obtaining the member 'ndim' of a type (line 1397)
    ndim_178432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 11), c_178431, 'ndim')
    int_178433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1397, 21), 'int')
    # Applying the binary operator '==' (line 1397)
    result_eq_178434 = python_operator(stypy.reporting.localization.Localization(__file__, 1397, 11), '==', ndim_178432, int_178433)
    
    # Testing the type of an if condition (line 1397)
    if_condition_178435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1397, 8), result_eq_178434)
    # Assigning a type to the variable 'if_condition_178435' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 8), 'if_condition_178435', if_condition_178435)
    # SSA begins for if statement (line 1397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1398):
    
    # Assigning a Call to a Name (line 1398):
    
    # Call to zeros(...): (line 1398)
    # Processing the call arguments (line 1398)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1398)
    tuple_178438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1398, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1398)
    # Adding element type (line 1398)
    # Getting the type of 'lmax' (line 1398)
    lmax_178439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 27), 'lmax', False)
    int_178440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1398, 34), 'int')
    # Applying the binary operator '+' (line 1398)
    result_add_178441 = python_operator(stypy.reporting.localization.Localization(__file__, 1398, 27), '+', lmax_178439, int_178440)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1398, 27), tuple_178438, result_add_178441)
    # Adding element type (line 1398)
    
    # Obtaining the type of the subscript
    int_178442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1398, 45), 'int')
    # Getting the type of 'c' (line 1398)
    c_178443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 37), 'c', False)
    # Obtaining the member 'shape' of a type (line 1398)
    shape_178444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 37), c_178443, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1398)
    getitem___178445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 37), shape_178444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1398)
    subscript_call_result_178446 = invoke(stypy.reporting.localization.Localization(__file__, 1398, 37), getitem___178445, int_178442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1398, 27), tuple_178438, subscript_call_result_178446)
    
    # Processing the call keyword arguments (line 1398)
    # Getting the type of 'c' (line 1398)
    c_178447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 56), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1398)
    dtype_178448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 56), c_178447, 'dtype')
    keyword_178449 = dtype_178448
    kwargs_178450 = {'dtype': keyword_178449}
    # Getting the type of 'np' (line 1398)
    np_178436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1398)
    zeros_178437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 17), np_178436, 'zeros')
    # Calling zeros(args, kwargs) (line 1398)
    zeros_call_result_178451 = invoke(stypy.reporting.localization.Localization(__file__, 1398, 17), zeros_178437, *[tuple_178438], **kwargs_178450)
    
    # Assigning a type to the variable 'cc' (line 1398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1398, 12), 'cc', zeros_call_result_178451)
    # SSA branch for the else part of an if statement (line 1397)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1400):
    
    # Assigning a Call to a Name (line 1400):
    
    # Call to zeros(...): (line 1400)
    # Processing the call arguments (line 1400)
    # Getting the type of 'lmax' (line 1400)
    lmax_178454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 26), 'lmax', False)
    int_178455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1400, 33), 'int')
    # Applying the binary operator '+' (line 1400)
    result_add_178456 = python_operator(stypy.reporting.localization.Localization(__file__, 1400, 26), '+', lmax_178454, int_178455)
    
    # Processing the call keyword arguments (line 1400)
    # Getting the type of 'c' (line 1400)
    c_178457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 42), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1400)
    dtype_178458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 42), c_178457, 'dtype')
    keyword_178459 = dtype_178458
    kwargs_178460 = {'dtype': keyword_178459}
    # Getting the type of 'np' (line 1400)
    np_178452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1400)
    zeros_178453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 17), np_178452, 'zeros')
    # Calling zeros(args, kwargs) (line 1400)
    zeros_call_result_178461 = invoke(stypy.reporting.localization.Localization(__file__, 1400, 17), zeros_178453, *[result_add_178456], **kwargs_178460)
    
    # Assigning a type to the variable 'cc' (line 1400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1400, 12), 'cc', zeros_call_result_178461)
    # SSA join for if statement (line 1397)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1401):
    
    # Assigning a Name to a Subscript (line 1401):
    # Getting the type of 'c' (line 1401)
    c_178462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 18), 'c')
    # Getting the type of 'cc' (line 1401)
    cc_178463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 8), 'cc')
    # Getting the type of 'deg' (line 1401)
    deg_178464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 11), 'deg')
    # Storing an element on a container (line 1401)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1401, 8), cc_178463, (deg_178464, c_178462))
    
    # Assigning a Name to a Name (line 1402):
    
    # Assigning a Name to a Name (line 1402):
    # Getting the type of 'cc' (line 1402)
    cc_178465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1402, 12), 'cc')
    # Assigning a type to the variable 'c' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'c', cc_178465)
    # SSA join for if statement (line 1396)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1405)
    rank_178466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 7), 'rank')
    # Getting the type of 'order' (line 1405)
    order_178467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 15), 'order')
    # Applying the binary operator '!=' (line 1405)
    result_ne_178468 = python_operator(stypy.reporting.localization.Localization(__file__, 1405, 7), '!=', rank_178466, order_178467)
    
    
    # Getting the type of 'full' (line 1405)
    full_178469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 29), 'full')
    # Applying the 'not' unary operator (line 1405)
    result_not__178470 = python_operator(stypy.reporting.localization.Localization(__file__, 1405, 25), 'not', full_178469)
    
    # Applying the binary operator 'and' (line 1405)
    result_and_keyword_178471 = python_operator(stypy.reporting.localization.Localization(__file__, 1405, 7), 'and', result_ne_178468, result_not__178470)
    
    # Testing the type of an if condition (line 1405)
    if_condition_178472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1405, 4), result_and_keyword_178471)
    # Assigning a type to the variable 'if_condition_178472' (line 1405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1405, 4), 'if_condition_178472', if_condition_178472)
    # SSA begins for if statement (line 1405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1406):
    
    # Assigning a Str to a Name (line 1406):
    str_178473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1406, 14), 'str', 'The fit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 1406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1406, 8), 'msg', str_178473)
    
    # Call to warn(...): (line 1407)
    # Processing the call arguments (line 1407)
    # Getting the type of 'msg' (line 1407)
    msg_178476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 22), 'msg', False)
    # Getting the type of 'pu' (line 1407)
    pu_178477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 27), 'pu', False)
    # Obtaining the member 'RankWarning' of a type (line 1407)
    RankWarning_178478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1407, 27), pu_178477, 'RankWarning')
    # Processing the call keyword arguments (line 1407)
    kwargs_178479 = {}
    # Getting the type of 'warnings' (line 1407)
    warnings_178474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1407)
    warn_178475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1407, 8), warnings_178474, 'warn')
    # Calling warn(args, kwargs) (line 1407)
    warn_call_result_178480 = invoke(stypy.reporting.localization.Localization(__file__, 1407, 8), warn_178475, *[msg_178476, RankWarning_178478], **kwargs_178479)
    
    # SSA join for if statement (line 1405)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 1409)
    full_178481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 7), 'full')
    # Testing the type of an if condition (line 1409)
    if_condition_178482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1409, 4), full_178481)
    # Assigning a type to the variable 'if_condition_178482' (line 1409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1409, 4), 'if_condition_178482', if_condition_178482)
    # SSA begins for if statement (line 1409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1410)
    tuple_178483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1410, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1410)
    # Adding element type (line 1410)
    # Getting the type of 'c' (line 1410)
    c_178484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1410, 15), tuple_178483, c_178484)
    # Adding element type (line 1410)
    
    # Obtaining an instance of the builtin type 'list' (line 1410)
    list_178485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1410, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1410)
    # Adding element type (line 1410)
    # Getting the type of 'resids' (line 1410)
    resids_178486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 19), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1410, 18), list_178485, resids_178486)
    # Adding element type (line 1410)
    # Getting the type of 'rank' (line 1410)
    rank_178487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 27), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1410, 18), list_178485, rank_178487)
    # Adding element type (line 1410)
    # Getting the type of 's' (line 1410)
    s_178488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 33), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1410, 18), list_178485, s_178488)
    # Adding element type (line 1410)
    # Getting the type of 'rcond' (line 1410)
    rcond_178489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 36), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1410, 18), list_178485, rcond_178489)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1410, 15), tuple_178483, list_178485)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1410, 8), 'stypy_return_type', tuple_178483)
    # SSA branch for the else part of an if statement (line 1409)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 1412)
    c_178490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1412, 8), 'stypy_return_type', c_178490)
    # SSA join for if statement (line 1409)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polyfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyfit' in the type store
    # Getting the type of 'stypy_return_type' (line 1196)
    stypy_return_type_178491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_178491)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyfit'
    return stypy_return_type_178491

# Assigning a type to the variable 'polyfit' (line 1196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 0), 'polyfit', polyfit)

@norecursion
def polycompanion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polycompanion'
    module_type_store = module_type_store.open_function_context('polycompanion', 1415, 0, False)
    
    # Passed parameters checking function
    polycompanion.stypy_localization = localization
    polycompanion.stypy_type_of_self = None
    polycompanion.stypy_type_store = module_type_store
    polycompanion.stypy_function_name = 'polycompanion'
    polycompanion.stypy_param_names_list = ['c']
    polycompanion.stypy_varargs_param_name = None
    polycompanion.stypy_kwargs_param_name = None
    polycompanion.stypy_call_defaults = defaults
    polycompanion.stypy_call_varargs = varargs
    polycompanion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polycompanion', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polycompanion', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polycompanion(...)' code ##################

    str_178492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1439, (-1)), 'str', '\n    Return the companion matrix of c.\n\n    The companion matrix for power series cannot be made symmetric by\n    scaling the basis, so this function differs from those for the\n    orthogonal polynomials.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of polynomial coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ')
    
    # Assigning a Call to a List (line 1441):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1441)
    # Processing the call arguments (line 1441)
    
    # Obtaining an instance of the builtin type 'list' (line 1441)
    list_178495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1441, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1441)
    # Adding element type (line 1441)
    # Getting the type of 'c' (line 1441)
    c_178496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1441, 23), list_178495, c_178496)
    
    # Processing the call keyword arguments (line 1441)
    kwargs_178497 = {}
    # Getting the type of 'pu' (line 1441)
    pu_178493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1441)
    as_series_178494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1441, 10), pu_178493, 'as_series')
    # Calling as_series(args, kwargs) (line 1441)
    as_series_call_result_178498 = invoke(stypy.reporting.localization.Localization(__file__, 1441, 10), as_series_178494, *[list_178495], **kwargs_178497)
    
    # Assigning a type to the variable 'call_assignment_176527' (line 1441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1441, 4), 'call_assignment_176527', as_series_call_result_178498)
    
    # Assigning a Call to a Name (line 1441):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1441, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178502 = {}
    # Getting the type of 'call_assignment_176527' (line 1441)
    call_assignment_176527_178499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 4), 'call_assignment_176527', False)
    # Obtaining the member '__getitem__' of a type (line 1441)
    getitem___178500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1441, 4), call_assignment_176527_178499, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178503 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178500, *[int_178501], **kwargs_178502)
    
    # Assigning a type to the variable 'call_assignment_176528' (line 1441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1441, 4), 'call_assignment_176528', getitem___call_result_178503)
    
    # Assigning a Name to a Name (line 1441):
    # Getting the type of 'call_assignment_176528' (line 1441)
    call_assignment_176528_178504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 4), 'call_assignment_176528')
    # Assigning a type to the variable 'c' (line 1441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1441, 5), 'c', call_assignment_176528_178504)
    
    
    
    # Call to len(...): (line 1442)
    # Processing the call arguments (line 1442)
    # Getting the type of 'c' (line 1442)
    c_178506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1442, 11), 'c', False)
    # Processing the call keyword arguments (line 1442)
    kwargs_178507 = {}
    # Getting the type of 'len' (line 1442)
    len_178505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1442, 7), 'len', False)
    # Calling len(args, kwargs) (line 1442)
    len_call_result_178508 = invoke(stypy.reporting.localization.Localization(__file__, 1442, 7), len_178505, *[c_178506], **kwargs_178507)
    
    int_178509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1442, 16), 'int')
    # Applying the binary operator '<' (line 1442)
    result_lt_178510 = python_operator(stypy.reporting.localization.Localization(__file__, 1442, 7), '<', len_call_result_178508, int_178509)
    
    # Testing the type of an if condition (line 1442)
    if_condition_178511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1442, 4), result_lt_178510)
    # Assigning a type to the variable 'if_condition_178511' (line 1442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1442, 4), 'if_condition_178511', if_condition_178511)
    # SSA begins for if statement (line 1442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1443)
    # Processing the call arguments (line 1443)
    str_178513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1443, 25), 'str', 'Series must have maximum degree of at least 1.')
    # Processing the call keyword arguments (line 1443)
    kwargs_178514 = {}
    # Getting the type of 'ValueError' (line 1443)
    ValueError_178512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1443, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1443)
    ValueError_call_result_178515 = invoke(stypy.reporting.localization.Localization(__file__, 1443, 14), ValueError_178512, *[str_178513], **kwargs_178514)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1443, 8), ValueError_call_result_178515, 'raise parameter', BaseException)
    # SSA join for if statement (line 1442)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1444)
    # Processing the call arguments (line 1444)
    # Getting the type of 'c' (line 1444)
    c_178517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1444, 11), 'c', False)
    # Processing the call keyword arguments (line 1444)
    kwargs_178518 = {}
    # Getting the type of 'len' (line 1444)
    len_178516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1444, 7), 'len', False)
    # Calling len(args, kwargs) (line 1444)
    len_call_result_178519 = invoke(stypy.reporting.localization.Localization(__file__, 1444, 7), len_178516, *[c_178517], **kwargs_178518)
    
    int_178520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1444, 17), 'int')
    # Applying the binary operator '==' (line 1444)
    result_eq_178521 = python_operator(stypy.reporting.localization.Localization(__file__, 1444, 7), '==', len_call_result_178519, int_178520)
    
    # Testing the type of an if condition (line 1444)
    if_condition_178522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1444, 4), result_eq_178521)
    # Assigning a type to the variable 'if_condition_178522' (line 1444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1444, 4), 'if_condition_178522', if_condition_178522)
    # SSA begins for if statement (line 1444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1445)
    # Processing the call arguments (line 1445)
    
    # Obtaining an instance of the builtin type 'list' (line 1445)
    list_178525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1445)
    # Adding element type (line 1445)
    
    # Obtaining an instance of the builtin type 'list' (line 1445)
    list_178526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1445)
    # Adding element type (line 1445)
    
    
    # Obtaining the type of the subscript
    int_178527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 29), 'int')
    # Getting the type of 'c' (line 1445)
    c_178528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 27), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1445)
    getitem___178529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 27), c_178528, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1445)
    subscript_call_result_178530 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 27), getitem___178529, int_178527)
    
    # Applying the 'usub' unary operator (line 1445)
    result___neg___178531 = python_operator(stypy.reporting.localization.Localization(__file__, 1445, 26), 'usub', subscript_call_result_178530)
    
    
    # Obtaining the type of the subscript
    int_178532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 34), 'int')
    # Getting the type of 'c' (line 1445)
    c_178533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 32), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1445)
    getitem___178534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 32), c_178533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1445)
    subscript_call_result_178535 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 32), getitem___178534, int_178532)
    
    # Applying the binary operator 'div' (line 1445)
    result_div_178536 = python_operator(stypy.reporting.localization.Localization(__file__, 1445, 26), 'div', result___neg___178531, subscript_call_result_178535)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1445, 25), list_178526, result_div_178536)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1445, 24), list_178525, list_178526)
    
    # Processing the call keyword arguments (line 1445)
    kwargs_178537 = {}
    # Getting the type of 'np' (line 1445)
    np_178523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1445)
    array_178524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 15), np_178523, 'array')
    # Calling array(args, kwargs) (line 1445)
    array_call_result_178538 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 15), array_178524, *[list_178525], **kwargs_178537)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 8), 'stypy_return_type', array_call_result_178538)
    # SSA join for if statement (line 1444)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1447):
    
    # Assigning a BinOp to a Name (line 1447):
    
    # Call to len(...): (line 1447)
    # Processing the call arguments (line 1447)
    # Getting the type of 'c' (line 1447)
    c_178540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1447, 12), 'c', False)
    # Processing the call keyword arguments (line 1447)
    kwargs_178541 = {}
    # Getting the type of 'len' (line 1447)
    len_178539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1447, 8), 'len', False)
    # Calling len(args, kwargs) (line 1447)
    len_call_result_178542 = invoke(stypy.reporting.localization.Localization(__file__, 1447, 8), len_178539, *[c_178540], **kwargs_178541)
    
    int_178543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1447, 17), 'int')
    # Applying the binary operator '-' (line 1447)
    result_sub_178544 = python_operator(stypy.reporting.localization.Localization(__file__, 1447, 8), '-', len_call_result_178542, int_178543)
    
    # Assigning a type to the variable 'n' (line 1447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1447, 4), 'n', result_sub_178544)
    
    # Assigning a Call to a Name (line 1448):
    
    # Assigning a Call to a Name (line 1448):
    
    # Call to zeros(...): (line 1448)
    # Processing the call arguments (line 1448)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1448)
    tuple_178547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1448, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1448)
    # Adding element type (line 1448)
    # Getting the type of 'n' (line 1448)
    n_178548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1448, 20), tuple_178547, n_178548)
    # Adding element type (line 1448)
    # Getting the type of 'n' (line 1448)
    n_178549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1448, 20), tuple_178547, n_178549)
    
    # Processing the call keyword arguments (line 1448)
    # Getting the type of 'c' (line 1448)
    c_178550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 33), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1448)
    dtype_178551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1448, 33), c_178550, 'dtype')
    keyword_178552 = dtype_178551
    kwargs_178553 = {'dtype': keyword_178552}
    # Getting the type of 'np' (line 1448)
    np_178545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1448)
    zeros_178546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1448, 10), np_178545, 'zeros')
    # Calling zeros(args, kwargs) (line 1448)
    zeros_call_result_178554 = invoke(stypy.reporting.localization.Localization(__file__, 1448, 10), zeros_178546, *[tuple_178547], **kwargs_178553)
    
    # Assigning a type to the variable 'mat' (line 1448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1448, 4), 'mat', zeros_call_result_178554)
    
    # Assigning a Subscript to a Name (line 1449):
    
    # Assigning a Subscript to a Name (line 1449):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1449)
    n_178555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 26), 'n')
    # Getting the type of 'n' (line 1449)
    n_178556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 29), 'n')
    int_178557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1449, 31), 'int')
    # Applying the binary operator '+' (line 1449)
    result_add_178558 = python_operator(stypy.reporting.localization.Localization(__file__, 1449, 29), '+', n_178556, int_178557)
    
    slice_178559 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1449, 10), n_178555, None, result_add_178558)
    
    # Call to reshape(...): (line 1449)
    # Processing the call arguments (line 1449)
    int_178562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1449, 22), 'int')
    # Processing the call keyword arguments (line 1449)
    kwargs_178563 = {}
    # Getting the type of 'mat' (line 1449)
    mat_178560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1449)
    reshape_178561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1449, 10), mat_178560, 'reshape')
    # Calling reshape(args, kwargs) (line 1449)
    reshape_call_result_178564 = invoke(stypy.reporting.localization.Localization(__file__, 1449, 10), reshape_178561, *[int_178562], **kwargs_178563)
    
    # Obtaining the member '__getitem__' of a type (line 1449)
    getitem___178565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1449, 10), reshape_call_result_178564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1449)
    subscript_call_result_178566 = invoke(stypy.reporting.localization.Localization(__file__, 1449, 10), getitem___178565, slice_178559)
    
    # Assigning a type to the variable 'bot' (line 1449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1449, 4), 'bot', subscript_call_result_178566)
    
    # Assigning a Num to a Subscript (line 1450):
    
    # Assigning a Num to a Subscript (line 1450):
    int_178567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1450, 15), 'int')
    # Getting the type of 'bot' (line 1450)
    bot_178568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 4), 'bot')
    Ellipsis_178569 = Ellipsis
    # Storing an element on a container (line 1450)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1450, 4), bot_178568, (Ellipsis_178569, int_178567))
    
    # Getting the type of 'mat' (line 1451)
    mat_178570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'mat')
    
    # Obtaining the type of the subscript
    slice_178571 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1451, 4), None, None, None)
    int_178572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 11), 'int')
    # Getting the type of 'mat' (line 1451)
    mat_178573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'mat')
    # Obtaining the member '__getitem__' of a type (line 1451)
    getitem___178574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 4), mat_178573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1451)
    subscript_call_result_178575 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 4), getitem___178574, (slice_178571, int_178572))
    
    
    # Obtaining the type of the subscript
    int_178576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 21), 'int')
    slice_178577 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1451, 18), None, int_178576, None)
    # Getting the type of 'c' (line 1451)
    c_178578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 18), 'c')
    # Obtaining the member '__getitem__' of a type (line 1451)
    getitem___178579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 18), c_178578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1451)
    subscript_call_result_178580 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 18), getitem___178579, slice_178577)
    
    
    # Obtaining the type of the subscript
    int_178581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 27), 'int')
    # Getting the type of 'c' (line 1451)
    c_178582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 25), 'c')
    # Obtaining the member '__getitem__' of a type (line 1451)
    getitem___178583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 25), c_178582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1451)
    subscript_call_result_178584 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 25), getitem___178583, int_178581)
    
    # Applying the binary operator 'div' (line 1451)
    result_div_178585 = python_operator(stypy.reporting.localization.Localization(__file__, 1451, 18), 'div', subscript_call_result_178580, subscript_call_result_178584)
    
    # Applying the binary operator '-=' (line 1451)
    result_isub_178586 = python_operator(stypy.reporting.localization.Localization(__file__, 1451, 4), '-=', subscript_call_result_178575, result_div_178585)
    # Getting the type of 'mat' (line 1451)
    mat_178587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'mat')
    slice_178588 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1451, 4), None, None, None)
    int_178589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 11), 'int')
    # Storing an element on a container (line 1451)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1451, 4), mat_178587, ((slice_178588, int_178589), result_isub_178586))
    
    # Getting the type of 'mat' (line 1452)
    mat_178590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1452, 11), 'mat')
    # Assigning a type to the variable 'stypy_return_type' (line 1452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1452, 4), 'stypy_return_type', mat_178590)
    
    # ################# End of 'polycompanion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polycompanion' in the type store
    # Getting the type of 'stypy_return_type' (line 1415)
    stypy_return_type_178591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_178591)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polycompanion'
    return stypy_return_type_178591

# Assigning a type to the variable 'polycompanion' (line 1415)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1415, 0), 'polycompanion', polycompanion)

@norecursion
def polyroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyroots'
    module_type_store = module_type_store.open_function_context('polyroots', 1455, 0, False)
    
    # Passed parameters checking function
    polyroots.stypy_localization = localization
    polyroots.stypy_type_of_self = None
    polyroots.stypy_type_store = module_type_store
    polyroots.stypy_function_name = 'polyroots'
    polyroots.stypy_param_names_list = ['c']
    polyroots.stypy_varargs_param_name = None
    polyroots.stypy_kwargs_param_name = None
    polyroots.stypy_call_defaults = defaults
    polyroots.stypy_call_varargs = varargs
    polyroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyroots', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyroots', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyroots(...)' code ##################

    str_178592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1499, (-1)), 'str', '\n    Compute the roots of a polynomial.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * x^i.\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of polynomial coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the polynomial. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    chebroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the power series for such\n    values. Roots with multiplicity greater than 1 will also show larger\n    errors as the value of the series near such points is relatively\n    insensitive to errors in the roots. Isolated roots near the origin can\n    be improved by a few iterations of Newton\'s method.\n\n    Examples\n    --------\n    >>> import numpy.polynomial.polynomial as poly\n    >>> poly.polyroots(poly.polyfromroots((-1,0,1)))\n    array([-1.,  0.,  1.])\n    >>> poly.polyroots(poly.polyfromroots((-1,0,1))).dtype\n    dtype(\'float64\')\n    >>> j = complex(0,1)\n    >>> poly.polyroots(poly.polyfromroots((-j,0,j)))\n    array([  0.00000000e+00+0.j,   0.00000000e+00+1.j,   2.77555756e-17-1.j])\n\n    ')
    
    # Assigning a Call to a List (line 1501):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1501)
    # Processing the call arguments (line 1501)
    
    # Obtaining an instance of the builtin type 'list' (line 1501)
    list_178595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1501, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1501)
    # Adding element type (line 1501)
    # Getting the type of 'c' (line 1501)
    c_178596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1501, 23), list_178595, c_178596)
    
    # Processing the call keyword arguments (line 1501)
    kwargs_178597 = {}
    # Getting the type of 'pu' (line 1501)
    pu_178593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1501)
    as_series_178594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1501, 10), pu_178593, 'as_series')
    # Calling as_series(args, kwargs) (line 1501)
    as_series_call_result_178598 = invoke(stypy.reporting.localization.Localization(__file__, 1501, 10), as_series_178594, *[list_178595], **kwargs_178597)
    
    # Assigning a type to the variable 'call_assignment_176529' (line 1501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1501, 4), 'call_assignment_176529', as_series_call_result_178598)
    
    # Assigning a Call to a Name (line 1501):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_178601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1501, 4), 'int')
    # Processing the call keyword arguments
    kwargs_178602 = {}
    # Getting the type of 'call_assignment_176529' (line 1501)
    call_assignment_176529_178599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 4), 'call_assignment_176529', False)
    # Obtaining the member '__getitem__' of a type (line 1501)
    getitem___178600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1501, 4), call_assignment_176529_178599, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_178603 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178600, *[int_178601], **kwargs_178602)
    
    # Assigning a type to the variable 'call_assignment_176530' (line 1501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1501, 4), 'call_assignment_176530', getitem___call_result_178603)
    
    # Assigning a Name to a Name (line 1501):
    # Getting the type of 'call_assignment_176530' (line 1501)
    call_assignment_176530_178604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 4), 'call_assignment_176530')
    # Assigning a type to the variable 'c' (line 1501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1501, 5), 'c', call_assignment_176530_178604)
    
    
    
    # Call to len(...): (line 1502)
    # Processing the call arguments (line 1502)
    # Getting the type of 'c' (line 1502)
    c_178606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 11), 'c', False)
    # Processing the call keyword arguments (line 1502)
    kwargs_178607 = {}
    # Getting the type of 'len' (line 1502)
    len_178605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 7), 'len', False)
    # Calling len(args, kwargs) (line 1502)
    len_call_result_178608 = invoke(stypy.reporting.localization.Localization(__file__, 1502, 7), len_178605, *[c_178606], **kwargs_178607)
    
    int_178609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 16), 'int')
    # Applying the binary operator '<' (line 1502)
    result_lt_178610 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 7), '<', len_call_result_178608, int_178609)
    
    # Testing the type of an if condition (line 1502)
    if_condition_178611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1502, 4), result_lt_178610)
    # Assigning a type to the variable 'if_condition_178611' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 4), 'if_condition_178611', if_condition_178611)
    # SSA begins for if statement (line 1502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1503)
    # Processing the call arguments (line 1503)
    
    # Obtaining an instance of the builtin type 'list' (line 1503)
    list_178614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1503, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1503)
    
    # Processing the call keyword arguments (line 1503)
    # Getting the type of 'c' (line 1503)
    c_178615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 34), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1503)
    dtype_178616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1503, 34), c_178615, 'dtype')
    keyword_178617 = dtype_178616
    kwargs_178618 = {'dtype': keyword_178617}
    # Getting the type of 'np' (line 1503)
    np_178612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1503)
    array_178613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1503, 15), np_178612, 'array')
    # Calling array(args, kwargs) (line 1503)
    array_call_result_178619 = invoke(stypy.reporting.localization.Localization(__file__, 1503, 15), array_178613, *[list_178614], **kwargs_178618)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1503, 8), 'stypy_return_type', array_call_result_178619)
    # SSA join for if statement (line 1502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1504)
    # Processing the call arguments (line 1504)
    # Getting the type of 'c' (line 1504)
    c_178621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 11), 'c', False)
    # Processing the call keyword arguments (line 1504)
    kwargs_178622 = {}
    # Getting the type of 'len' (line 1504)
    len_178620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 7), 'len', False)
    # Calling len(args, kwargs) (line 1504)
    len_call_result_178623 = invoke(stypy.reporting.localization.Localization(__file__, 1504, 7), len_178620, *[c_178621], **kwargs_178622)
    
    int_178624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1504, 17), 'int')
    # Applying the binary operator '==' (line 1504)
    result_eq_178625 = python_operator(stypy.reporting.localization.Localization(__file__, 1504, 7), '==', len_call_result_178623, int_178624)
    
    # Testing the type of an if condition (line 1504)
    if_condition_178626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1504, 4), result_eq_178625)
    # Assigning a type to the variable 'if_condition_178626' (line 1504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1504, 4), 'if_condition_178626', if_condition_178626)
    # SSA begins for if statement (line 1504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1505)
    # Processing the call arguments (line 1505)
    
    # Obtaining an instance of the builtin type 'list' (line 1505)
    list_178629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1505)
    # Adding element type (line 1505)
    
    
    # Obtaining the type of the subscript
    int_178630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 28), 'int')
    # Getting the type of 'c' (line 1505)
    c_178631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 26), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1505)
    getitem___178632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1505, 26), c_178631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1505)
    subscript_call_result_178633 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 26), getitem___178632, int_178630)
    
    # Applying the 'usub' unary operator (line 1505)
    result___neg___178634 = python_operator(stypy.reporting.localization.Localization(__file__, 1505, 25), 'usub', subscript_call_result_178633)
    
    
    # Obtaining the type of the subscript
    int_178635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 33), 'int')
    # Getting the type of 'c' (line 1505)
    c_178636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 31), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1505)
    getitem___178637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1505, 31), c_178636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1505)
    subscript_call_result_178638 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 31), getitem___178637, int_178635)
    
    # Applying the binary operator 'div' (line 1505)
    result_div_178639 = python_operator(stypy.reporting.localization.Localization(__file__, 1505, 25), 'div', result___neg___178634, subscript_call_result_178638)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1505, 24), list_178629, result_div_178639)
    
    # Processing the call keyword arguments (line 1505)
    kwargs_178640 = {}
    # Getting the type of 'np' (line 1505)
    np_178627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1505)
    array_178628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1505, 15), np_178627, 'array')
    # Calling array(args, kwargs) (line 1505)
    array_call_result_178641 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 15), array_178628, *[list_178629], **kwargs_178640)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1505, 8), 'stypy_return_type', array_call_result_178641)
    # SSA join for if statement (line 1504)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1507):
    
    # Assigning a Call to a Name (line 1507):
    
    # Call to polycompanion(...): (line 1507)
    # Processing the call arguments (line 1507)
    # Getting the type of 'c' (line 1507)
    c_178643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 22), 'c', False)
    # Processing the call keyword arguments (line 1507)
    kwargs_178644 = {}
    # Getting the type of 'polycompanion' (line 1507)
    polycompanion_178642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 8), 'polycompanion', False)
    # Calling polycompanion(args, kwargs) (line 1507)
    polycompanion_call_result_178645 = invoke(stypy.reporting.localization.Localization(__file__, 1507, 8), polycompanion_178642, *[c_178643], **kwargs_178644)
    
    # Assigning a type to the variable 'm' (line 1507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1507, 4), 'm', polycompanion_call_result_178645)
    
    # Assigning a Call to a Name (line 1508):
    
    # Assigning a Call to a Name (line 1508):
    
    # Call to eigvals(...): (line 1508)
    # Processing the call arguments (line 1508)
    # Getting the type of 'm' (line 1508)
    m_178648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 19), 'm', False)
    # Processing the call keyword arguments (line 1508)
    kwargs_178649 = {}
    # Getting the type of 'la' (line 1508)
    la_178646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 8), 'la', False)
    # Obtaining the member 'eigvals' of a type (line 1508)
    eigvals_178647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1508, 8), la_178646, 'eigvals')
    # Calling eigvals(args, kwargs) (line 1508)
    eigvals_call_result_178650 = invoke(stypy.reporting.localization.Localization(__file__, 1508, 8), eigvals_178647, *[m_178648], **kwargs_178649)
    
    # Assigning a type to the variable 'r' (line 1508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1508, 4), 'r', eigvals_call_result_178650)
    
    # Call to sort(...): (line 1509)
    # Processing the call keyword arguments (line 1509)
    kwargs_178653 = {}
    # Getting the type of 'r' (line 1509)
    r_178651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1509, 4), 'r', False)
    # Obtaining the member 'sort' of a type (line 1509)
    sort_178652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1509, 4), r_178651, 'sort')
    # Calling sort(args, kwargs) (line 1509)
    sort_call_result_178654 = invoke(stypy.reporting.localization.Localization(__file__, 1509, 4), sort_178652, *[], **kwargs_178653)
    
    # Getting the type of 'r' (line 1510)
    r_178655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'stypy_return_type', r_178655)
    
    # ################# End of 'polyroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyroots' in the type store
    # Getting the type of 'stypy_return_type' (line 1455)
    stypy_return_type_178656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_178656)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyroots'
    return stypy_return_type_178656

# Assigning a type to the variable 'polyroots' (line 1455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1455, 0), 'polyroots', polyroots)
# Declaration of the 'Polynomial' class
# Getting the type of 'ABCPolyBase' (line 1517)
ABCPolyBase_178657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 17), 'ABCPolyBase')

class Polynomial(ABCPolyBase_178657, ):
    str_178658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1538, (-1)), 'str', "A power series class.\n\n    The Polynomial class provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the\n    attributes and methods listed in the `ABCPolyBase` documentation.\n\n    Parameters\n    ----------\n    coef : array_like\n        Polynomial coefficients in order of increasing degree, i.e.,\n        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is [-1, 1].\n    window : (2,) array_like, optional\n        Window, see `domain` for its use. The default value is [-1, 1].\n\n        .. versionadded:: 1.6.0\n\n    ")
    
    # Assigning a Call to a Name (line 1540):
    
    # Assigning a Call to a Name (line 1541):
    
    # Assigning a Call to a Name (line 1542):
    
    # Assigning a Call to a Name (line 1543):
    
    # Assigning a Call to a Name (line 1544):
    
    # Assigning a Call to a Name (line 1545):
    
    # Assigning a Call to a Name (line 1546):
    
    # Assigning a Call to a Name (line 1547):
    
    # Assigning a Call to a Name (line 1548):
    
    # Assigning a Call to a Name (line 1549):
    
    # Assigning a Call to a Name (line 1550):
    
    # Assigning a Call to a Name (line 1551):
    
    # Assigning a Str to a Name (line 1554):
    
    # Assigning a Call to a Name (line 1555):
    
    # Assigning a Call to a Name (line 1556):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1517, 0, False)
        # Assigning a type to the variable 'self' (line 1518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1518, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Polynomial.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Polynomial' (line 1517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1517, 0), 'Polynomial', Polynomial)

# Assigning a Call to a Name (line 1540):

# Call to staticmethod(...): (line 1540)
# Processing the call arguments (line 1540)
# Getting the type of 'polyadd' (line 1540)
polyadd_178660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 24), 'polyadd', False)
# Processing the call keyword arguments (line 1540)
kwargs_178661 = {}
# Getting the type of 'staticmethod' (line 1540)
staticmethod_178659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1540)
staticmethod_call_result_178662 = invoke(stypy.reporting.localization.Localization(__file__, 1540, 11), staticmethod_178659, *[polyadd_178660], **kwargs_178661)

# Getting the type of 'Polynomial'
Polynomial_178663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178663, '_add', staticmethod_call_result_178662)

# Assigning a Call to a Name (line 1541):

# Call to staticmethod(...): (line 1541)
# Processing the call arguments (line 1541)
# Getting the type of 'polysub' (line 1541)
polysub_178665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 24), 'polysub', False)
# Processing the call keyword arguments (line 1541)
kwargs_178666 = {}
# Getting the type of 'staticmethod' (line 1541)
staticmethod_178664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1541)
staticmethod_call_result_178667 = invoke(stypy.reporting.localization.Localization(__file__, 1541, 11), staticmethod_178664, *[polysub_178665], **kwargs_178666)

# Getting the type of 'Polynomial'
Polynomial_178668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_sub' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178668, '_sub', staticmethod_call_result_178667)

# Assigning a Call to a Name (line 1542):

# Call to staticmethod(...): (line 1542)
# Processing the call arguments (line 1542)
# Getting the type of 'polymul' (line 1542)
polymul_178670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 24), 'polymul', False)
# Processing the call keyword arguments (line 1542)
kwargs_178671 = {}
# Getting the type of 'staticmethod' (line 1542)
staticmethod_178669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1542)
staticmethod_call_result_178672 = invoke(stypy.reporting.localization.Localization(__file__, 1542, 11), staticmethod_178669, *[polymul_178670], **kwargs_178671)

# Getting the type of 'Polynomial'
Polynomial_178673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_mul' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178673, '_mul', staticmethod_call_result_178672)

# Assigning a Call to a Name (line 1543):

# Call to staticmethod(...): (line 1543)
# Processing the call arguments (line 1543)
# Getting the type of 'polydiv' (line 1543)
polydiv_178675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 24), 'polydiv', False)
# Processing the call keyword arguments (line 1543)
kwargs_178676 = {}
# Getting the type of 'staticmethod' (line 1543)
staticmethod_178674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1543)
staticmethod_call_result_178677 = invoke(stypy.reporting.localization.Localization(__file__, 1543, 11), staticmethod_178674, *[polydiv_178675], **kwargs_178676)

# Getting the type of 'Polynomial'
Polynomial_178678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_div' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178678, '_div', staticmethod_call_result_178677)

# Assigning a Call to a Name (line 1544):

# Call to staticmethod(...): (line 1544)
# Processing the call arguments (line 1544)
# Getting the type of 'polypow' (line 1544)
polypow_178680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 24), 'polypow', False)
# Processing the call keyword arguments (line 1544)
kwargs_178681 = {}
# Getting the type of 'staticmethod' (line 1544)
staticmethod_178679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1544)
staticmethod_call_result_178682 = invoke(stypy.reporting.localization.Localization(__file__, 1544, 11), staticmethod_178679, *[polypow_178680], **kwargs_178681)

# Getting the type of 'Polynomial'
Polynomial_178683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_pow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178683, '_pow', staticmethod_call_result_178682)

# Assigning a Call to a Name (line 1545):

# Call to staticmethod(...): (line 1545)
# Processing the call arguments (line 1545)
# Getting the type of 'polyval' (line 1545)
polyval_178685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 24), 'polyval', False)
# Processing the call keyword arguments (line 1545)
kwargs_178686 = {}
# Getting the type of 'staticmethod' (line 1545)
staticmethod_178684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1545)
staticmethod_call_result_178687 = invoke(stypy.reporting.localization.Localization(__file__, 1545, 11), staticmethod_178684, *[polyval_178685], **kwargs_178686)

# Getting the type of 'Polynomial'
Polynomial_178688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_val' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178688, '_val', staticmethod_call_result_178687)

# Assigning a Call to a Name (line 1546):

# Call to staticmethod(...): (line 1546)
# Processing the call arguments (line 1546)
# Getting the type of 'polyint' (line 1546)
polyint_178690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 24), 'polyint', False)
# Processing the call keyword arguments (line 1546)
kwargs_178691 = {}
# Getting the type of 'staticmethod' (line 1546)
staticmethod_178689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1546)
staticmethod_call_result_178692 = invoke(stypy.reporting.localization.Localization(__file__, 1546, 11), staticmethod_178689, *[polyint_178690], **kwargs_178691)

# Getting the type of 'Polynomial'
Polynomial_178693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_int' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178693, '_int', staticmethod_call_result_178692)

# Assigning a Call to a Name (line 1547):

# Call to staticmethod(...): (line 1547)
# Processing the call arguments (line 1547)
# Getting the type of 'polyder' (line 1547)
polyder_178695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 24), 'polyder', False)
# Processing the call keyword arguments (line 1547)
kwargs_178696 = {}
# Getting the type of 'staticmethod' (line 1547)
staticmethod_178694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1547)
staticmethod_call_result_178697 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 11), staticmethod_178694, *[polyder_178695], **kwargs_178696)

# Getting the type of 'Polynomial'
Polynomial_178698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_der' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178698, '_der', staticmethod_call_result_178697)

# Assigning a Call to a Name (line 1548):

# Call to staticmethod(...): (line 1548)
# Processing the call arguments (line 1548)
# Getting the type of 'polyfit' (line 1548)
polyfit_178700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1548, 24), 'polyfit', False)
# Processing the call keyword arguments (line 1548)
kwargs_178701 = {}
# Getting the type of 'staticmethod' (line 1548)
staticmethod_178699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1548, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1548)
staticmethod_call_result_178702 = invoke(stypy.reporting.localization.Localization(__file__, 1548, 11), staticmethod_178699, *[polyfit_178700], **kwargs_178701)

# Getting the type of 'Polynomial'
Polynomial_178703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_fit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178703, '_fit', staticmethod_call_result_178702)

# Assigning a Call to a Name (line 1549):

# Call to staticmethod(...): (line 1549)
# Processing the call arguments (line 1549)
# Getting the type of 'polyline' (line 1549)
polyline_178705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 25), 'polyline', False)
# Processing the call keyword arguments (line 1549)
kwargs_178706 = {}
# Getting the type of 'staticmethod' (line 1549)
staticmethod_178704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 12), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1549)
staticmethod_call_result_178707 = invoke(stypy.reporting.localization.Localization(__file__, 1549, 12), staticmethod_178704, *[polyline_178705], **kwargs_178706)

# Getting the type of 'Polynomial'
Polynomial_178708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_line' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178708, '_line', staticmethod_call_result_178707)

# Assigning a Call to a Name (line 1550):

# Call to staticmethod(...): (line 1550)
# Processing the call arguments (line 1550)
# Getting the type of 'polyroots' (line 1550)
polyroots_178710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 26), 'polyroots', False)
# Processing the call keyword arguments (line 1550)
kwargs_178711 = {}
# Getting the type of 'staticmethod' (line 1550)
staticmethod_178709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 13), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1550)
staticmethod_call_result_178712 = invoke(stypy.reporting.localization.Localization(__file__, 1550, 13), staticmethod_178709, *[polyroots_178710], **kwargs_178711)

# Getting the type of 'Polynomial'
Polynomial_178713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_roots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178713, '_roots', staticmethod_call_result_178712)

# Assigning a Call to a Name (line 1551):

# Call to staticmethod(...): (line 1551)
# Processing the call arguments (line 1551)
# Getting the type of 'polyfromroots' (line 1551)
polyfromroots_178715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 30), 'polyfromroots', False)
# Processing the call keyword arguments (line 1551)
kwargs_178716 = {}
# Getting the type of 'staticmethod' (line 1551)
staticmethod_178714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 17), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1551)
staticmethod_call_result_178717 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 17), staticmethod_178714, *[polyfromroots_178715], **kwargs_178716)

# Getting the type of 'Polynomial'
Polynomial_178718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member '_fromroots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178718, '_fromroots', staticmethod_call_result_178717)

# Assigning a Str to a Name (line 1554):
str_178719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1554, 15), 'str', 'poly')
# Getting the type of 'Polynomial'
Polynomial_178720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member 'nickname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178720, 'nickname', str_178719)

# Assigning a Call to a Name (line 1555):

# Call to array(...): (line 1555)
# Processing the call arguments (line 1555)
# Getting the type of 'polydomain' (line 1555)
polydomain_178723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 22), 'polydomain', False)
# Processing the call keyword arguments (line 1555)
kwargs_178724 = {}
# Getting the type of 'np' (line 1555)
np_178721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1555)
array_178722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1555, 13), np_178721, 'array')
# Calling array(args, kwargs) (line 1555)
array_call_result_178725 = invoke(stypy.reporting.localization.Localization(__file__, 1555, 13), array_178722, *[polydomain_178723], **kwargs_178724)

# Getting the type of 'Polynomial'
Polynomial_178726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member 'domain' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178726, 'domain', array_call_result_178725)

# Assigning a Call to a Name (line 1556):

# Call to array(...): (line 1556)
# Processing the call arguments (line 1556)
# Getting the type of 'polydomain' (line 1556)
polydomain_178729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 22), 'polydomain', False)
# Processing the call keyword arguments (line 1556)
kwargs_178730 = {}
# Getting the type of 'np' (line 1556)
np_178727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1556)
array_178728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1556, 13), np_178727, 'array')
# Calling array(args, kwargs) (line 1556)
array_call_result_178731 = invoke(stypy.reporting.localization.Localization(__file__, 1556, 13), array_178728, *[polydomain_178729], **kwargs_178730)

# Getting the type of 'Polynomial'
Polynomial_178732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Polynomial')
# Setting the type of the member 'window' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Polynomial_178732, 'window', array_call_result_178731)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
