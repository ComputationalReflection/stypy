
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Objects for dealing with Chebyshev series.
3: 
4: This module provides a number of objects (mostly functions) useful for
5: dealing with Chebyshev series, including a `Chebyshev` class that
6: encapsulates the usual arithmetic operations.  (General information
7: on how this module represents and works with such polynomials is in the
8: docstring for its "parent" sub-package, `numpy.polynomial`).
9: 
10: Constants
11: ---------
12: - `chebdomain` -- Chebyshev series default domain, [-1,1].
13: - `chebzero` -- (Coefficients of the) Chebyshev series that evaluates
14:   identically to 0.
15: - `chebone` -- (Coefficients of the) Chebyshev series that evaluates
16:   identically to 1.
17: - `chebx` -- (Coefficients of the) Chebyshev series for the identity map,
18:   ``f(x) = x``.
19: 
20: Arithmetic
21: ----------
22: - `chebadd` -- add two Chebyshev series.
23: - `chebsub` -- subtract one Chebyshev series from another.
24: - `chebmul` -- multiply two Chebyshev series.
25: - `chebdiv` -- divide one Chebyshev series by another.
26: - `chebpow` -- raise a Chebyshev series to an positive integer power
27: - `chebval` -- evaluate a Chebyshev series at given points.
28: - `chebval2d` -- evaluate a 2D Chebyshev series at given points.
29: - `chebval3d` -- evaluate a 3D Chebyshev series at given points.
30: - `chebgrid2d` -- evaluate a 2D Chebyshev series on a Cartesian product.
31: - `chebgrid3d` -- evaluate a 3D Chebyshev series on a Cartesian product.
32: 
33: Calculus
34: --------
35: - `chebder` -- differentiate a Chebyshev series.
36: - `chebint` -- integrate a Chebyshev series.
37: 
38: Misc Functions
39: --------------
40: - `chebfromroots` -- create a Chebyshev series with specified roots.
41: - `chebroots` -- find the roots of a Chebyshev series.
42: - `chebvander` -- Vandermonde-like matrix for Chebyshev polynomials.
43: - `chebvander2d` -- Vandermonde-like matrix for 2D power series.
44: - `chebvander3d` -- Vandermonde-like matrix for 3D power series.
45: - `chebgauss` -- Gauss-Chebyshev quadrature, points and weights.
46: - `chebweight` -- Chebyshev weight function.
47: - `chebcompanion` -- symmetrized companion matrix in Chebyshev form.
48: - `chebfit` -- least-squares fit returning a Chebyshev series.
49: - `chebpts1` -- Chebyshev points of the first kind.
50: - `chebpts2` -- Chebyshev points of the second kind.
51: - `chebtrim` -- trim leading coefficients from a Chebyshev series.
52: - `chebline` -- Chebyshev series representing given straight line.
53: - `cheb2poly` -- convert a Chebyshev series to a polynomial.
54: - `poly2cheb` -- convert a polynomial to a Chebyshev series.
55: 
56: Classes
57: -------
58: - `Chebyshev` -- A Chebyshev series class.
59: 
60: See also
61: --------
62: `numpy.polynomial`
63: 
64: Notes
65: -----
66: The implementations of multiplication, division, integration, and
67: differentiation use the algebraic identities [1]_:
68: 
69: .. math ::
70:     T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
71:     z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.
72: 
73: where
74: 
75: .. math :: x = \\frac{z + z^{-1}}{2}.
76: 
77: These identities allow a Chebyshev series to be expressed as a finite,
78: symmetric Laurent series.  In this module, this sort of Laurent series
79: is referred to as a "z-series."
80: 
81: References
82: ----------
83: .. [1] A. T. Benjamin, et al., "Combinatorial Trigonometry with Chebyshev
84:   Polynomials," *Journal of Statistical Planning and Inference 14*, 2008
85:   (preprint: http://www.math.hmc.edu/~benjamin/papers/CombTrig.pdf, pg. 4)
86: 
87: '''
88: from __future__ import division, absolute_import, print_function
89: 
90: import warnings
91: import numpy as np
92: import numpy.linalg as la
93: 
94: from . import polyutils as pu
95: from ._polybase import ABCPolyBase
96: 
97: __all__ = [
98:     'chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline', 'chebadd',
99:     'chebsub', 'chebmulx', 'chebmul', 'chebdiv', 'chebpow', 'chebval',
100:     'chebder', 'chebint', 'cheb2poly', 'poly2cheb', 'chebfromroots',
101:     'chebvander', 'chebfit', 'chebtrim', 'chebroots', 'chebpts1',
102:     'chebpts2', 'Chebyshev', 'chebval2d', 'chebval3d', 'chebgrid2d',
103:     'chebgrid3d', 'chebvander2d', 'chebvander3d', 'chebcompanion',
104:     'chebgauss', 'chebweight']
105: 
106: chebtrim = pu.trimcoef
107: 
108: #
109: # A collection of functions for manipulating z-series. These are private
110: # functions and do minimal error checking.
111: #
112: 
113: def _cseries_to_zseries(c):
114:     '''Covert Chebyshev series to z-series.
115: 
116:     Covert a Chebyshev series to the equivalent z-series. The result is
117:     never an empty array. The dtype of the return is the same as that of
118:     the input. No checks are run on the arguments as this routine is for
119:     internal use.
120: 
121:     Parameters
122:     ----------
123:     c : 1-D ndarray
124:         Chebyshev coefficients, ordered from low to high
125: 
126:     Returns
127:     -------
128:     zs : 1-D ndarray
129:         Odd length symmetric z-series, ordered from  low to high.
130: 
131:     '''
132:     n = c.size
133:     zs = np.zeros(2*n-1, dtype=c.dtype)
134:     zs[n-1:] = c/2
135:     return zs + zs[::-1]
136: 
137: 
138: def _zseries_to_cseries(zs):
139:     '''Covert z-series to a Chebyshev series.
140: 
141:     Covert a z series to the equivalent Chebyshev series. The result is
142:     never an empty array. The dtype of the return is the same as that of
143:     the input. No checks are run on the arguments as this routine is for
144:     internal use.
145: 
146:     Parameters
147:     ----------
148:     zs : 1-D ndarray
149:         Odd length symmetric z-series, ordered from  low to high.
150: 
151:     Returns
152:     -------
153:     c : 1-D ndarray
154:         Chebyshev coefficients, ordered from  low to high.
155: 
156:     '''
157:     n = (zs.size + 1)//2
158:     c = zs[n-1:].copy()
159:     c[1:n] *= 2
160:     return c
161: 
162: 
163: def _zseries_mul(z1, z2):
164:     '''Multiply two z-series.
165: 
166:     Multiply two z-series to produce a z-series.
167: 
168:     Parameters
169:     ----------
170:     z1, z2 : 1-D ndarray
171:         The arrays must be 1-D but this is not checked.
172: 
173:     Returns
174:     -------
175:     product : 1-D ndarray
176:         The product z-series.
177: 
178:     Notes
179:     -----
180:     This is simply convolution. If symmetric/anti-symmetric z-series are
181:     denoted by S/A then the following rules apply:
182: 
183:     S*S, A*A -> S
184:     S*A, A*S -> A
185: 
186:     '''
187:     return np.convolve(z1, z2)
188: 
189: 
190: def _zseries_div(z1, z2):
191:     '''Divide the first z-series by the second.
192: 
193:     Divide `z1` by `z2` and return the quotient and remainder as z-series.
194:     Warning: this implementation only applies when both z1 and z2 have the
195:     same symmetry, which is sufficient for present purposes.
196: 
197:     Parameters
198:     ----------
199:     z1, z2 : 1-D ndarray
200:         The arrays must be 1-D and have the same symmetry, but this is not
201:         checked.
202: 
203:     Returns
204:     -------
205: 
206:     (quotient, remainder) : 1-D ndarrays
207:         Quotient and remainder as z-series.
208: 
209:     Notes
210:     -----
211:     This is not the same as polynomial division on account of the desired form
212:     of the remainder. If symmetric/anti-symmetric z-series are denoted by S/A
213:     then the following rules apply:
214: 
215:     S/S -> S,S
216:     A/A -> S,A
217: 
218:     The restriction to types of the same symmetry could be fixed but seems like
219:     unneeded generality. There is no natural form for the remainder in the case
220:     where there is no symmetry.
221: 
222:     '''
223:     z1 = z1.copy()
224:     z2 = z2.copy()
225:     len1 = len(z1)
226:     len2 = len(z2)
227:     if len2 == 1:
228:         z1 /= z2
229:         return z1, z1[:1]*0
230:     elif len1 < len2:
231:         return z1[:1]*0, z1
232:     else:
233:         dlen = len1 - len2
234:         scl = z2[0]
235:         z2 /= scl
236:         quo = np.empty(dlen + 1, dtype=z1.dtype)
237:         i = 0
238:         j = dlen
239:         while i < j:
240:             r = z1[i]
241:             quo[i] = z1[i]
242:             quo[dlen - i] = r
243:             tmp = r*z2
244:             z1[i:i+len2] -= tmp
245:             z1[j:j+len2] -= tmp
246:             i += 1
247:             j -= 1
248:         r = z1[i]
249:         quo[i] = r
250:         tmp = r*z2
251:         z1[i:i+len2] -= tmp
252:         quo /= scl
253:         rem = z1[i+1:i-1+len2].copy()
254:         return quo, rem
255: 
256: 
257: def _zseries_der(zs):
258:     '''Differentiate a z-series.
259: 
260:     The derivative is with respect to x, not z. This is achieved using the
261:     chain rule and the value of dx/dz given in the module notes.
262: 
263:     Parameters
264:     ----------
265:     zs : z-series
266:         The z-series to differentiate.
267: 
268:     Returns
269:     -------
270:     derivative : z-series
271:         The derivative
272: 
273:     Notes
274:     -----
275:     The zseries for x (ns) has been multiplied by two in order to avoid
276:     using floats that are incompatible with Decimal and likely other
277:     specialized scalar types. This scaling has been compensated by
278:     multiplying the value of zs by two also so that the two cancels in the
279:     division.
280: 
281:     '''
282:     n = len(zs)//2
283:     ns = np.array([-1, 0, 1], dtype=zs.dtype)
284:     zs *= np.arange(-n, n+1)*2
285:     d, r = _zseries_div(zs, ns)
286:     return d
287: 
288: 
289: def _zseries_int(zs):
290:     '''Integrate a z-series.
291: 
292:     The integral is with respect to x, not z. This is achieved by a change
293:     of variable using dx/dz given in the module notes.
294: 
295:     Parameters
296:     ----------
297:     zs : z-series
298:         The z-series to integrate
299: 
300:     Returns
301:     -------
302:     integral : z-series
303:         The indefinite integral
304: 
305:     Notes
306:     -----
307:     The zseries for x (ns) has been multiplied by two in order to avoid
308:     using floats that are incompatible with Decimal and likely other
309:     specialized scalar types. This scaling has been compensated by
310:     dividing the resulting zs by two.
311: 
312:     '''
313:     n = 1 + len(zs)//2
314:     ns = np.array([-1, 0, 1], dtype=zs.dtype)
315:     zs = _zseries_mul(zs, ns)
316:     div = np.arange(-n, n+1)*2
317:     zs[:n] /= div[:n]
318:     zs[n+1:] /= div[n+1:]
319:     zs[n] = 0
320:     return zs
321: 
322: #
323: # Chebyshev series functions
324: #
325: 
326: 
327: def poly2cheb(pol):
328:     '''
329:     Convert a polynomial to a Chebyshev series.
330: 
331:     Convert an array representing the coefficients of a polynomial (relative
332:     to the "standard" basis) ordered from lowest degree to highest, to an
333:     array of the coefficients of the equivalent Chebyshev series, ordered
334:     from lowest to highest degree.
335: 
336:     Parameters
337:     ----------
338:     pol : array_like
339:         1-D array containing the polynomial coefficients
340: 
341:     Returns
342:     -------
343:     c : ndarray
344:         1-D array containing the coefficients of the equivalent Chebyshev
345:         series.
346: 
347:     See Also
348:     --------
349:     cheb2poly
350: 
351:     Notes
352:     -----
353:     The easy way to do conversions between polynomial basis sets
354:     is to use the convert method of a class instance.
355: 
356:     Examples
357:     --------
358:     >>> from numpy import polynomial as P
359:     >>> p = P.Polynomial(range(4))
360:     >>> p
361:     Polynomial([ 0.,  1.,  2.,  3.], [-1.,  1.])
362:     >>> c = p.convert(kind=P.Chebyshev)
363:     >>> c
364:     Chebyshev([ 1.  ,  3.25,  1.  ,  0.75], [-1.,  1.])
365:     >>> P.poly2cheb(range(4))
366:     array([ 1.  ,  3.25,  1.  ,  0.75])
367: 
368:     '''
369:     [pol] = pu.as_series([pol])
370:     deg = len(pol) - 1
371:     res = 0
372:     for i in range(deg, -1, -1):
373:         res = chebadd(chebmulx(res), pol[i])
374:     return res
375: 
376: 
377: def cheb2poly(c):
378:     '''
379:     Convert a Chebyshev series to a polynomial.
380: 
381:     Convert an array representing the coefficients of a Chebyshev series,
382:     ordered from lowest degree to highest, to an array of the coefficients
383:     of the equivalent polynomial (relative to the "standard" basis) ordered
384:     from lowest to highest degree.
385: 
386:     Parameters
387:     ----------
388:     c : array_like
389:         1-D array containing the Chebyshev series coefficients, ordered
390:         from lowest order term to highest.
391: 
392:     Returns
393:     -------
394:     pol : ndarray
395:         1-D array containing the coefficients of the equivalent polynomial
396:         (relative to the "standard" basis) ordered from lowest order term
397:         to highest.
398: 
399:     See Also
400:     --------
401:     poly2cheb
402: 
403:     Notes
404:     -----
405:     The easy way to do conversions between polynomial basis sets
406:     is to use the convert method of a class instance.
407: 
408:     Examples
409:     --------
410:     >>> from numpy import polynomial as P
411:     >>> c = P.Chebyshev(range(4))
412:     >>> c
413:     Chebyshev([ 0.,  1.,  2.,  3.], [-1.,  1.])
414:     >>> p = c.convert(kind=P.Polynomial)
415:     >>> p
416:     Polynomial([ -2.,  -8.,   4.,  12.], [-1.,  1.])
417:     >>> P.cheb2poly(range(4))
418:     array([ -2.,  -8.,   4.,  12.])
419: 
420:     '''
421:     from .polynomial import polyadd, polysub, polymulx
422: 
423:     [c] = pu.as_series([c])
424:     n = len(c)
425:     if n < 3:
426:         return c
427:     else:
428:         c0 = c[-2]
429:         c1 = c[-1]
430:         # i is the current degree of c1
431:         for i in range(n - 1, 1, -1):
432:             tmp = c0
433:             c0 = polysub(c[i - 2], c1)
434:             c1 = polyadd(tmp, polymulx(c1)*2)
435:         return polyadd(c0, polymulx(c1))
436: 
437: 
438: #
439: # These are constant arrays are of integer type so as to be compatible
440: # with the widest range of other types, such as Decimal.
441: #
442: 
443: # Chebyshev default domain.
444: chebdomain = np.array([-1, 1])
445: 
446: # Chebyshev coefficients representing zero.
447: chebzero = np.array([0])
448: 
449: # Chebyshev coefficients representing one.
450: chebone = np.array([1])
451: 
452: # Chebyshev coefficients representing the identity x.
453: chebx = np.array([0, 1])
454: 
455: 
456: def chebline(off, scl):
457:     '''
458:     Chebyshev series whose graph is a straight line.
459: 
460: 
461: 
462:     Parameters
463:     ----------
464:     off, scl : scalars
465:         The specified line is given by ``off + scl*x``.
466: 
467:     Returns
468:     -------
469:     y : ndarray
470:         This module's representation of the Chebyshev series for
471:         ``off + scl*x``.
472: 
473:     See Also
474:     --------
475:     polyline
476: 
477:     Examples
478:     --------
479:     >>> import numpy.polynomial.chebyshev as C
480:     >>> C.chebline(3,2)
481:     array([3, 2])
482:     >>> C.chebval(-3, C.chebline(3,2)) # should be -3
483:     -3.0
484: 
485:     '''
486:     if scl != 0:
487:         return np.array([off, scl])
488:     else:
489:         return np.array([off])
490: 
491: 
492: def chebfromroots(roots):
493:     '''
494:     Generate a Chebyshev series with given roots.
495: 
496:     The function returns the coefficients of the polynomial
497: 
498:     .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
499: 
500:     in Chebyshev form, where the `r_n` are the roots specified in `roots`.
501:     If a zero has multiplicity n, then it must appear in `roots` n times.
502:     For instance, if 2 is a root of multiplicity three and 3 is a root of
503:     multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
504:     roots can appear in any order.
505: 
506:     If the returned coefficients are `c`, then
507: 
508:     .. math:: p(x) = c_0 + c_1 * T_1(x) + ... +  c_n * T_n(x)
509: 
510:     The coefficient of the last term is not generally 1 for monic
511:     polynomials in Chebyshev form.
512: 
513:     Parameters
514:     ----------
515:     roots : array_like
516:         Sequence containing the roots.
517: 
518:     Returns
519:     -------
520:     out : ndarray
521:         1-D array of coefficients.  If all roots are real then `out` is a
522:         real array, if some of the roots are complex, then `out` is complex
523:         even if all the coefficients in the result are real (see Examples
524:         below).
525: 
526:     See Also
527:     --------
528:     polyfromroots, legfromroots, lagfromroots, hermfromroots,
529:     hermefromroots.
530: 
531:     Examples
532:     --------
533:     >>> import numpy.polynomial.chebyshev as C
534:     >>> C.chebfromroots((-1,0,1)) # x^3 - x relative to the standard basis
535:     array([ 0.  , -0.25,  0.  ,  0.25])
536:     >>> j = complex(0,1)
537:     >>> C.chebfromroots((-j,j)) # x^2 + 1 relative to the standard basis
538:     array([ 1.5+0.j,  0.0+0.j,  0.5+0.j])
539: 
540:     '''
541:     if len(roots) == 0:
542:         return np.ones(1)
543:     else:
544:         [roots] = pu.as_series([roots], trim=False)
545:         roots.sort()
546:         p = [chebline(-r, 1) for r in roots]
547:         n = len(p)
548:         while n > 1:
549:             m, r = divmod(n, 2)
550:             tmp = [chebmul(p[i], p[i+m]) for i in range(m)]
551:             if r:
552:                 tmp[0] = chebmul(tmp[0], p[-1])
553:             p = tmp
554:             n = m
555:         return p[0]
556: 
557: 
558: def chebadd(c1, c2):
559:     '''
560:     Add one Chebyshev series to another.
561: 
562:     Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments
563:     are sequences of coefficients ordered from lowest order term to
564:     highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.
565: 
566:     Parameters
567:     ----------
568:     c1, c2 : array_like
569:         1-D arrays of Chebyshev series coefficients ordered from low to
570:         high.
571: 
572:     Returns
573:     -------
574:     out : ndarray
575:         Array representing the Chebyshev series of their sum.
576: 
577:     See Also
578:     --------
579:     chebsub, chebmul, chebdiv, chebpow
580: 
581:     Notes
582:     -----
583:     Unlike multiplication, division, etc., the sum of two Chebyshev series
584:     is a Chebyshev series (without having to "reproject" the result onto
585:     the basis set) so addition, just like that of "standard" polynomials,
586:     is simply "component-wise."
587: 
588:     Examples
589:     --------
590:     >>> from numpy.polynomial import chebyshev as C
591:     >>> c1 = (1,2,3)
592:     >>> c2 = (3,2,1)
593:     >>> C.chebadd(c1,c2)
594:     array([ 4.,  4.,  4.])
595: 
596:     '''
597:     # c1, c2 are trimmed copies
598:     [c1, c2] = pu.as_series([c1, c2])
599:     if len(c1) > len(c2):
600:         c1[:c2.size] += c2
601:         ret = c1
602:     else:
603:         c2[:c1.size] += c1
604:         ret = c2
605:     return pu.trimseq(ret)
606: 
607: 
608: def chebsub(c1, c2):
609:     '''
610:     Subtract one Chebyshev series from another.
611: 
612:     Returns the difference of two Chebyshev series `c1` - `c2`.  The
613:     sequences of coefficients are from lowest order term to highest, i.e.,
614:     [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.
615: 
616:     Parameters
617:     ----------
618:     c1, c2 : array_like
619:         1-D arrays of Chebyshev series coefficients ordered from low to
620:         high.
621: 
622:     Returns
623:     -------
624:     out : ndarray
625:         Of Chebyshev series coefficients representing their difference.
626: 
627:     See Also
628:     --------
629:     chebadd, chebmul, chebdiv, chebpow
630: 
631:     Notes
632:     -----
633:     Unlike multiplication, division, etc., the difference of two Chebyshev
634:     series is a Chebyshev series (without having to "reproject" the result
635:     onto the basis set) so subtraction, just like that of "standard"
636:     polynomials, is simply "component-wise."
637: 
638:     Examples
639:     --------
640:     >>> from numpy.polynomial import chebyshev as C
641:     >>> c1 = (1,2,3)
642:     >>> c2 = (3,2,1)
643:     >>> C.chebsub(c1,c2)
644:     array([-2.,  0.,  2.])
645:     >>> C.chebsub(c2,c1) # -C.chebsub(c1,c2)
646:     array([ 2.,  0., -2.])
647: 
648:     '''
649:     # c1, c2 are trimmed copies
650:     [c1, c2] = pu.as_series([c1, c2])
651:     if len(c1) > len(c2):
652:         c1[:c2.size] -= c2
653:         ret = c1
654:     else:
655:         c2 = -c2
656:         c2[:c1.size] += c1
657:         ret = c2
658:     return pu.trimseq(ret)
659: 
660: 
661: def chebmulx(c):
662:     '''Multiply a Chebyshev series by x.
663: 
664:     Multiply the polynomial `c` by x, where x is the independent
665:     variable.
666: 
667: 
668:     Parameters
669:     ----------
670:     c : array_like
671:         1-D array of Chebyshev series coefficients ordered from low to
672:         high.
673: 
674:     Returns
675:     -------
676:     out : ndarray
677:         Array representing the result of the multiplication.
678: 
679:     Notes
680:     -----
681: 
682:     .. versionadded:: 1.5.0
683: 
684:     '''
685:     # c is a trimmed copy
686:     [c] = pu.as_series([c])
687:     # The zero series needs special treatment
688:     if len(c) == 1 and c[0] == 0:
689:         return c
690: 
691:     prd = np.empty(len(c) + 1, dtype=c.dtype)
692:     prd[0] = c[0]*0
693:     prd[1] = c[0]
694:     if len(c) > 1:
695:         tmp = c[1:]/2
696:         prd[2:] = tmp
697:         prd[0:-2] += tmp
698:     return prd
699: 
700: 
701: def chebmul(c1, c2):
702:     '''
703:     Multiply one Chebyshev series by another.
704: 
705:     Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
706:     are sequences of coefficients, from lowest order "term" to highest,
707:     e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.
708: 
709:     Parameters
710:     ----------
711:     c1, c2 : array_like
712:         1-D arrays of Chebyshev series coefficients ordered from low to
713:         high.
714: 
715:     Returns
716:     -------
717:     out : ndarray
718:         Of Chebyshev series coefficients representing their product.
719: 
720:     See Also
721:     --------
722:     chebadd, chebsub, chebdiv, chebpow
723: 
724:     Notes
725:     -----
726:     In general, the (polynomial) product of two C-series results in terms
727:     that are not in the Chebyshev polynomial basis set.  Thus, to express
728:     the product as a C-series, it is typically necessary to "reproject"
729:     the product onto said basis set, which typically produces
730:     "unintuitive live" (but correct) results; see Examples section below.
731: 
732:     Examples
733:     --------
734:     >>> from numpy.polynomial import chebyshev as C
735:     >>> c1 = (1,2,3)
736:     >>> c2 = (3,2,1)
737:     >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
738:     array([  6.5,  12. ,  12. ,   4. ,   1.5])
739: 
740:     '''
741:     # c1, c2 are trimmed copies
742:     [c1, c2] = pu.as_series([c1, c2])
743:     z1 = _cseries_to_zseries(c1)
744:     z2 = _cseries_to_zseries(c2)
745:     prd = _zseries_mul(z1, z2)
746:     ret = _zseries_to_cseries(prd)
747:     return pu.trimseq(ret)
748: 
749: 
750: def chebdiv(c1, c2):
751:     '''
752:     Divide one Chebyshev series by another.
753: 
754:     Returns the quotient-with-remainder of two Chebyshev series
755:     `c1` / `c2`.  The arguments are sequences of coefficients from lowest
756:     order "term" to highest, e.g., [1,2,3] represents the series
757:     ``T_0 + 2*T_1 + 3*T_2``.
758: 
759:     Parameters
760:     ----------
761:     c1, c2 : array_like
762:         1-D arrays of Chebyshev series coefficients ordered from low to
763:         high.
764: 
765:     Returns
766:     -------
767:     [quo, rem] : ndarrays
768:         Of Chebyshev series coefficients representing the quotient and
769:         remainder.
770: 
771:     See Also
772:     --------
773:     chebadd, chebsub, chebmul, chebpow
774: 
775:     Notes
776:     -----
777:     In general, the (polynomial) division of one C-series by another
778:     results in quotient and remainder terms that are not in the Chebyshev
779:     polynomial basis set.  Thus, to express these results as C-series, it
780:     is typically necessary to "reproject" the results onto said basis
781:     set, which typically produces "unintuitive" (but correct) results;
782:     see Examples section below.
783: 
784:     Examples
785:     --------
786:     >>> from numpy.polynomial import chebyshev as C
787:     >>> c1 = (1,2,3)
788:     >>> c2 = (3,2,1)
789:     >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not
790:     (array([ 3.]), array([-8., -4.]))
791:     >>> c2 = (0,1,2,3)
792:     >>> C.chebdiv(c2,c1) # neither "intuitive"
793:     (array([ 0.,  2.]), array([-2., -4.]))
794: 
795:     '''
796:     # c1, c2 are trimmed copies
797:     [c1, c2] = pu.as_series([c1, c2])
798:     if c2[-1] == 0:
799:         raise ZeroDivisionError()
800: 
801:     lc1 = len(c1)
802:     lc2 = len(c2)
803:     if lc1 < lc2:
804:         return c1[:1]*0, c1
805:     elif lc2 == 1:
806:         return c1/c2[-1], c1[:1]*0
807:     else:
808:         z1 = _cseries_to_zseries(c1)
809:         z2 = _cseries_to_zseries(c2)
810:         quo, rem = _zseries_div(z1, z2)
811:         quo = pu.trimseq(_zseries_to_cseries(quo))
812:         rem = pu.trimseq(_zseries_to_cseries(rem))
813:         return quo, rem
814: 
815: 
816: def chebpow(c, pow, maxpower=16):
817:     '''Raise a Chebyshev series to a power.
818: 
819:     Returns the Chebyshev series `c` raised to the power `pow`. The
820:     argument `c` is a sequence of coefficients ordered from low to high.
821:     i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``
822: 
823:     Parameters
824:     ----------
825:     c : array_like
826:         1-D array of Chebyshev series coefficients ordered from low to
827:         high.
828:     pow : integer
829:         Power to which the series will be raised
830:     maxpower : integer, optional
831:         Maximum power allowed. This is mainly to limit growth of the series
832:         to unmanageable size. Default is 16
833: 
834:     Returns
835:     -------
836:     coef : ndarray
837:         Chebyshev series of power.
838: 
839:     See Also
840:     --------
841:     chebadd, chebsub, chebmul, chebdiv
842: 
843:     Examples
844:     --------
845: 
846:     '''
847:     # c is a trimmed copy
848:     [c] = pu.as_series([c])
849:     power = int(pow)
850:     if power != pow or power < 0:
851:         raise ValueError("Power must be a non-negative integer.")
852:     elif maxpower is not None and power > maxpower:
853:         raise ValueError("Power is too large")
854:     elif power == 0:
855:         return np.array([1], dtype=c.dtype)
856:     elif power == 1:
857:         return c
858:     else:
859:         # This can be made more efficient by using powers of two
860:         # in the usual way.
861:         zs = _cseries_to_zseries(c)
862:         prd = zs
863:         for i in range(2, power + 1):
864:             prd = np.convolve(prd, zs)
865:         return _zseries_to_cseries(prd)
866: 
867: 
868: def chebder(c, m=1, scl=1, axis=0):
869:     '''
870:     Differentiate a Chebyshev series.
871: 
872:     Returns the Chebyshev series coefficients `c` differentiated `m` times
873:     along `axis`.  At each iteration the result is multiplied by `scl` (the
874:     scaling factor is for use in a linear change of variable). The argument
875:     `c` is an array of coefficients from low to high degree along each
876:     axis, e.g., [1,2,3] represents the series ``1*T_0 + 2*T_1 + 3*T_2``
877:     while [[1,2],[1,2]] represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) +
878:     2*T_0(x)*T_1(y) + 2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is
879:     ``y``.
880: 
881:     Parameters
882:     ----------
883:     c : array_like
884:         Array of Chebyshev series coefficients. If c is multidimensional
885:         the different axis correspond to different variables with the
886:         degree in each axis given by the corresponding index.
887:     m : int, optional
888:         Number of derivatives taken, must be non-negative. (Default: 1)
889:     scl : scalar, optional
890:         Each differentiation is multiplied by `scl`.  The end result is
891:         multiplication by ``scl**m``.  This is for use in a linear change of
892:         variable. (Default: 1)
893:     axis : int, optional
894:         Axis over which the derivative is taken. (Default: 0).
895: 
896:         .. versionadded:: 1.7.0
897: 
898:     Returns
899:     -------
900:     der : ndarray
901:         Chebyshev series of the derivative.
902: 
903:     See Also
904:     --------
905:     chebint
906: 
907:     Notes
908:     -----
909:     In general, the result of differentiating a C-series needs to be
910:     "reprojected" onto the C-series basis set. Thus, typically, the
911:     result of this function is "unintuitive," albeit correct; see Examples
912:     section below.
913: 
914:     Examples
915:     --------
916:     >>> from numpy.polynomial import chebyshev as C
917:     >>> c = (1,2,3,4)
918:     >>> C.chebder(c)
919:     array([ 14.,  12.,  24.])
920:     >>> C.chebder(c,3)
921:     array([ 96.])
922:     >>> C.chebder(c,scl=-1)
923:     array([-14., -12., -24.])
924:     >>> C.chebder(c,2,-1)
925:     array([ 12.,  96.])
926: 
927:     '''
928:     c = np.array(c, ndmin=1, copy=1)
929:     if c.dtype.char in '?bBhHiIlLqQpP':
930:         c = c.astype(np.double)
931:     cnt, iaxis = [int(t) for t in [m, axis]]
932: 
933:     if cnt != m:
934:         raise ValueError("The order of derivation must be integer")
935:     if cnt < 0:
936:         raise ValueError("The order of derivation must be non-negative")
937:     if iaxis != axis:
938:         raise ValueError("The axis must be integer")
939:     if not -c.ndim <= iaxis < c.ndim:
940:         raise ValueError("The axis is out of range")
941:     if iaxis < 0:
942:         iaxis += c.ndim
943: 
944:     if cnt == 0:
945:         return c
946: 
947:     c = np.rollaxis(c, iaxis)
948:     n = len(c)
949:     if cnt >= n:
950:         c = c[:1]*0
951:     else:
952:         for i in range(cnt):
953:             n = n - 1
954:             c *= scl
955:             der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
956:             for j in range(n, 2, -1):
957:                 der[j - 1] = (2*j)*c[j]
958:                 c[j - 2] += (j*c[j])/(j - 2)
959:             if n > 1:
960:                 der[1] = 4*c[2]
961:             der[0] = c[1]
962:             c = der
963:     c = np.rollaxis(c, 0, iaxis + 1)
964:     return c
965: 
966: 
967: def chebint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
968:     '''
969:     Integrate a Chebyshev series.
970: 
971:     Returns the Chebyshev series coefficients `c` integrated `m` times from
972:     `lbnd` along `axis`. At each iteration the resulting series is
973:     **multiplied** by `scl` and an integration constant, `k`, is added.
974:     The scaling factor is for use in a linear change of variable.  ("Buyer
975:     beware": note that, depending on what one is doing, one may want `scl`
976:     to be the reciprocal of what one might expect; for more information,
977:     see the Notes section below.)  The argument `c` is an array of
978:     coefficients from low to high degree along each axis, e.g., [1,2,3]
979:     represents the series ``T_0 + 2*T_1 + 3*T_2`` while [[1,2],[1,2]]
980:     represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) + 2*T_0(x)*T_1(y) +
981:     2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
982: 
983:     Parameters
984:     ----------
985:     c : array_like
986:         Array of Chebyshev series coefficients. If c is multidimensional
987:         the different axis correspond to different variables with the
988:         degree in each axis given by the corresponding index.
989:     m : int, optional
990:         Order of integration, must be positive. (Default: 1)
991:     k : {[], list, scalar}, optional
992:         Integration constant(s).  The value of the first integral at zero
993:         is the first value in the list, the value of the second integral
994:         at zero is the second value, etc.  If ``k == []`` (the default),
995:         all constants are set to zero.  If ``m == 1``, a single scalar can
996:         be given instead of a list.
997:     lbnd : scalar, optional
998:         The lower bound of the integral. (Default: 0)
999:     scl : scalar, optional
1000:         Following each integration the result is *multiplied* by `scl`
1001:         before the integration constant is added. (Default: 1)
1002:     axis : int, optional
1003:         Axis over which the integral is taken. (Default: 0).
1004: 
1005:         .. versionadded:: 1.7.0
1006: 
1007:     Returns
1008:     -------
1009:     S : ndarray
1010:         C-series coefficients of the integral.
1011: 
1012:     Raises
1013:     ------
1014:     ValueError
1015:         If ``m < 1``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
1016:         ``np.isscalar(scl) == False``.
1017: 
1018:     See Also
1019:     --------
1020:     chebder
1021: 
1022:     Notes
1023:     -----
1024:     Note that the result of each integration is *multiplied* by `scl`.
1025:     Why is this important to note?  Say one is making a linear change of
1026:     variable :math:`u = ax + b` in an integral relative to `x`.  Then
1027:     .. math::`dx = du/a`, so one will need to set `scl` equal to
1028:     :math:`1/a`- perhaps not what one would have first thought.
1029: 
1030:     Also note that, in general, the result of integrating a C-series needs
1031:     to be "reprojected" onto the C-series basis set.  Thus, typically,
1032:     the result of this function is "unintuitive," albeit correct; see
1033:     Examples section below.
1034: 
1035:     Examples
1036:     --------
1037:     >>> from numpy.polynomial import chebyshev as C
1038:     >>> c = (1,2,3)
1039:     >>> C.chebint(c)
1040:     array([ 0.5, -0.5,  0.5,  0.5])
1041:     >>> C.chebint(c,3)
1042:     array([ 0.03125   , -0.1875    ,  0.04166667, -0.05208333,  0.01041667,
1043:             0.00625   ])
1044:     >>> C.chebint(c, k=3)
1045:     array([ 3.5, -0.5,  0.5,  0.5])
1046:     >>> C.chebint(c,lbnd=-2)
1047:     array([ 8.5, -0.5,  0.5,  0.5])
1048:     >>> C.chebint(c,scl=-2)
1049:     array([-1.,  1., -1., -1.])
1050: 
1051:     '''
1052:     c = np.array(c, ndmin=1, copy=1)
1053:     if c.dtype.char in '?bBhHiIlLqQpP':
1054:         c = c.astype(np.double)
1055:     if not np.iterable(k):
1056:         k = [k]
1057:     cnt, iaxis = [int(t) for t in [m, axis]]
1058: 
1059:     if cnt != m:
1060:         raise ValueError("The order of integration must be integer")
1061:     if cnt < 0:
1062:         raise ValueError("The order of integration must be non-negative")
1063:     if len(k) > cnt:
1064:         raise ValueError("Too many integration constants")
1065:     if iaxis != axis:
1066:         raise ValueError("The axis must be integer")
1067:     if not -c.ndim <= iaxis < c.ndim:
1068:         raise ValueError("The axis is out of range")
1069:     if iaxis < 0:
1070:         iaxis += c.ndim
1071: 
1072:     if cnt == 0:
1073:         return c
1074: 
1075:     c = np.rollaxis(c, iaxis)
1076:     k = list(k) + [0]*(cnt - len(k))
1077:     for i in range(cnt):
1078:         n = len(c)
1079:         c *= scl
1080:         if n == 1 and np.all(c[0] == 0):
1081:             c[0] += k[i]
1082:         else:
1083:             tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
1084:             tmp[0] = c[0]*0
1085:             tmp[1] = c[0]
1086:             if n > 1:
1087:                 tmp[2] = c[1]/4
1088:             for j in range(2, n):
1089:                 t = c[j]/(2*j + 1)
1090:                 tmp[j + 1] = c[j]/(2*(j + 1))
1091:                 tmp[j - 1] -= c[j]/(2*(j - 1))
1092:             tmp[0] += k[i] - chebval(lbnd, tmp)
1093:             c = tmp
1094:     c = np.rollaxis(c, 0, iaxis + 1)
1095:     return c
1096: 
1097: 
1098: def chebval(x, c, tensor=True):
1099:     '''
1100:     Evaluate a Chebyshev series at points x.
1101: 
1102:     If `c` is of length `n + 1`, this function returns the value:
1103: 
1104:     .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)
1105: 
1106:     The parameter `x` is converted to an array only if it is a tuple or a
1107:     list, otherwise it is treated as a scalar. In either case, either `x`
1108:     or its elements must support multiplication and addition both with
1109:     themselves and with the elements of `c`.
1110: 
1111:     If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
1112:     `c` is multidimensional, then the shape of the result depends on the
1113:     value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
1114:     x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
1115:     scalars have shape (,).
1116: 
1117:     Trailing zeros in the coefficients will be used in the evaluation, so
1118:     they should be avoided if efficiency is a concern.
1119: 
1120:     Parameters
1121:     ----------
1122:     x : array_like, compatible object
1123:         If `x` is a list or tuple, it is converted to an ndarray, otherwise
1124:         it is left unchanged and treated as a scalar. In either case, `x`
1125:         or its elements must support addition and multiplication with
1126:         with themselves and with the elements of `c`.
1127:     c : array_like
1128:         Array of coefficients ordered so that the coefficients for terms of
1129:         degree n are contained in c[n]. If `c` is multidimensional the
1130:         remaining indices enumerate multiple polynomials. In the two
1131:         dimensional case the coefficients may be thought of as stored in
1132:         the columns of `c`.
1133:     tensor : boolean, optional
1134:         If True, the shape of the coefficient array is extended with ones
1135:         on the right, one for each dimension of `x`. Scalars have dimension 0
1136:         for this action. The result is that every column of coefficients in
1137:         `c` is evaluated for every element of `x`. If False, `x` is broadcast
1138:         over the columns of `c` for the evaluation.  This keyword is useful
1139:         when `c` is multidimensional. The default value is True.
1140: 
1141:         .. versionadded:: 1.7.0
1142: 
1143:     Returns
1144:     -------
1145:     values : ndarray, algebra_like
1146:         The shape of the return value is described above.
1147: 
1148:     See Also
1149:     --------
1150:     chebval2d, chebgrid2d, chebval3d, chebgrid3d
1151: 
1152:     Notes
1153:     -----
1154:     The evaluation uses Clenshaw recursion, aka synthetic division.
1155: 
1156:     Examples
1157:     --------
1158: 
1159:     '''
1160:     c = np.array(c, ndmin=1, copy=1)
1161:     if c.dtype.char in '?bBhHiIlLqQpP':
1162:         c = c.astype(np.double)
1163:     if isinstance(x, (tuple, list)):
1164:         x = np.asarray(x)
1165:     if isinstance(x, np.ndarray) and tensor:
1166:         c = c.reshape(c.shape + (1,)*x.ndim)
1167: 
1168:     if len(c) == 1:
1169:         c0 = c[0]
1170:         c1 = 0
1171:     elif len(c) == 2:
1172:         c0 = c[0]
1173:         c1 = c[1]
1174:     else:
1175:         x2 = 2*x
1176:         c0 = c[-2]
1177:         c1 = c[-1]
1178:         for i in range(3, len(c) + 1):
1179:             tmp = c0
1180:             c0 = c[-i] - c1
1181:             c1 = tmp + c1*x2
1182:     return c0 + c1*x
1183: 
1184: 
1185: def chebval2d(x, y, c):
1186:     '''
1187:     Evaluate a 2-D Chebyshev series at points (x, y).
1188: 
1189:     This function returns the values:
1190: 
1191:     .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * T_i(x) * T_j(y)
1192: 
1193:     The parameters `x` and `y` are converted to arrays only if they are
1194:     tuples or a lists, otherwise they are treated as a scalars and they
1195:     must have the same shape after conversion. In either case, either `x`
1196:     and `y` or their elements must support multiplication and addition both
1197:     with themselves and with the elements of `c`.
1198: 
1199:     If `c` is a 1-D array a one is implicitly appended to its shape to make
1200:     it 2-D. The shape of the result will be c.shape[2:] + x.shape.
1201: 
1202:     Parameters
1203:     ----------
1204:     x, y : array_like, compatible objects
1205:         The two dimensional series is evaluated at the points `(x, y)`,
1206:         where `x` and `y` must have the same shape. If `x` or `y` is a list
1207:         or tuple, it is first converted to an ndarray, otherwise it is left
1208:         unchanged and if it isn't an ndarray it is treated as a scalar.
1209:     c : array_like
1210:         Array of coefficients ordered so that the coefficient of the term
1211:         of multi-degree i,j is contained in ``c[i,j]``. If `c` has
1212:         dimension greater than 2 the remaining indices enumerate multiple
1213:         sets of coefficients.
1214: 
1215:     Returns
1216:     -------
1217:     values : ndarray, compatible object
1218:         The values of the two dimensional Chebyshev series at points formed
1219:         from pairs of corresponding values from `x` and `y`.
1220: 
1221:     See Also
1222:     --------
1223:     chebval, chebgrid2d, chebval3d, chebgrid3d
1224: 
1225:     Notes
1226:     -----
1227: 
1228:     .. versionadded::1.7.0
1229: 
1230:     '''
1231:     try:
1232:         x, y = np.array((x, y), copy=0)
1233:     except:
1234:         raise ValueError('x, y are incompatible')
1235: 
1236:     c = chebval(x, c)
1237:     c = chebval(y, c, tensor=False)
1238:     return c
1239: 
1240: 
1241: def chebgrid2d(x, y, c):
1242:     '''
1243:     Evaluate a 2-D Chebyshev series on the Cartesian product of x and y.
1244: 
1245:     This function returns the values:
1246: 
1247:     .. math:: p(a,b) = \sum_{i,j} c_{i,j} * T_i(a) * T_j(b),
1248: 
1249:     where the points `(a, b)` consist of all pairs formed by taking
1250:     `a` from `x` and `b` from `y`. The resulting points form a grid with
1251:     `x` in the first dimension and `y` in the second.
1252: 
1253:     The parameters `x` and `y` are converted to arrays only if they are
1254:     tuples or a lists, otherwise they are treated as a scalars. In either
1255:     case, either `x` and `y` or their elements must support multiplication
1256:     and addition both with themselves and with the elements of `c`.
1257: 
1258:     If `c` has fewer than two dimensions, ones are implicitly appended to
1259:     its shape to make it 2-D. The shape of the result will be c.shape[2:] +
1260:     x.shape + y.shape.
1261: 
1262:     Parameters
1263:     ----------
1264:     x, y : array_like, compatible objects
1265:         The two dimensional series is evaluated at the points in the
1266:         Cartesian product of `x` and `y`.  If `x` or `y` is a list or
1267:         tuple, it is first converted to an ndarray, otherwise it is left
1268:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
1269:     c : array_like
1270:         Array of coefficients ordered so that the coefficient of the term of
1271:         multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
1272:         greater than two the remaining indices enumerate multiple sets of
1273:         coefficients.
1274: 
1275:     Returns
1276:     -------
1277:     values : ndarray, compatible object
1278:         The values of the two dimensional Chebyshev series at points in the
1279:         Cartesian product of `x` and `y`.
1280: 
1281:     See Also
1282:     --------
1283:     chebval, chebval2d, chebval3d, chebgrid3d
1284: 
1285:     Notes
1286:     -----
1287: 
1288:     .. versionadded::1.7.0
1289: 
1290:     '''
1291:     c = chebval(x, c)
1292:     c = chebval(y, c)
1293:     return c
1294: 
1295: 
1296: def chebval3d(x, y, z, c):
1297:     '''
1298:     Evaluate a 3-D Chebyshev series at points (x, y, z).
1299: 
1300:     This function returns the values:
1301: 
1302:     .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * T_i(x) * T_j(y) * T_k(z)
1303: 
1304:     The parameters `x`, `y`, and `z` are converted to arrays only if
1305:     they are tuples or a lists, otherwise they are treated as a scalars and
1306:     they must have the same shape after conversion. In either case, either
1307:     `x`, `y`, and `z` or their elements must support multiplication and
1308:     addition both with themselves and with the elements of `c`.
1309: 
1310:     If `c` has fewer than 3 dimensions, ones are implicitly appended to its
1311:     shape to make it 3-D. The shape of the result will be c.shape[3:] +
1312:     x.shape.
1313: 
1314:     Parameters
1315:     ----------
1316:     x, y, z : array_like, compatible object
1317:         The three dimensional series is evaluated at the points
1318:         `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
1319:         any of `x`, `y`, or `z` is a list or tuple, it is first converted
1320:         to an ndarray, otherwise it is left unchanged and if it isn't an
1321:         ndarray it is  treated as a scalar.
1322:     c : array_like
1323:         Array of coefficients ordered so that the coefficient of the term of
1324:         multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
1325:         greater than 3 the remaining indices enumerate multiple sets of
1326:         coefficients.
1327: 
1328:     Returns
1329:     -------
1330:     values : ndarray, compatible object
1331:         The values of the multidimensional polynomial on points formed with
1332:         triples of corresponding values from `x`, `y`, and `z`.
1333: 
1334:     See Also
1335:     --------
1336:     chebval, chebval2d, chebgrid2d, chebgrid3d
1337: 
1338:     Notes
1339:     -----
1340: 
1341:     .. versionadded::1.7.0
1342: 
1343:     '''
1344:     try:
1345:         x, y, z = np.array((x, y, z), copy=0)
1346:     except:
1347:         raise ValueError('x, y, z are incompatible')
1348: 
1349:     c = chebval(x, c)
1350:     c = chebval(y, c, tensor=False)
1351:     c = chebval(z, c, tensor=False)
1352:     return c
1353: 
1354: 
1355: def chebgrid3d(x, y, z, c):
1356:     '''
1357:     Evaluate a 3-D Chebyshev series on the Cartesian product of x, y, and z.
1358: 
1359:     This function returns the values:
1360: 
1361:     .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * T_i(a) * T_j(b) * T_k(c)
1362: 
1363:     where the points `(a, b, c)` consist of all triples formed by taking
1364:     `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
1365:     a grid with `x` in the first dimension, `y` in the second, and `z` in
1366:     the third.
1367: 
1368:     The parameters `x`, `y`, and `z` are converted to arrays only if they
1369:     are tuples or a lists, otherwise they are treated as a scalars. In
1370:     either case, either `x`, `y`, and `z` or their elements must support
1371:     multiplication and addition both with themselves and with the elements
1372:     of `c`.
1373: 
1374:     If `c` has fewer than three dimensions, ones are implicitly appended to
1375:     its shape to make it 3-D. The shape of the result will be c.shape[3:] +
1376:     x.shape + y.shape + z.shape.
1377: 
1378:     Parameters
1379:     ----------
1380:     x, y, z : array_like, compatible objects
1381:         The three dimensional series is evaluated at the points in the
1382:         Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
1383:         list or tuple, it is first converted to an ndarray, otherwise it is
1384:         left unchanged and, if it isn't an ndarray, it is treated as a
1385:         scalar.
1386:     c : array_like
1387:         Array of coefficients ordered so that the coefficients for terms of
1388:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
1389:         greater than two the remaining indices enumerate multiple sets of
1390:         coefficients.
1391: 
1392:     Returns
1393:     -------
1394:     values : ndarray, compatible object
1395:         The values of the two dimensional polynomial at points in the Cartesian
1396:         product of `x` and `y`.
1397: 
1398:     See Also
1399:     --------
1400:     chebval, chebval2d, chebgrid2d, chebval3d
1401: 
1402:     Notes
1403:     -----
1404: 
1405:     .. versionadded::1.7.0
1406: 
1407:     '''
1408:     c = chebval(x, c)
1409:     c = chebval(y, c)
1410:     c = chebval(z, c)
1411:     return c
1412: 
1413: 
1414: def chebvander(x, deg):
1415:     '''Pseudo-Vandermonde matrix of given degree.
1416: 
1417:     Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
1418:     `x`. The pseudo-Vandermonde matrix is defined by
1419: 
1420:     .. math:: V[..., i] = T_i(x),
1421: 
1422:     where `0 <= i <= deg`. The leading indices of `V` index the elements of
1423:     `x` and the last index is the degree of the Chebyshev polynomial.
1424: 
1425:     If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
1426:     matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
1427:     ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
1428:     useful both for least squares fitting and for the evaluation of a large
1429:     number of Chebyshev series of the same degree and sample points.
1430: 
1431:     Parameters
1432:     ----------
1433:     x : array_like
1434:         Array of points. The dtype is converted to float64 or complex128
1435:         depending on whether any of the elements are complex. If `x` is
1436:         scalar it is converted to a 1-D array.
1437:     deg : int
1438:         Degree of the resulting matrix.
1439: 
1440:     Returns
1441:     -------
1442:     vander : ndarray
1443:         The pseudo Vandermonde matrix. The shape of the returned matrix is
1444:         ``x.shape + (deg + 1,)``, where The last index is the degree of the
1445:         corresponding Chebyshev polynomial.  The dtype will be the same as
1446:         the converted `x`.
1447: 
1448:     '''
1449:     ideg = int(deg)
1450:     if ideg != deg:
1451:         raise ValueError("deg must be integer")
1452:     if ideg < 0:
1453:         raise ValueError("deg must be non-negative")
1454: 
1455:     x = np.array(x, copy=0, ndmin=1) + 0.0
1456:     dims = (ideg + 1,) + x.shape
1457:     dtyp = x.dtype
1458:     v = np.empty(dims, dtype=dtyp)
1459:     # Use forward recursion to generate the entries.
1460:     v[0] = x*0 + 1
1461:     if ideg > 0:
1462:         x2 = 2*x
1463:         v[1] = x
1464:         for i in range(2, ideg + 1):
1465:             v[i] = v[i-1]*x2 - v[i-2]
1466:     return np.rollaxis(v, 0, v.ndim)
1467: 
1468: 
1469: def chebvander2d(x, y, deg):
1470:     '''Pseudo-Vandermonde matrix of given degrees.
1471: 
1472:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1473:     points `(x, y)`. The pseudo-Vandermonde matrix is defined by
1474: 
1475:     .. math:: V[..., deg[1]*i + j] = T_i(x) * T_j(y),
1476: 
1477:     where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
1478:     `V` index the points `(x, y)` and the last index encodes the degrees of
1479:     the Chebyshev polynomials.
1480: 
1481:     If ``V = chebvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
1482:     correspond to the elements of a 2-D coefficient array `c` of shape
1483:     (xdeg + 1, ydeg + 1) in the order
1484: 
1485:     .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
1486: 
1487:     and ``np.dot(V, c.flat)`` and ``chebval2d(x, y, c)`` will be the same
1488:     up to roundoff. This equivalence is useful both for least squares
1489:     fitting and for the evaluation of a large number of 2-D Chebyshev
1490:     series of the same degrees and sample points.
1491: 
1492:     Parameters
1493:     ----------
1494:     x, y : array_like
1495:         Arrays of point coordinates, all of the same shape. The dtypes
1496:         will be converted to either float64 or complex128 depending on
1497:         whether any of the elements are complex. Scalars are converted to
1498:         1-D arrays.
1499:     deg : list of ints
1500:         List of maximum degrees of the form [x_deg, y_deg].
1501: 
1502:     Returns
1503:     -------
1504:     vander2d : ndarray
1505:         The shape of the returned matrix is ``x.shape + (order,)``, where
1506:         :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
1507:         as the converted `x` and `y`.
1508: 
1509:     See Also
1510:     --------
1511:     chebvander, chebvander3d. chebval2d, chebval3d
1512: 
1513:     Notes
1514:     -----
1515: 
1516:     .. versionadded::1.7.0
1517: 
1518:     '''
1519:     ideg = [int(d) for d in deg]
1520:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1521:     if is_valid != [1, 1]:
1522:         raise ValueError("degrees must be non-negative integers")
1523:     degx, degy = ideg
1524:     x, y = np.array((x, y), copy=0) + 0.0
1525: 
1526:     vx = chebvander(x, degx)
1527:     vy = chebvander(y, degy)
1528:     v = vx[..., None]*vy[..., None,:]
1529:     return v.reshape(v.shape[:-2] + (-1,))
1530: 
1531: 
1532: def chebvander3d(x, y, z, deg):
1533:     '''Pseudo-Vandermonde matrix of given degrees.
1534: 
1535:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1536:     points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
1537:     then The pseudo-Vandermonde matrix is defined by
1538: 
1539:     .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x)*T_j(y)*T_k(z),
1540: 
1541:     where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
1542:     indices of `V` index the points `(x, y, z)` and the last index encodes
1543:     the degrees of the Chebyshev polynomials.
1544: 
1545:     If ``V = chebvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
1546:     of `V` correspond to the elements of a 3-D coefficient array `c` of
1547:     shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
1548: 
1549:     .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
1550: 
1551:     and ``np.dot(V, c.flat)`` and ``chebval3d(x, y, z, c)`` will be the
1552:     same up to roundoff. This equivalence is useful both for least squares
1553:     fitting and for the evaluation of a large number of 3-D Chebyshev
1554:     series of the same degrees and sample points.
1555: 
1556:     Parameters
1557:     ----------
1558:     x, y, z : array_like
1559:         Arrays of point coordinates, all of the same shape. The dtypes will
1560:         be converted to either float64 or complex128 depending on whether
1561:         any of the elements are complex. Scalars are converted to 1-D
1562:         arrays.
1563:     deg : list of ints
1564:         List of maximum degrees of the form [x_deg, y_deg, z_deg].
1565: 
1566:     Returns
1567:     -------
1568:     vander3d : ndarray
1569:         The shape of the returned matrix is ``x.shape + (order,)``, where
1570:         :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
1571:         be the same as the converted `x`, `y`, and `z`.
1572: 
1573:     See Also
1574:     --------
1575:     chebvander, chebvander3d. chebval2d, chebval3d
1576: 
1577:     Notes
1578:     -----
1579: 
1580:     .. versionadded::1.7.0
1581: 
1582:     '''
1583:     ideg = [int(d) for d in deg]
1584:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1585:     if is_valid != [1, 1, 1]:
1586:         raise ValueError("degrees must be non-negative integers")
1587:     degx, degy, degz = ideg
1588:     x, y, z = np.array((x, y, z), copy=0) + 0.0
1589: 
1590:     vx = chebvander(x, degx)
1591:     vy = chebvander(y, degy)
1592:     vz = chebvander(z, degz)
1593:     v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
1594:     return v.reshape(v.shape[:-3] + (-1,))
1595: 
1596: 
1597: def chebfit(x, y, deg, rcond=None, full=False, w=None):
1598:     '''
1599:     Least squares fit of Chebyshev series to data.
1600: 
1601:     Return the coefficients of a Legendre series of degree `deg` that is the
1602:     least squares fit to the data values `y` given at points `x`. If `y` is
1603:     1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
1604:     fits are done, one for each column of `y`, and the resulting
1605:     coefficients are stored in the corresponding columns of a 2-D return.
1606:     The fitted polynomial(s) are in the form
1607: 
1608:     .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),
1609: 
1610:     where `n` is `deg`.
1611: 
1612:     Parameters
1613:     ----------
1614:     x : array_like, shape (M,)
1615:         x-coordinates of the M sample points ``(x[i], y[i])``.
1616:     y : array_like, shape (M,) or (M, K)
1617:         y-coordinates of the sample points. Several data sets of sample
1618:         points sharing the same x-coordinates can be fitted at once by
1619:         passing in a 2D-array that contains one dataset per column.
1620:     deg : int or 1-D array_like
1621:         Degree(s) of the fitting polynomials. If `deg` is a single integer
1622:         all terms up to and including the `deg`'th term are included in the
1623:         fit. For Numpy versions >= 1.11 a list of integers specifying the
1624:         degrees of the terms to include may be used instead.
1625:     rcond : float, optional
1626:         Relative condition number of the fit. Singular values smaller than
1627:         this relative to the largest singular value will be ignored. The
1628:         default value is len(x)*eps, where eps is the relative precision of
1629:         the float type, about 2e-16 in most cases.
1630:     full : bool, optional
1631:         Switch determining nature of return value. When it is False (the
1632:         default) just the coefficients are returned, when True diagnostic
1633:         information from the singular value decomposition is also returned.
1634:     w : array_like, shape (`M`,), optional
1635:         Weights. If not None, the contribution of each point
1636:         ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
1637:         weights are chosen so that the errors of the products ``w[i]*y[i]``
1638:         all have the same variance.  The default value is None.
1639: 
1640:         .. versionadded:: 1.5.0
1641: 
1642:     Returns
1643:     -------
1644:     coef : ndarray, shape (M,) or (M, K)
1645:         Chebyshev coefficients ordered from low to high. If `y` was 2-D,
1646:         the coefficients for the data in column k  of `y` are in column
1647:         `k`.
1648: 
1649:     [residuals, rank, singular_values, rcond] : list
1650:         These values are only returned if `full` = True
1651: 
1652:         resid -- sum of squared residuals of the least squares fit
1653:         rank -- the numerical rank of the scaled Vandermonde matrix
1654:         sv -- singular values of the scaled Vandermonde matrix
1655:         rcond -- value of `rcond`.
1656: 
1657:         For more details, see `linalg.lstsq`.
1658: 
1659:     Warns
1660:     -----
1661:     RankWarning
1662:         The rank of the coefficient matrix in the least-squares fit is
1663:         deficient. The warning is only raised if `full` = False.  The
1664:         warnings can be turned off by
1665: 
1666:         >>> import warnings
1667:         >>> warnings.simplefilter('ignore', RankWarning)
1668: 
1669:     See Also
1670:     --------
1671:     polyfit, legfit, lagfit, hermfit, hermefit
1672:     chebval : Evaluates a Chebyshev series.
1673:     chebvander : Vandermonde matrix of Chebyshev series.
1674:     chebweight : Chebyshev weight function.
1675:     linalg.lstsq : Computes a least-squares fit from the matrix.
1676:     scipy.interpolate.UnivariateSpline : Computes spline fits.
1677: 
1678:     Notes
1679:     -----
1680:     The solution is the coefficients of the Chebyshev series `p` that
1681:     minimizes the sum of the weighted squared errors
1682: 
1683:     .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
1684: 
1685:     where :math:`w_j` are the weights. This problem is solved by setting up
1686:     as the (typically) overdetermined matrix equation
1687: 
1688:     .. math:: V(x) * c = w * y,
1689: 
1690:     where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
1691:     coefficients to be solved for, `w` are the weights, and `y` are the
1692:     observed values.  This equation is then solved using the singular value
1693:     decomposition of `V`.
1694: 
1695:     If some of the singular values of `V` are so small that they are
1696:     neglected, then a `RankWarning` will be issued. This means that the
1697:     coefficient values may be poorly determined. Using a lower order fit
1698:     will usually get rid of the warning.  The `rcond` parameter can also be
1699:     set to a value smaller than its default, but the resulting fit may be
1700:     spurious and have large contributions from roundoff error.
1701: 
1702:     Fits using Chebyshev series are usually better conditioned than fits
1703:     using power series, but much can depend on the distribution of the
1704:     sample points and the smoothness of the data. If the quality of the fit
1705:     is inadequate splines may be a good alternative.
1706: 
1707:     References
1708:     ----------
1709:     .. [1] Wikipedia, "Curve fitting",
1710:            http://en.wikipedia.org/wiki/Curve_fitting
1711: 
1712:     Examples
1713:     --------
1714: 
1715:     '''
1716:     x = np.asarray(x) + 0.0
1717:     y = np.asarray(y) + 0.0
1718:     deg = np.asarray(deg)
1719: 
1720:     # check arguments.
1721:     if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
1722:         raise TypeError("deg must be an int or non-empty 1-D array of int")
1723:     if deg.min() < 0:
1724:         raise ValueError("expected deg >= 0")
1725:     if x.ndim != 1:
1726:         raise TypeError("expected 1D vector for x")
1727:     if x.size == 0:
1728:         raise TypeError("expected non-empty vector for x")
1729:     if y.ndim < 1 or y.ndim > 2:
1730:         raise TypeError("expected 1D or 2D array for y")
1731:     if len(x) != len(y):
1732:         raise TypeError("expected x and y to have same length")
1733: 
1734:     if deg.ndim == 0:
1735:         lmax = deg
1736:         order = lmax + 1
1737:         van = chebvander(x, lmax)
1738:     else:
1739:         deg = np.sort(deg)
1740:         lmax = deg[-1]
1741:         order = len(deg)
1742:         van = chebvander(x, lmax)[:, deg]
1743: 
1744:     # set up the least squares matrices in transposed form
1745:     lhs = van.T
1746:     rhs = y.T
1747:     if w is not None:
1748:         w = np.asarray(w) + 0.0
1749:         if w.ndim != 1:
1750:             raise TypeError("expected 1D vector for w")
1751:         if len(x) != len(w):
1752:             raise TypeError("expected x and w to have same length")
1753:         # apply weights. Don't use inplace operations as they
1754:         # can cause problems with NA.
1755:         lhs = lhs * w
1756:         rhs = rhs * w
1757: 
1758:     # set rcond
1759:     if rcond is None:
1760:         rcond = len(x)*np.finfo(x.dtype).eps
1761: 
1762:     # Determine the norms of the design matrix columns.
1763:     if issubclass(lhs.dtype.type, np.complexfloating):
1764:         scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
1765:     else:
1766:         scl = np.sqrt(np.square(lhs).sum(1))
1767:     scl[scl == 0] = 1
1768: 
1769:     # Solve the least squares problem.
1770:     c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
1771:     c = (c.T/scl).T
1772: 
1773:     # Expand c to include non-fitted coefficients which are set to zero
1774:     if deg.ndim > 0:
1775:         if c.ndim == 2:
1776:             cc = np.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
1777:         else:
1778:             cc = np.zeros(lmax + 1, dtype=c.dtype)
1779:         cc[deg] = c
1780:         c = cc
1781: 
1782:     # warn on rank reduction
1783:     if rank != order and not full:
1784:         msg = "The fit may be poorly conditioned"
1785:         warnings.warn(msg, pu.RankWarning)
1786: 
1787:     if full:
1788:         return c, [resids, rank, s, rcond]
1789:     else:
1790:         return c
1791: 
1792: 
1793: def chebcompanion(c):
1794:     '''Return the scaled companion matrix of c.
1795: 
1796:     The basis polynomials are scaled so that the companion matrix is
1797:     symmetric when `c` is a Chebyshev basis polynomial. This provides
1798:     better eigenvalue estimates than the unscaled case and for basis
1799:     polynomials the eigenvalues are guaranteed to be real if
1800:     `numpy.linalg.eigvalsh` is used to obtain them.
1801: 
1802:     Parameters
1803:     ----------
1804:     c : array_like
1805:         1-D array of Chebyshev series coefficients ordered from low to high
1806:         degree.
1807: 
1808:     Returns
1809:     -------
1810:     mat : ndarray
1811:         Scaled companion matrix of dimensions (deg, deg).
1812: 
1813:     Notes
1814:     -----
1815: 
1816:     .. versionadded::1.7.0
1817: 
1818:     '''
1819:     # c is a trimmed copy
1820:     [c] = pu.as_series([c])
1821:     if len(c) < 2:
1822:         raise ValueError('Series must have maximum degree of at least 1.')
1823:     if len(c) == 2:
1824:         return np.array([[-c[0]/c[1]]])
1825: 
1826:     n = len(c) - 1
1827:     mat = np.zeros((n, n), dtype=c.dtype)
1828:     scl = np.array([1.] + [np.sqrt(.5)]*(n-1))
1829:     top = mat.reshape(-1)[1::n+1]
1830:     bot = mat.reshape(-1)[n::n+1]
1831:     top[0] = np.sqrt(.5)
1832:     top[1:] = 1/2
1833:     bot[...] = top
1834:     mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*.5
1835:     return mat
1836: 
1837: 
1838: def chebroots(c):
1839:     '''
1840:     Compute the roots of a Chebyshev series.
1841: 
1842:     Return the roots (a.k.a. "zeros") of the polynomial
1843: 
1844:     .. math:: p(x) = \\sum_i c[i] * T_i(x).
1845: 
1846:     Parameters
1847:     ----------
1848:     c : 1-D array_like
1849:         1-D array of coefficients.
1850: 
1851:     Returns
1852:     -------
1853:     out : ndarray
1854:         Array of the roots of the series. If all the roots are real,
1855:         then `out` is also real, otherwise it is complex.
1856: 
1857:     See Also
1858:     --------
1859:     polyroots, legroots, lagroots, hermroots, hermeroots
1860: 
1861:     Notes
1862:     -----
1863:     The root estimates are obtained as the eigenvalues of the companion
1864:     matrix, Roots far from the origin of the complex plane may have large
1865:     errors due to the numerical instability of the series for such
1866:     values. Roots with multiplicity greater than 1 will also show larger
1867:     errors as the value of the series near such points is relatively
1868:     insensitive to errors in the roots. Isolated roots near the origin can
1869:     be improved by a few iterations of Newton's method.
1870: 
1871:     The Chebyshev series basis polynomials aren't powers of `x` so the
1872:     results of this function may seem unintuitive.
1873: 
1874:     Examples
1875:     --------
1876:     >>> import numpy.polynomial.chebyshev as cheb
1877:     >>> cheb.chebroots((-1, 1,-1, 1)) # T3 - T2 + T1 - T0 has real roots
1878:     array([ -5.00000000e-01,   2.60860684e-17,   1.00000000e+00])
1879: 
1880:     '''
1881:     # c is a trimmed copy
1882:     [c] = pu.as_series([c])
1883:     if len(c) < 2:
1884:         return np.array([], dtype=c.dtype)
1885:     if len(c) == 2:
1886:         return np.array([-c[0]/c[1]])
1887: 
1888:     m = chebcompanion(c)
1889:     r = la.eigvals(m)
1890:     r.sort()
1891:     return r
1892: 
1893: 
1894: def chebgauss(deg):
1895:     '''
1896:     Gauss-Chebyshev quadrature.
1897: 
1898:     Computes the sample points and weights for Gauss-Chebyshev quadrature.
1899:     These sample points and weights will correctly integrate polynomials of
1900:     degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
1901:     the weight function :math:`f(x) = 1/\sqrt{1 - x^2}`.
1902: 
1903:     Parameters
1904:     ----------
1905:     deg : int
1906:         Number of sample points and weights. It must be >= 1.
1907: 
1908:     Returns
1909:     -------
1910:     x : ndarray
1911:         1-D ndarray containing the sample points.
1912:     y : ndarray
1913:         1-D ndarray containing the weights.
1914: 
1915:     Notes
1916:     -----
1917: 
1918:     .. versionadded:: 1.7.0
1919: 
1920:     The results have only been tested up to degree 100, higher degrees may
1921:     be problematic. For Gauss-Chebyshev there are closed form solutions for
1922:     the sample points and weights. If n = `deg`, then
1923: 
1924:     .. math:: x_i = \cos(\pi (2 i - 1) / (2 n))
1925: 
1926:     .. math:: w_i = \pi / n
1927: 
1928:     '''
1929:     ideg = int(deg)
1930:     if ideg != deg or ideg < 1:
1931:         raise ValueError("deg must be a non-negative integer")
1932: 
1933:     x = np.cos(np.pi * np.arange(1, 2*ideg, 2) / (2.0*ideg))
1934:     w = np.ones(ideg)*(np.pi/ideg)
1935: 
1936:     return x, w
1937: 
1938: 
1939: def chebweight(x):
1940:     '''
1941:     The weight function of the Chebyshev polynomials.
1942: 
1943:     The weight function is :math:`1/\sqrt{1 - x^2}` and the interval of
1944:     integration is :math:`[-1, 1]`. The Chebyshev polynomials are
1945:     orthogonal, but not normalized, with respect to this weight function.
1946: 
1947:     Parameters
1948:     ----------
1949:     x : array_like
1950:        Values at which the weight function will be computed.
1951: 
1952:     Returns
1953:     -------
1954:     w : ndarray
1955:        The weight function at `x`.
1956: 
1957:     Notes
1958:     -----
1959: 
1960:     .. versionadded:: 1.7.0
1961: 
1962:     '''
1963:     w = 1./(np.sqrt(1. + x) * np.sqrt(1. - x))
1964:     return w
1965: 
1966: 
1967: def chebpts1(npts):
1968:     '''
1969:     Chebyshev points of the first kind.
1970: 
1971:     The Chebyshev points of the first kind are the points ``cos(x)``,
1972:     where ``x = [pi*(k + .5)/npts for k in range(npts)]``.
1973: 
1974:     Parameters
1975:     ----------
1976:     npts : int
1977:         Number of sample points desired.
1978: 
1979:     Returns
1980:     -------
1981:     pts : ndarray
1982:         The Chebyshev points of the first kind.
1983: 
1984:     See Also
1985:     --------
1986:     chebpts2
1987: 
1988:     Notes
1989:     -----
1990: 
1991:     .. versionadded:: 1.5.0
1992: 
1993:     '''
1994:     _npts = int(npts)
1995:     if _npts != npts:
1996:         raise ValueError("npts must be integer")
1997:     if _npts < 1:
1998:         raise ValueError("npts must be >= 1")
1999: 
2000:     x = np.linspace(-np.pi, 0, _npts, endpoint=False) + np.pi/(2*_npts)
2001:     return np.cos(x)
2002: 
2003: 
2004: def chebpts2(npts):
2005:     '''
2006:     Chebyshev points of the second kind.
2007: 
2008:     The Chebyshev points of the second kind are the points ``cos(x)``,
2009:     where ``x = [pi*k/(npts - 1) for k in range(npts)]``.
2010: 
2011:     Parameters
2012:     ----------
2013:     npts : int
2014:         Number of sample points desired.
2015: 
2016:     Returns
2017:     -------
2018:     pts : ndarray
2019:         The Chebyshev points of the second kind.
2020: 
2021:     Notes
2022:     -----
2023: 
2024:     .. versionadded:: 1.5.0
2025: 
2026:     '''
2027:     _npts = int(npts)
2028:     if _npts != npts:
2029:         raise ValueError("npts must be integer")
2030:     if _npts < 2:
2031:         raise ValueError("npts must be >= 2")
2032: 
2033:     x = np.linspace(-np.pi, 0, _npts)
2034:     return np.cos(x)
2035: 
2036: 
2037: #
2038: # Chebyshev series class
2039: #
2040: 
2041: class Chebyshev(ABCPolyBase):
2042:     '''A Chebyshev series class.
2043: 
2044:     The Chebyshev class provides the standard Python numerical methods
2045:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
2046:     methods listed below.
2047: 
2048:     Parameters
2049:     ----------
2050:     coef : array_like
2051:         Chebyshev coefficients in order of increasing degree, i.e.,
2052:         ``(1, 2, 3)`` gives ``1*T_0(x) + 2*T_1(x) + 3*T_2(x)``.
2053:     domain : (2,) array_like, optional
2054:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
2055:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
2056:         The default value is [-1, 1].
2057:     window : (2,) array_like, optional
2058:         Window, see `domain` for its use. The default value is [-1, 1].
2059: 
2060:         .. versionadded:: 1.6.0
2061: 
2062:     '''
2063:     # Virtual Functions
2064:     _add = staticmethod(chebadd)
2065:     _sub = staticmethod(chebsub)
2066:     _mul = staticmethod(chebmul)
2067:     _div = staticmethod(chebdiv)
2068:     _pow = staticmethod(chebpow)
2069:     _val = staticmethod(chebval)
2070:     _int = staticmethod(chebint)
2071:     _der = staticmethod(chebder)
2072:     _fit = staticmethod(chebfit)
2073:     _line = staticmethod(chebline)
2074:     _roots = staticmethod(chebroots)
2075:     _fromroots = staticmethod(chebfromroots)
2076: 
2077:     # Virtual properties
2078:     nickname = 'cheb'
2079:     domain = np.array(chebdomain)
2080:     window = np.array(chebdomain)
2081: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_161934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\nObjects for dealing with Chebyshev series.\n\nThis module provides a number of objects (mostly functions) useful for\ndealing with Chebyshev series, including a `Chebyshev` class that\nencapsulates the usual arithmetic operations.  (General information\non how this module represents and works with such polynomials is in the\ndocstring for its "parent" sub-package, `numpy.polynomial`).\n\nConstants\n---------\n- `chebdomain` -- Chebyshev series default domain, [-1,1].\n- `chebzero` -- (Coefficients of the) Chebyshev series that evaluates\n  identically to 0.\n- `chebone` -- (Coefficients of the) Chebyshev series that evaluates\n  identically to 1.\n- `chebx` -- (Coefficients of the) Chebyshev series for the identity map,\n  ``f(x) = x``.\n\nArithmetic\n----------\n- `chebadd` -- add two Chebyshev series.\n- `chebsub` -- subtract one Chebyshev series from another.\n- `chebmul` -- multiply two Chebyshev series.\n- `chebdiv` -- divide one Chebyshev series by another.\n- `chebpow` -- raise a Chebyshev series to an positive integer power\n- `chebval` -- evaluate a Chebyshev series at given points.\n- `chebval2d` -- evaluate a 2D Chebyshev series at given points.\n- `chebval3d` -- evaluate a 3D Chebyshev series at given points.\n- `chebgrid2d` -- evaluate a 2D Chebyshev series on a Cartesian product.\n- `chebgrid3d` -- evaluate a 3D Chebyshev series on a Cartesian product.\n\nCalculus\n--------\n- `chebder` -- differentiate a Chebyshev series.\n- `chebint` -- integrate a Chebyshev series.\n\nMisc Functions\n--------------\n- `chebfromroots` -- create a Chebyshev series with specified roots.\n- `chebroots` -- find the roots of a Chebyshev series.\n- `chebvander` -- Vandermonde-like matrix for Chebyshev polynomials.\n- `chebvander2d` -- Vandermonde-like matrix for 2D power series.\n- `chebvander3d` -- Vandermonde-like matrix for 3D power series.\n- `chebgauss` -- Gauss-Chebyshev quadrature, points and weights.\n- `chebweight` -- Chebyshev weight function.\n- `chebcompanion` -- symmetrized companion matrix in Chebyshev form.\n- `chebfit` -- least-squares fit returning a Chebyshev series.\n- `chebpts1` -- Chebyshev points of the first kind.\n- `chebpts2` -- Chebyshev points of the second kind.\n- `chebtrim` -- trim leading coefficients from a Chebyshev series.\n- `chebline` -- Chebyshev series representing given straight line.\n- `cheb2poly` -- convert a Chebyshev series to a polynomial.\n- `poly2cheb` -- convert a polynomial to a Chebyshev series.\n\nClasses\n-------\n- `Chebyshev` -- A Chebyshev series class.\n\nSee also\n--------\n`numpy.polynomial`\n\nNotes\n-----\nThe implementations of multiplication, division, integration, and\ndifferentiation use the algebraic identities [1]_:\n\n.. math ::\n    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\\n    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.\n\nwhere\n\n.. math :: x = \\frac{z + z^{-1}}{2}.\n\nThese identities allow a Chebyshev series to be expressed as a finite,\nsymmetric Laurent series.  In this module, this sort of Laurent series\nis referred to as a "z-series."\n\nReferences\n----------\n.. [1] A. T. Benjamin, et al., "Combinatorial Trigonometry with Chebyshev\n  Polynomials," *Journal of Statistical Planning and Inference 14*, 2008\n  (preprint: http://www.math.hmc.edu/~benjamin/papers/CombTrig.pdf, pg. 4)\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 90, 0))

# 'import warnings' statement (line 90)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 0))

# 'import numpy' statement (line 91)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_161935 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'numpy')

if (type(import_161935) is not StypyTypeError):

    if (import_161935 != 'pyd_module'):
        __import__(import_161935)
        sys_modules_161936 = sys.modules[import_161935]
        import_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'np', sys_modules_161936.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'numpy', import_161935)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 92, 0))

# 'import numpy.linalg' statement (line 92)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_161937 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'numpy.linalg')

if (type(import_161937) is not StypyTypeError):

    if (import_161937 != 'pyd_module'):
        __import__(import_161937)
        sys_modules_161938 = sys.modules[import_161937]
        import_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'la', sys_modules_161938.module_type_store, module_type_store)
    else:
        import numpy.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'la', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'numpy.linalg', import_161937)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 0))

# 'from numpy.polynomial import pu' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_161939 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.polynomial')

if (type(import_161939) is not StypyTypeError):

    if (import_161939 != 'pyd_module'):
        __import__(import_161939)
        sys_modules_161940 = sys.modules[import_161939]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.polynomial', sys_modules_161940.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 0), __file__, sys_modules_161940, sys_modules_161940.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.polynomial', import_161939)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 95, 0))

# 'from numpy.polynomial._polybase import ABCPolyBase' statement (line 95)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_161941 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 95, 0), 'numpy.polynomial._polybase')

if (type(import_161941) is not StypyTypeError):

    if (import_161941 != 'pyd_module'):
        __import__(import_161941)
        sys_modules_161942 = sys.modules[import_161941]
        import_from_module(stypy.reporting.localization.Localization(__file__, 95, 0), 'numpy.polynomial._polybase', sys_modules_161942.module_type_store, module_type_store, ['ABCPolyBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 95, 0), __file__, sys_modules_161942, sys_modules_161942.module_type_store, module_type_store)
    else:
        from numpy.polynomial._polybase import ABCPolyBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 95, 0), 'numpy.polynomial._polybase', None, module_type_store, ['ABCPolyBase'], [ABCPolyBase])

else:
    # Assigning a type to the variable 'numpy.polynomial._polybase' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'numpy.polynomial._polybase', import_161941)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 97):

# Assigning a List to a Name (line 97):
__all__ = ['chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline', 'chebadd', 'chebsub', 'chebmulx', 'chebmul', 'chebdiv', 'chebpow', 'chebval', 'chebder', 'chebint', 'cheb2poly', 'poly2cheb', 'chebfromroots', 'chebvander', 'chebfit', 'chebtrim', 'chebroots', 'chebpts1', 'chebpts2', 'Chebyshev', 'chebval2d', 'chebval3d', 'chebgrid2d', 'chebgrid3d', 'chebvander2d', 'chebvander3d', 'chebcompanion', 'chebgauss', 'chebweight']
module_type_store.set_exportable_members(['chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline', 'chebadd', 'chebsub', 'chebmulx', 'chebmul', 'chebdiv', 'chebpow', 'chebval', 'chebder', 'chebint', 'cheb2poly', 'poly2cheb', 'chebfromroots', 'chebvander', 'chebfit', 'chebtrim', 'chebroots', 'chebpts1', 'chebpts2', 'Chebyshev', 'chebval2d', 'chebval3d', 'chebgrid2d', 'chebgrid3d', 'chebvander2d', 'chebvander3d', 'chebcompanion', 'chebgauss', 'chebweight'])

# Obtaining an instance of the builtin type 'list' (line 97)
list_161943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 97)
# Adding element type (line 97)
str_161944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'str', 'chebzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161944)
# Adding element type (line 97)
str_161945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'str', 'chebone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161945)
# Adding element type (line 97)
str_161946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 27), 'str', 'chebx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161946)
# Adding element type (line 97)
str_161947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 36), 'str', 'chebdomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161947)
# Adding element type (line 97)
str_161948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 50), 'str', 'chebline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161948)
# Adding element type (line 97)
str_161949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 62), 'str', 'chebadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161949)
# Adding element type (line 97)
str_161950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'str', 'chebsub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161950)
# Adding element type (line 97)
str_161951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 15), 'str', 'chebmulx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161951)
# Adding element type (line 97)
str_161952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 27), 'str', 'chebmul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161952)
# Adding element type (line 97)
str_161953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 38), 'str', 'chebdiv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161953)
# Adding element type (line 97)
str_161954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 49), 'str', 'chebpow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161954)
# Adding element type (line 97)
str_161955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 60), 'str', 'chebval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161955)
# Adding element type (line 97)
str_161956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'str', 'chebder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161956)
# Adding element type (line 97)
str_161957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'str', 'chebint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161957)
# Adding element type (line 97)
str_161958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 26), 'str', 'cheb2poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161958)
# Adding element type (line 97)
str_161959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 39), 'str', 'poly2cheb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161959)
# Adding element type (line 97)
str_161960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 52), 'str', 'chebfromroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161960)
# Adding element type (line 97)
str_161961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'str', 'chebvander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161961)
# Adding element type (line 97)
str_161962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'str', 'chebfit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161962)
# Adding element type (line 97)
str_161963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'str', 'chebtrim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161963)
# Adding element type (line 97)
str_161964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 41), 'str', 'chebroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161964)
# Adding element type (line 97)
str_161965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 54), 'str', 'chebpts1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161965)
# Adding element type (line 97)
str_161966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'str', 'chebpts2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161966)
# Adding element type (line 97)
str_161967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 16), 'str', 'Chebyshev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161967)
# Adding element type (line 97)
str_161968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'str', 'chebval2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161968)
# Adding element type (line 97)
str_161969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'str', 'chebval3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161969)
# Adding element type (line 97)
str_161970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 55), 'str', 'chebgrid2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161970)
# Adding element type (line 97)
str_161971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'str', 'chebgrid3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161971)
# Adding element type (line 97)
str_161972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'str', 'chebvander2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161972)
# Adding element type (line 97)
str_161973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'str', 'chebvander3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161973)
# Adding element type (line 97)
str_161974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 50), 'str', 'chebcompanion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161974)
# Adding element type (line 97)
str_161975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'str', 'chebgauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161975)
# Adding element type (line 97)
str_161976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'str', 'chebweight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 10), list_161943, str_161976)

# Assigning a type to the variable '__all__' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), '__all__', list_161943)

# Assigning a Attribute to a Name (line 106):

# Assigning a Attribute to a Name (line 106):
# Getting the type of 'pu' (line 106)
pu_161977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'pu')
# Obtaining the member 'trimcoef' of a type (line 106)
trimcoef_161978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), pu_161977, 'trimcoef')
# Assigning a type to the variable 'chebtrim' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'chebtrim', trimcoef_161978)

@norecursion
def _cseries_to_zseries(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_cseries_to_zseries'
    module_type_store = module_type_store.open_function_context('_cseries_to_zseries', 113, 0, False)
    
    # Passed parameters checking function
    _cseries_to_zseries.stypy_localization = localization
    _cseries_to_zseries.stypy_type_of_self = None
    _cseries_to_zseries.stypy_type_store = module_type_store
    _cseries_to_zseries.stypy_function_name = '_cseries_to_zseries'
    _cseries_to_zseries.stypy_param_names_list = ['c']
    _cseries_to_zseries.stypy_varargs_param_name = None
    _cseries_to_zseries.stypy_kwargs_param_name = None
    _cseries_to_zseries.stypy_call_defaults = defaults
    _cseries_to_zseries.stypy_call_varargs = varargs
    _cseries_to_zseries.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cseries_to_zseries', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cseries_to_zseries', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cseries_to_zseries(...)' code ##################

    str_161979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, (-1)), 'str', 'Covert Chebyshev series to z-series.\n\n    Covert a Chebyshev series to the equivalent z-series. The result is\n    never an empty array. The dtype of the return is the same as that of\n    the input. No checks are run on the arguments as this routine is for\n    internal use.\n\n    Parameters\n    ----------\n    c : 1-D ndarray\n        Chebyshev coefficients, ordered from low to high\n\n    Returns\n    -------\n    zs : 1-D ndarray\n        Odd length symmetric z-series, ordered from  low to high.\n\n    ')
    
    # Assigning a Attribute to a Name (line 132):
    
    # Assigning a Attribute to a Name (line 132):
    # Getting the type of 'c' (line 132)
    c_161980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'c')
    # Obtaining the member 'size' of a type (line 132)
    size_161981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), c_161980, 'size')
    # Assigning a type to the variable 'n' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'n', size_161981)
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to zeros(...): (line 133)
    # Processing the call arguments (line 133)
    int_161984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'int')
    # Getting the type of 'n' (line 133)
    n_161985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'n', False)
    # Applying the binary operator '*' (line 133)
    result_mul_161986 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 18), '*', int_161984, n_161985)
    
    int_161987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'int')
    # Applying the binary operator '-' (line 133)
    result_sub_161988 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 18), '-', result_mul_161986, int_161987)
    
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'c' (line 133)
    c_161989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'c', False)
    # Obtaining the member 'dtype' of a type (line 133)
    dtype_161990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 31), c_161989, 'dtype')
    keyword_161991 = dtype_161990
    kwargs_161992 = {'dtype': keyword_161991}
    # Getting the type of 'np' (line 133)
    np_161982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 133)
    zeros_161983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 9), np_161982, 'zeros')
    # Calling zeros(args, kwargs) (line 133)
    zeros_call_result_161993 = invoke(stypy.reporting.localization.Localization(__file__, 133, 9), zeros_161983, *[result_sub_161988], **kwargs_161992)
    
    # Assigning a type to the variable 'zs' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'zs', zeros_call_result_161993)
    
    # Assigning a BinOp to a Subscript (line 134):
    
    # Assigning a BinOp to a Subscript (line 134):
    # Getting the type of 'c' (line 134)
    c_161994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'c')
    int_161995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'int')
    # Applying the binary operator 'div' (line 134)
    result_div_161996 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), 'div', c_161994, int_161995)
    
    # Getting the type of 'zs' (line 134)
    zs_161997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'zs')
    # Getting the type of 'n' (line 134)
    n_161998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 7), 'n')
    int_161999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 9), 'int')
    # Applying the binary operator '-' (line 134)
    result_sub_162000 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 7), '-', n_161998, int_161999)
    
    slice_162001 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 4), result_sub_162000, None, None)
    # Storing an element on a container (line 134)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 4), zs_161997, (slice_162001, result_div_161996))
    # Getting the type of 'zs' (line 135)
    zs_162002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'zs')
    
    # Obtaining the type of the subscript
    int_162003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'int')
    slice_162004 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 135, 16), None, None, int_162003)
    # Getting the type of 'zs' (line 135)
    zs_162005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'zs')
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___162006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), zs_162005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_162007 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getitem___162006, slice_162004)
    
    # Applying the binary operator '+' (line 135)
    result_add_162008 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 11), '+', zs_162002, subscript_call_result_162007)
    
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type', result_add_162008)
    
    # ################# End of '_cseries_to_zseries(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cseries_to_zseries' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_162009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cseries_to_zseries'
    return stypy_return_type_162009

# Assigning a type to the variable '_cseries_to_zseries' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), '_cseries_to_zseries', _cseries_to_zseries)

@norecursion
def _zseries_to_cseries(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zseries_to_cseries'
    module_type_store = module_type_store.open_function_context('_zseries_to_cseries', 138, 0, False)
    
    # Passed parameters checking function
    _zseries_to_cseries.stypy_localization = localization
    _zseries_to_cseries.stypy_type_of_self = None
    _zseries_to_cseries.stypy_type_store = module_type_store
    _zseries_to_cseries.stypy_function_name = '_zseries_to_cseries'
    _zseries_to_cseries.stypy_param_names_list = ['zs']
    _zseries_to_cseries.stypy_varargs_param_name = None
    _zseries_to_cseries.stypy_kwargs_param_name = None
    _zseries_to_cseries.stypy_call_defaults = defaults
    _zseries_to_cseries.stypy_call_varargs = varargs
    _zseries_to_cseries.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zseries_to_cseries', ['zs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zseries_to_cseries', localization, ['zs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zseries_to_cseries(...)' code ##################

    str_162010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', 'Covert z-series to a Chebyshev series.\n\n    Covert a z series to the equivalent Chebyshev series. The result is\n    never an empty array. The dtype of the return is the same as that of\n    the input. No checks are run on the arguments as this routine is for\n    internal use.\n\n    Parameters\n    ----------\n    zs : 1-D ndarray\n        Odd length symmetric z-series, ordered from  low to high.\n\n    Returns\n    -------\n    c : 1-D ndarray\n        Chebyshev coefficients, ordered from  low to high.\n\n    ')
    
    # Assigning a BinOp to a Name (line 157):
    
    # Assigning a BinOp to a Name (line 157):
    # Getting the type of 'zs' (line 157)
    zs_162011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 9), 'zs')
    # Obtaining the member 'size' of a type (line 157)
    size_162012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 9), zs_162011, 'size')
    int_162013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'int')
    # Applying the binary operator '+' (line 157)
    result_add_162014 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 9), '+', size_162012, int_162013)
    
    int_162015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
    # Applying the binary operator '//' (line 157)
    result_floordiv_162016 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 8), '//', result_add_162014, int_162015)
    
    # Assigning a type to the variable 'n' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'n', result_floordiv_162016)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to copy(...): (line 158)
    # Processing the call keyword arguments (line 158)
    kwargs_162025 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 158)
    n_162017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'n', False)
    int_162018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 13), 'int')
    # Applying the binary operator '-' (line 158)
    result_sub_162019 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), '-', n_162017, int_162018)
    
    slice_162020 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 8), result_sub_162019, None, None)
    # Getting the type of 'zs' (line 158)
    zs_162021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'zs', False)
    # Obtaining the member '__getitem__' of a type (line 158)
    getitem___162022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), zs_162021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 158)
    subscript_call_result_162023 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___162022, slice_162020)
    
    # Obtaining the member 'copy' of a type (line 158)
    copy_162024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), subscript_call_result_162023, 'copy')
    # Calling copy(args, kwargs) (line 158)
    copy_call_result_162026 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), copy_162024, *[], **kwargs_162025)
    
    # Assigning a type to the variable 'c' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'c', copy_call_result_162026)
    
    # Getting the type of 'c' (line 159)
    c_162027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'c')
    
    # Obtaining the type of the subscript
    int_162028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 6), 'int')
    # Getting the type of 'n' (line 159)
    n_162029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'n')
    slice_162030 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 4), int_162028, n_162029, None)
    # Getting the type of 'c' (line 159)
    c_162031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'c')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___162032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), c_162031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_162033 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___162032, slice_162030)
    
    int_162034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 14), 'int')
    # Applying the binary operator '*=' (line 159)
    result_imul_162035 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 4), '*=', subscript_call_result_162033, int_162034)
    # Getting the type of 'c' (line 159)
    c_162036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'c')
    int_162037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 6), 'int')
    # Getting the type of 'n' (line 159)
    n_162038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'n')
    slice_162039 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 4), int_162037, n_162038, None)
    # Storing an element on a container (line 159)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 4), c_162036, (slice_162039, result_imul_162035))
    
    # Getting the type of 'c' (line 160)
    c_162040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type', c_162040)
    
    # ################# End of '_zseries_to_cseries(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zseries_to_cseries' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_162041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zseries_to_cseries'
    return stypy_return_type_162041

# Assigning a type to the variable '_zseries_to_cseries' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), '_zseries_to_cseries', _zseries_to_cseries)

@norecursion
def _zseries_mul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zseries_mul'
    module_type_store = module_type_store.open_function_context('_zseries_mul', 163, 0, False)
    
    # Passed parameters checking function
    _zseries_mul.stypy_localization = localization
    _zseries_mul.stypy_type_of_self = None
    _zseries_mul.stypy_type_store = module_type_store
    _zseries_mul.stypy_function_name = '_zseries_mul'
    _zseries_mul.stypy_param_names_list = ['z1', 'z2']
    _zseries_mul.stypy_varargs_param_name = None
    _zseries_mul.stypy_kwargs_param_name = None
    _zseries_mul.stypy_call_defaults = defaults
    _zseries_mul.stypy_call_varargs = varargs
    _zseries_mul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zseries_mul', ['z1', 'z2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zseries_mul', localization, ['z1', 'z2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zseries_mul(...)' code ##################

    str_162042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, (-1)), 'str', 'Multiply two z-series.\n\n    Multiply two z-series to produce a z-series.\n\n    Parameters\n    ----------\n    z1, z2 : 1-D ndarray\n        The arrays must be 1-D but this is not checked.\n\n    Returns\n    -------\n    product : 1-D ndarray\n        The product z-series.\n\n    Notes\n    -----\n    This is simply convolution. If symmetric/anti-symmetric z-series are\n    denoted by S/A then the following rules apply:\n\n    S*S, A*A -> S\n    S*A, A*S -> A\n\n    ')
    
    # Call to convolve(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'z1' (line 187)
    z1_162045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'z1', False)
    # Getting the type of 'z2' (line 187)
    z2_162046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'z2', False)
    # Processing the call keyword arguments (line 187)
    kwargs_162047 = {}
    # Getting the type of 'np' (line 187)
    np_162043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'np', False)
    # Obtaining the member 'convolve' of a type (line 187)
    convolve_162044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), np_162043, 'convolve')
    # Calling convolve(args, kwargs) (line 187)
    convolve_call_result_162048 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), convolve_162044, *[z1_162045, z2_162046], **kwargs_162047)
    
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', convolve_call_result_162048)
    
    # ################# End of '_zseries_mul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zseries_mul' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_162049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162049)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zseries_mul'
    return stypy_return_type_162049

# Assigning a type to the variable '_zseries_mul' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), '_zseries_mul', _zseries_mul)

@norecursion
def _zseries_div(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zseries_div'
    module_type_store = module_type_store.open_function_context('_zseries_div', 190, 0, False)
    
    # Passed parameters checking function
    _zseries_div.stypy_localization = localization
    _zseries_div.stypy_type_of_self = None
    _zseries_div.stypy_type_store = module_type_store
    _zseries_div.stypy_function_name = '_zseries_div'
    _zseries_div.stypy_param_names_list = ['z1', 'z2']
    _zseries_div.stypy_varargs_param_name = None
    _zseries_div.stypy_kwargs_param_name = None
    _zseries_div.stypy_call_defaults = defaults
    _zseries_div.stypy_call_varargs = varargs
    _zseries_div.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zseries_div', ['z1', 'z2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zseries_div', localization, ['z1', 'z2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zseries_div(...)' code ##################

    str_162050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, (-1)), 'str', 'Divide the first z-series by the second.\n\n    Divide `z1` by `z2` and return the quotient and remainder as z-series.\n    Warning: this implementation only applies when both z1 and z2 have the\n    same symmetry, which is sufficient for present purposes.\n\n    Parameters\n    ----------\n    z1, z2 : 1-D ndarray\n        The arrays must be 1-D and have the same symmetry, but this is not\n        checked.\n\n    Returns\n    -------\n\n    (quotient, remainder) : 1-D ndarrays\n        Quotient and remainder as z-series.\n\n    Notes\n    -----\n    This is not the same as polynomial division on account of the desired form\n    of the remainder. If symmetric/anti-symmetric z-series are denoted by S/A\n    then the following rules apply:\n\n    S/S -> S,S\n    A/A -> S,A\n\n    The restriction to types of the same symmetry could be fixed but seems like\n    unneeded generality. There is no natural form for the remainder in the case\n    where there is no symmetry.\n\n    ')
    
    # Assigning a Call to a Name (line 223):
    
    # Assigning a Call to a Name (line 223):
    
    # Call to copy(...): (line 223)
    # Processing the call keyword arguments (line 223)
    kwargs_162053 = {}
    # Getting the type of 'z1' (line 223)
    z1_162051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'z1', False)
    # Obtaining the member 'copy' of a type (line 223)
    copy_162052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 9), z1_162051, 'copy')
    # Calling copy(args, kwargs) (line 223)
    copy_call_result_162054 = invoke(stypy.reporting.localization.Localization(__file__, 223, 9), copy_162052, *[], **kwargs_162053)
    
    # Assigning a type to the variable 'z1' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'z1', copy_call_result_162054)
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to copy(...): (line 224)
    # Processing the call keyword arguments (line 224)
    kwargs_162057 = {}
    # Getting the type of 'z2' (line 224)
    z2_162055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'z2', False)
    # Obtaining the member 'copy' of a type (line 224)
    copy_162056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 9), z2_162055, 'copy')
    # Calling copy(args, kwargs) (line 224)
    copy_call_result_162058 = invoke(stypy.reporting.localization.Localization(__file__, 224, 9), copy_162056, *[], **kwargs_162057)
    
    # Assigning a type to the variable 'z2' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'z2', copy_call_result_162058)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to len(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'z1' (line 225)
    z1_162060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'z1', False)
    # Processing the call keyword arguments (line 225)
    kwargs_162061 = {}
    # Getting the type of 'len' (line 225)
    len_162059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'len', False)
    # Calling len(args, kwargs) (line 225)
    len_call_result_162062 = invoke(stypy.reporting.localization.Localization(__file__, 225, 11), len_162059, *[z1_162060], **kwargs_162061)
    
    # Assigning a type to the variable 'len1' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'len1', len_call_result_162062)
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to len(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'z2' (line 226)
    z2_162064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'z2', False)
    # Processing the call keyword arguments (line 226)
    kwargs_162065 = {}
    # Getting the type of 'len' (line 226)
    len_162063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'len', False)
    # Calling len(args, kwargs) (line 226)
    len_call_result_162066 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), len_162063, *[z2_162064], **kwargs_162065)
    
    # Assigning a type to the variable 'len2' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'len2', len_call_result_162066)
    
    
    # Getting the type of 'len2' (line 227)
    len2_162067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'len2')
    int_162068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 15), 'int')
    # Applying the binary operator '==' (line 227)
    result_eq_162069 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), '==', len2_162067, int_162068)
    
    # Testing the type of an if condition (line 227)
    if_condition_162070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), result_eq_162069)
    # Assigning a type to the variable 'if_condition_162070' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_162070', if_condition_162070)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'z1' (line 228)
    z1_162071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'z1')
    # Getting the type of 'z2' (line 228)
    z2_162072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 14), 'z2')
    # Applying the binary operator 'div=' (line 228)
    result_div_162073 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 8), 'div=', z1_162071, z2_162072)
    # Assigning a type to the variable 'z1' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'z1', result_div_162073)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 229)
    tuple_162074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 229)
    # Adding element type (line 229)
    # Getting the type of 'z1' (line 229)
    z1_162075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'z1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 15), tuple_162074, z1_162075)
    # Adding element type (line 229)
    
    # Obtaining the type of the subscript
    int_162076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'int')
    slice_162077 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 229, 19), None, int_162076, None)
    # Getting the type of 'z1' (line 229)
    z1_162078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'z1')
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___162079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 19), z1_162078, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_162080 = invoke(stypy.reporting.localization.Localization(__file__, 229, 19), getitem___162079, slice_162077)
    
    int_162081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'int')
    # Applying the binary operator '*' (line 229)
    result_mul_162082 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 19), '*', subscript_call_result_162080, int_162081)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 15), tuple_162074, result_mul_162082)
    
    # Assigning a type to the variable 'stypy_return_type' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', tuple_162074)
    # SSA branch for the else part of an if statement (line 227)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'len1' (line 230)
    len1_162083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 9), 'len1')
    # Getting the type of 'len2' (line 230)
    len2_162084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'len2')
    # Applying the binary operator '<' (line 230)
    result_lt_162085 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 9), '<', len1_162083, len2_162084)
    
    # Testing the type of an if condition (line 230)
    if_condition_162086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 9), result_lt_162085)
    # Assigning a type to the variable 'if_condition_162086' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 9), 'if_condition_162086', if_condition_162086)
    # SSA begins for if statement (line 230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 231)
    tuple_162087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 231)
    # Adding element type (line 231)
    
    # Obtaining the type of the subscript
    int_162088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 19), 'int')
    slice_162089 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 231, 15), None, int_162088, None)
    # Getting the type of 'z1' (line 231)
    z1_162090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'z1')
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___162091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 15), z1_162090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_162092 = invoke(stypy.reporting.localization.Localization(__file__, 231, 15), getitem___162091, slice_162089)
    
    int_162093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'int')
    # Applying the binary operator '*' (line 231)
    result_mul_162094 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 15), '*', subscript_call_result_162092, int_162093)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 15), tuple_162087, result_mul_162094)
    # Adding element type (line 231)
    # Getting the type of 'z1' (line 231)
    z1_162095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'z1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 15), tuple_162087, z1_162095)
    
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', tuple_162087)
    # SSA branch for the else part of an if statement (line 230)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 233):
    
    # Assigning a BinOp to a Name (line 233):
    # Getting the type of 'len1' (line 233)
    len1_162096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'len1')
    # Getting the type of 'len2' (line 233)
    len2_162097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'len2')
    # Applying the binary operator '-' (line 233)
    result_sub_162098 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '-', len1_162096, len2_162097)
    
    # Assigning a type to the variable 'dlen' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'dlen', result_sub_162098)
    
    # Assigning a Subscript to a Name (line 234):
    
    # Assigning a Subscript to a Name (line 234):
    
    # Obtaining the type of the subscript
    int_162099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'int')
    # Getting the type of 'z2' (line 234)
    z2_162100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 14), 'z2')
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___162101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 14), z2_162100, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_162102 = invoke(stypy.reporting.localization.Localization(__file__, 234, 14), getitem___162101, int_162099)
    
    # Assigning a type to the variable 'scl' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'scl', subscript_call_result_162102)
    
    # Getting the type of 'z2' (line 235)
    z2_162103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'z2')
    # Getting the type of 'scl' (line 235)
    scl_162104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 14), 'scl')
    # Applying the binary operator 'div=' (line 235)
    result_div_162105 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 8), 'div=', z2_162103, scl_162104)
    # Assigning a type to the variable 'z2' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'z2', result_div_162105)
    
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to empty(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'dlen' (line 236)
    dlen_162108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'dlen', False)
    int_162109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 30), 'int')
    # Applying the binary operator '+' (line 236)
    result_add_162110 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 23), '+', dlen_162108, int_162109)
    
    # Processing the call keyword arguments (line 236)
    # Getting the type of 'z1' (line 236)
    z1_162111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 39), 'z1', False)
    # Obtaining the member 'dtype' of a type (line 236)
    dtype_162112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 39), z1_162111, 'dtype')
    keyword_162113 = dtype_162112
    kwargs_162114 = {'dtype': keyword_162113}
    # Getting the type of 'np' (line 236)
    np_162106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 236)
    empty_162107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 14), np_162106, 'empty')
    # Calling empty(args, kwargs) (line 236)
    empty_call_result_162115 = invoke(stypy.reporting.localization.Localization(__file__, 236, 14), empty_162107, *[result_add_162110], **kwargs_162114)
    
    # Assigning a type to the variable 'quo' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'quo', empty_call_result_162115)
    
    # Assigning a Num to a Name (line 237):
    
    # Assigning a Num to a Name (line 237):
    int_162116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'int')
    # Assigning a type to the variable 'i' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'i', int_162116)
    
    # Assigning a Name to a Name (line 238):
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'dlen' (line 238)
    dlen_162117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'dlen')
    # Assigning a type to the variable 'j' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'j', dlen_162117)
    
    
    # Getting the type of 'i' (line 239)
    i_162118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'i')
    # Getting the type of 'j' (line 239)
    j_162119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 18), 'j')
    # Applying the binary operator '<' (line 239)
    result_lt_162120 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 14), '<', i_162118, j_162119)
    
    # Testing the type of an if condition (line 239)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 8), result_lt_162120)
    # SSA begins for while statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 240):
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 240)
    i_162121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'i')
    # Getting the type of 'z1' (line 240)
    z1_162122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'z1')
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___162123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 16), z1_162122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_162124 = invoke(stypy.reporting.localization.Localization(__file__, 240, 16), getitem___162123, i_162121)
    
    # Assigning a type to the variable 'r' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'r', subscript_call_result_162124)
    
    # Assigning a Subscript to a Subscript (line 241):
    
    # Assigning a Subscript to a Subscript (line 241):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 241)
    i_162125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'i')
    # Getting the type of 'z1' (line 241)
    z1_162126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'z1')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___162127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 21), z1_162126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_162128 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), getitem___162127, i_162125)
    
    # Getting the type of 'quo' (line 241)
    quo_162129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'quo')
    # Getting the type of 'i' (line 241)
    i_162130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'i')
    # Storing an element on a container (line 241)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 12), quo_162129, (i_162130, subscript_call_result_162128))
    
    # Assigning a Name to a Subscript (line 242):
    
    # Assigning a Name to a Subscript (line 242):
    # Getting the type of 'r' (line 242)
    r_162131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'r')
    # Getting the type of 'quo' (line 242)
    quo_162132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'quo')
    # Getting the type of 'dlen' (line 242)
    dlen_162133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'dlen')
    # Getting the type of 'i' (line 242)
    i_162134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'i')
    # Applying the binary operator '-' (line 242)
    result_sub_162135 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 16), '-', dlen_162133, i_162134)
    
    # Storing an element on a container (line 242)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), quo_162132, (result_sub_162135, r_162131))
    
    # Assigning a BinOp to a Name (line 243):
    
    # Assigning a BinOp to a Name (line 243):
    # Getting the type of 'r' (line 243)
    r_162136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'r')
    # Getting the type of 'z2' (line 243)
    z2_162137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'z2')
    # Applying the binary operator '*' (line 243)
    result_mul_162138 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 18), '*', r_162136, z2_162137)
    
    # Assigning a type to the variable 'tmp' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'tmp', result_mul_162138)
    
    # Getting the type of 'z1' (line 244)
    z1_162139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'z1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 244)
    i_162140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'i')
    # Getting the type of 'i' (line 244)
    i_162141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'i')
    # Getting the type of 'len2' (line 244)
    len2_162142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'len2')
    # Applying the binary operator '+' (line 244)
    result_add_162143 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 17), '+', i_162141, len2_162142)
    
    slice_162144 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 244, 12), i_162140, result_add_162143, None)
    # Getting the type of 'z1' (line 244)
    z1_162145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'z1')
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___162146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), z1_162145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_162147 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), getitem___162146, slice_162144)
    
    # Getting the type of 'tmp' (line 244)
    tmp_162148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'tmp')
    # Applying the binary operator '-=' (line 244)
    result_isub_162149 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '-=', subscript_call_result_162147, tmp_162148)
    # Getting the type of 'z1' (line 244)
    z1_162150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'z1')
    # Getting the type of 'i' (line 244)
    i_162151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'i')
    # Getting the type of 'i' (line 244)
    i_162152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'i')
    # Getting the type of 'len2' (line 244)
    len2_162153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'len2')
    # Applying the binary operator '+' (line 244)
    result_add_162154 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 17), '+', i_162152, len2_162153)
    
    slice_162155 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 244, 12), i_162151, result_add_162154, None)
    # Storing an element on a container (line 244)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 12), z1_162150, (slice_162155, result_isub_162149))
    
    
    # Getting the type of 'z1' (line 245)
    z1_162156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'z1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 245)
    j_162157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'j')
    # Getting the type of 'j' (line 245)
    j_162158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'j')
    # Getting the type of 'len2' (line 245)
    len2_162159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'len2')
    # Applying the binary operator '+' (line 245)
    result_add_162160 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 17), '+', j_162158, len2_162159)
    
    slice_162161 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 12), j_162157, result_add_162160, None)
    # Getting the type of 'z1' (line 245)
    z1_162162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'z1')
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___162163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), z1_162162, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_162164 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), getitem___162163, slice_162161)
    
    # Getting the type of 'tmp' (line 245)
    tmp_162165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'tmp')
    # Applying the binary operator '-=' (line 245)
    result_isub_162166 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 12), '-=', subscript_call_result_162164, tmp_162165)
    # Getting the type of 'z1' (line 245)
    z1_162167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'z1')
    # Getting the type of 'j' (line 245)
    j_162168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'j')
    # Getting the type of 'j' (line 245)
    j_162169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'j')
    # Getting the type of 'len2' (line 245)
    len2_162170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'len2')
    # Applying the binary operator '+' (line 245)
    result_add_162171 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 17), '+', j_162169, len2_162170)
    
    slice_162172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 12), j_162168, result_add_162171, None)
    # Storing an element on a container (line 245)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), z1_162167, (slice_162172, result_isub_162166))
    
    
    # Getting the type of 'i' (line 246)
    i_162173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'i')
    int_162174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'int')
    # Applying the binary operator '+=' (line 246)
    result_iadd_162175 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 12), '+=', i_162173, int_162174)
    # Assigning a type to the variable 'i' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'i', result_iadd_162175)
    
    
    # Getting the type of 'j' (line 247)
    j_162176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'j')
    int_162177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 17), 'int')
    # Applying the binary operator '-=' (line 247)
    result_isub_162178 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), '-=', j_162176, int_162177)
    # Assigning a type to the variable 'j' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'j', result_isub_162178)
    
    # SSA join for while statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 248):
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 248)
    i_162179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'i')
    # Getting the type of 'z1' (line 248)
    z1_162180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'z1')
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___162181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), z1_162180, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_162182 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), getitem___162181, i_162179)
    
    # Assigning a type to the variable 'r' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'r', subscript_call_result_162182)
    
    # Assigning a Name to a Subscript (line 249):
    
    # Assigning a Name to a Subscript (line 249):
    # Getting the type of 'r' (line 249)
    r_162183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'r')
    # Getting the type of 'quo' (line 249)
    quo_162184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'quo')
    # Getting the type of 'i' (line 249)
    i_162185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'i')
    # Storing an element on a container (line 249)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 8), quo_162184, (i_162185, r_162183))
    
    # Assigning a BinOp to a Name (line 250):
    
    # Assigning a BinOp to a Name (line 250):
    # Getting the type of 'r' (line 250)
    r_162186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 14), 'r')
    # Getting the type of 'z2' (line 250)
    z2_162187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'z2')
    # Applying the binary operator '*' (line 250)
    result_mul_162188 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 14), '*', r_162186, z2_162187)
    
    # Assigning a type to the variable 'tmp' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'tmp', result_mul_162188)
    
    # Getting the type of 'z1' (line 251)
    z1_162189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'z1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 251)
    i_162190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'i')
    # Getting the type of 'i' (line 251)
    i_162191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'i')
    # Getting the type of 'len2' (line 251)
    len2_162192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'len2')
    # Applying the binary operator '+' (line 251)
    result_add_162193 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 13), '+', i_162191, len2_162192)
    
    slice_162194 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 251, 8), i_162190, result_add_162193, None)
    # Getting the type of 'z1' (line 251)
    z1_162195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'z1')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___162196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), z1_162195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_162197 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), getitem___162196, slice_162194)
    
    # Getting the type of 'tmp' (line 251)
    tmp_162198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'tmp')
    # Applying the binary operator '-=' (line 251)
    result_isub_162199 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 8), '-=', subscript_call_result_162197, tmp_162198)
    # Getting the type of 'z1' (line 251)
    z1_162200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'z1')
    # Getting the type of 'i' (line 251)
    i_162201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'i')
    # Getting the type of 'i' (line 251)
    i_162202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'i')
    # Getting the type of 'len2' (line 251)
    len2_162203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'len2')
    # Applying the binary operator '+' (line 251)
    result_add_162204 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 13), '+', i_162202, len2_162203)
    
    slice_162205 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 251, 8), i_162201, result_add_162204, None)
    # Storing an element on a container (line 251)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 8), z1_162200, (slice_162205, result_isub_162199))
    
    
    # Getting the type of 'quo' (line 252)
    quo_162206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'quo')
    # Getting the type of 'scl' (line 252)
    scl_162207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'scl')
    # Applying the binary operator 'div=' (line 252)
    result_div_162208 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 8), 'div=', quo_162206, scl_162207)
    # Assigning a type to the variable 'quo' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'quo', result_div_162208)
    
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to copy(...): (line 253)
    # Processing the call keyword arguments (line 253)
    kwargs_162222 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 253)
    i_162209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'i', False)
    int_162210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 19), 'int')
    # Applying the binary operator '+' (line 253)
    result_add_162211 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 17), '+', i_162209, int_162210)
    
    # Getting the type of 'i' (line 253)
    i_162212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), 'i', False)
    int_162213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 23), 'int')
    # Applying the binary operator '-' (line 253)
    result_sub_162214 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 21), '-', i_162212, int_162213)
    
    # Getting the type of 'len2' (line 253)
    len2_162215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 25), 'len2', False)
    # Applying the binary operator '+' (line 253)
    result_add_162216 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 24), '+', result_sub_162214, len2_162215)
    
    slice_162217 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 14), result_add_162211, result_add_162216, None)
    # Getting the type of 'z1' (line 253)
    z1_162218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'z1', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___162219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 14), z1_162218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_162220 = invoke(stypy.reporting.localization.Localization(__file__, 253, 14), getitem___162219, slice_162217)
    
    # Obtaining the member 'copy' of a type (line 253)
    copy_162221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 14), subscript_call_result_162220, 'copy')
    # Calling copy(args, kwargs) (line 253)
    copy_call_result_162223 = invoke(stypy.reporting.localization.Localization(__file__, 253, 14), copy_162221, *[], **kwargs_162222)
    
    # Assigning a type to the variable 'rem' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'rem', copy_call_result_162223)
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_162224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    # Getting the type of 'quo' (line 254)
    quo_162225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'quo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), tuple_162224, quo_162225)
    # Adding element type (line 254)
    # Getting the type of 'rem' (line 254)
    rem_162226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'rem')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), tuple_162224, rem_162226)
    
    # Assigning a type to the variable 'stypy_return_type' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type', tuple_162224)
    # SSA join for if statement (line 230)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_zseries_div(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zseries_div' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_162227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162227)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zseries_div'
    return stypy_return_type_162227

# Assigning a type to the variable '_zseries_div' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), '_zseries_div', _zseries_div)

@norecursion
def _zseries_der(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zseries_der'
    module_type_store = module_type_store.open_function_context('_zseries_der', 257, 0, False)
    
    # Passed parameters checking function
    _zseries_der.stypy_localization = localization
    _zseries_der.stypy_type_of_self = None
    _zseries_der.stypy_type_store = module_type_store
    _zseries_der.stypy_function_name = '_zseries_der'
    _zseries_der.stypy_param_names_list = ['zs']
    _zseries_der.stypy_varargs_param_name = None
    _zseries_der.stypy_kwargs_param_name = None
    _zseries_der.stypy_call_defaults = defaults
    _zseries_der.stypy_call_varargs = varargs
    _zseries_der.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zseries_der', ['zs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zseries_der', localization, ['zs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zseries_der(...)' code ##################

    str_162228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'str', 'Differentiate a z-series.\n\n    The derivative is with respect to x, not z. This is achieved using the\n    chain rule and the value of dx/dz given in the module notes.\n\n    Parameters\n    ----------\n    zs : z-series\n        The z-series to differentiate.\n\n    Returns\n    -------\n    derivative : z-series\n        The derivative\n\n    Notes\n    -----\n    The zseries for x (ns) has been multiplied by two in order to avoid\n    using floats that are incompatible with Decimal and likely other\n    specialized scalar types. This scaling has been compensated by\n    multiplying the value of zs by two also so that the two cancels in the\n    division.\n\n    ')
    
    # Assigning a BinOp to a Name (line 282):
    
    # Assigning a BinOp to a Name (line 282):
    
    # Call to len(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'zs' (line 282)
    zs_162230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'zs', False)
    # Processing the call keyword arguments (line 282)
    kwargs_162231 = {}
    # Getting the type of 'len' (line 282)
    len_162229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'len', False)
    # Calling len(args, kwargs) (line 282)
    len_call_result_162232 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), len_162229, *[zs_162230], **kwargs_162231)
    
    int_162233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 17), 'int')
    # Applying the binary operator '//' (line 282)
    result_floordiv_162234 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 8), '//', len_call_result_162232, int_162233)
    
    # Assigning a type to the variable 'n' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'n', result_floordiv_162234)
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to array(...): (line 283)
    # Processing the call arguments (line 283)
    
    # Obtaining an instance of the builtin type 'list' (line 283)
    list_162237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 283)
    # Adding element type (line 283)
    int_162238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 18), list_162237, int_162238)
    # Adding element type (line 283)
    int_162239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 18), list_162237, int_162239)
    # Adding element type (line 283)
    int_162240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 18), list_162237, int_162240)
    
    # Processing the call keyword arguments (line 283)
    # Getting the type of 'zs' (line 283)
    zs_162241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 36), 'zs', False)
    # Obtaining the member 'dtype' of a type (line 283)
    dtype_162242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 36), zs_162241, 'dtype')
    keyword_162243 = dtype_162242
    kwargs_162244 = {'dtype': keyword_162243}
    # Getting the type of 'np' (line 283)
    np_162235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 283)
    array_162236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 9), np_162235, 'array')
    # Calling array(args, kwargs) (line 283)
    array_call_result_162245 = invoke(stypy.reporting.localization.Localization(__file__, 283, 9), array_162236, *[list_162237], **kwargs_162244)
    
    # Assigning a type to the variable 'ns' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'ns', array_call_result_162245)
    
    # Getting the type of 'zs' (line 284)
    zs_162246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'zs')
    
    # Call to arange(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Getting the type of 'n' (line 284)
    n_162249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'n', False)
    # Applying the 'usub' unary operator (line 284)
    result___neg___162250 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 20), 'usub', n_162249)
    
    # Getting the type of 'n' (line 284)
    n_162251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'n', False)
    int_162252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 26), 'int')
    # Applying the binary operator '+' (line 284)
    result_add_162253 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 24), '+', n_162251, int_162252)
    
    # Processing the call keyword arguments (line 284)
    kwargs_162254 = {}
    # Getting the type of 'np' (line 284)
    np_162247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 10), 'np', False)
    # Obtaining the member 'arange' of a type (line 284)
    arange_162248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 10), np_162247, 'arange')
    # Calling arange(args, kwargs) (line 284)
    arange_call_result_162255 = invoke(stypy.reporting.localization.Localization(__file__, 284, 10), arange_162248, *[result___neg___162250, result_add_162253], **kwargs_162254)
    
    int_162256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 29), 'int')
    # Applying the binary operator '*' (line 284)
    result_mul_162257 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 10), '*', arange_call_result_162255, int_162256)
    
    # Applying the binary operator '*=' (line 284)
    result_imul_162258 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 4), '*=', zs_162246, result_mul_162257)
    # Assigning a type to the variable 'zs' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'zs', result_imul_162258)
    
    
    # Assigning a Call to a Tuple (line 285):
    
    # Assigning a Call to a Name:
    
    # Call to _zseries_div(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'zs' (line 285)
    zs_162260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'zs', False)
    # Getting the type of 'ns' (line 285)
    ns_162261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 28), 'ns', False)
    # Processing the call keyword arguments (line 285)
    kwargs_162262 = {}
    # Getting the type of '_zseries_div' (line 285)
    _zseries_div_162259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), '_zseries_div', False)
    # Calling _zseries_div(args, kwargs) (line 285)
    _zseries_div_call_result_162263 = invoke(stypy.reporting.localization.Localization(__file__, 285, 11), _zseries_div_162259, *[zs_162260, ns_162261], **kwargs_162262)
    
    # Assigning a type to the variable 'call_assignment_161873' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161873', _zseries_div_call_result_162263)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162267 = {}
    # Getting the type of 'call_assignment_161873' (line 285)
    call_assignment_161873_162264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161873', False)
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___162265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 4), call_assignment_161873_162264, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162268 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162265, *[int_162266], **kwargs_162267)
    
    # Assigning a type to the variable 'call_assignment_161874' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161874', getitem___call_result_162268)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'call_assignment_161874' (line 285)
    call_assignment_161874_162269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161874')
    # Assigning a type to the variable 'd' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'd', call_assignment_161874_162269)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162273 = {}
    # Getting the type of 'call_assignment_161873' (line 285)
    call_assignment_161873_162270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161873', False)
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___162271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 4), call_assignment_161873_162270, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162274 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162271, *[int_162272], **kwargs_162273)
    
    # Assigning a type to the variable 'call_assignment_161875' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161875', getitem___call_result_162274)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'call_assignment_161875' (line 285)
    call_assignment_161875_162275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_161875')
    # Assigning a type to the variable 'r' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'r', call_assignment_161875_162275)
    # Getting the type of 'd' (line 286)
    d_162276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type', d_162276)
    
    # ################# End of '_zseries_der(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zseries_der' in the type store
    # Getting the type of 'stypy_return_type' (line 257)
    stypy_return_type_162277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162277)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zseries_der'
    return stypy_return_type_162277

# Assigning a type to the variable '_zseries_der' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), '_zseries_der', _zseries_der)

@norecursion
def _zseries_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zseries_int'
    module_type_store = module_type_store.open_function_context('_zseries_int', 289, 0, False)
    
    # Passed parameters checking function
    _zseries_int.stypy_localization = localization
    _zseries_int.stypy_type_of_self = None
    _zseries_int.stypy_type_store = module_type_store
    _zseries_int.stypy_function_name = '_zseries_int'
    _zseries_int.stypy_param_names_list = ['zs']
    _zseries_int.stypy_varargs_param_name = None
    _zseries_int.stypy_kwargs_param_name = None
    _zseries_int.stypy_call_defaults = defaults
    _zseries_int.stypy_call_varargs = varargs
    _zseries_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zseries_int', ['zs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zseries_int', localization, ['zs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zseries_int(...)' code ##################

    str_162278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, (-1)), 'str', 'Integrate a z-series.\n\n    The integral is with respect to x, not z. This is achieved by a change\n    of variable using dx/dz given in the module notes.\n\n    Parameters\n    ----------\n    zs : z-series\n        The z-series to integrate\n\n    Returns\n    -------\n    integral : z-series\n        The indefinite integral\n\n    Notes\n    -----\n    The zseries for x (ns) has been multiplied by two in order to avoid\n    using floats that are incompatible with Decimal and likely other\n    specialized scalar types. This scaling has been compensated by\n    dividing the resulting zs by two.\n\n    ')
    
    # Assigning a BinOp to a Name (line 313):
    
    # Assigning a BinOp to a Name (line 313):
    int_162279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 8), 'int')
    
    # Call to len(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'zs' (line 313)
    zs_162281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'zs', False)
    # Processing the call keyword arguments (line 313)
    kwargs_162282 = {}
    # Getting the type of 'len' (line 313)
    len_162280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'len', False)
    # Calling len(args, kwargs) (line 313)
    len_call_result_162283 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), len_162280, *[zs_162281], **kwargs_162282)
    
    int_162284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 21), 'int')
    # Applying the binary operator '//' (line 313)
    result_floordiv_162285 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 12), '//', len_call_result_162283, int_162284)
    
    # Applying the binary operator '+' (line 313)
    result_add_162286 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 8), '+', int_162279, result_floordiv_162285)
    
    # Assigning a type to the variable 'n' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'n', result_add_162286)
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to array(...): (line 314)
    # Processing the call arguments (line 314)
    
    # Obtaining an instance of the builtin type 'list' (line 314)
    list_162289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 314)
    # Adding element type (line 314)
    int_162290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 18), list_162289, int_162290)
    # Adding element type (line 314)
    int_162291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 18), list_162289, int_162291)
    # Adding element type (line 314)
    int_162292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 18), list_162289, int_162292)
    
    # Processing the call keyword arguments (line 314)
    # Getting the type of 'zs' (line 314)
    zs_162293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'zs', False)
    # Obtaining the member 'dtype' of a type (line 314)
    dtype_162294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 36), zs_162293, 'dtype')
    keyword_162295 = dtype_162294
    kwargs_162296 = {'dtype': keyword_162295}
    # Getting the type of 'np' (line 314)
    np_162287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 314)
    array_162288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 9), np_162287, 'array')
    # Calling array(args, kwargs) (line 314)
    array_call_result_162297 = invoke(stypy.reporting.localization.Localization(__file__, 314, 9), array_162288, *[list_162289], **kwargs_162296)
    
    # Assigning a type to the variable 'ns' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'ns', array_call_result_162297)
    
    # Assigning a Call to a Name (line 315):
    
    # Assigning a Call to a Name (line 315):
    
    # Call to _zseries_mul(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'zs' (line 315)
    zs_162299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'zs', False)
    # Getting the type of 'ns' (line 315)
    ns_162300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 26), 'ns', False)
    # Processing the call keyword arguments (line 315)
    kwargs_162301 = {}
    # Getting the type of '_zseries_mul' (line 315)
    _zseries_mul_162298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 9), '_zseries_mul', False)
    # Calling _zseries_mul(args, kwargs) (line 315)
    _zseries_mul_call_result_162302 = invoke(stypy.reporting.localization.Localization(__file__, 315, 9), _zseries_mul_162298, *[zs_162299, ns_162300], **kwargs_162301)
    
    # Assigning a type to the variable 'zs' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'zs', _zseries_mul_call_result_162302)
    
    # Assigning a BinOp to a Name (line 316):
    
    # Assigning a BinOp to a Name (line 316):
    
    # Call to arange(...): (line 316)
    # Processing the call arguments (line 316)
    
    # Getting the type of 'n' (line 316)
    n_162305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'n', False)
    # Applying the 'usub' unary operator (line 316)
    result___neg___162306 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 20), 'usub', n_162305)
    
    # Getting the type of 'n' (line 316)
    n_162307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 24), 'n', False)
    int_162308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 26), 'int')
    # Applying the binary operator '+' (line 316)
    result_add_162309 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 24), '+', n_162307, int_162308)
    
    # Processing the call keyword arguments (line 316)
    kwargs_162310 = {}
    # Getting the type of 'np' (line 316)
    np_162303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 10), 'np', False)
    # Obtaining the member 'arange' of a type (line 316)
    arange_162304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 10), np_162303, 'arange')
    # Calling arange(args, kwargs) (line 316)
    arange_call_result_162311 = invoke(stypy.reporting.localization.Localization(__file__, 316, 10), arange_162304, *[result___neg___162306, result_add_162309], **kwargs_162310)
    
    int_162312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 29), 'int')
    # Applying the binary operator '*' (line 316)
    result_mul_162313 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 10), '*', arange_call_result_162311, int_162312)
    
    # Assigning a type to the variable 'div' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'div', result_mul_162313)
    
    # Getting the type of 'zs' (line 317)
    zs_162314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'zs')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 317)
    n_162315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'n')
    slice_162316 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 317, 4), None, n_162315, None)
    # Getting the type of 'zs' (line 317)
    zs_162317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'zs')
    # Obtaining the member '__getitem__' of a type (line 317)
    getitem___162318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 4), zs_162317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 317)
    subscript_call_result_162319 = invoke(stypy.reporting.localization.Localization(__file__, 317, 4), getitem___162318, slice_162316)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 317)
    n_162320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'n')
    slice_162321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 317, 14), None, n_162320, None)
    # Getting the type of 'div' (line 317)
    div_162322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), 'div')
    # Obtaining the member '__getitem__' of a type (line 317)
    getitem___162323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 14), div_162322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 317)
    subscript_call_result_162324 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), getitem___162323, slice_162321)
    
    # Applying the binary operator 'div=' (line 317)
    result_div_162325 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 4), 'div=', subscript_call_result_162319, subscript_call_result_162324)
    # Getting the type of 'zs' (line 317)
    zs_162326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'zs')
    # Getting the type of 'n' (line 317)
    n_162327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'n')
    slice_162328 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 317, 4), None, n_162327, None)
    # Storing an element on a container (line 317)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 4), zs_162326, (slice_162328, result_div_162325))
    
    
    # Getting the type of 'zs' (line 318)
    zs_162329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'zs')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 318)
    n_162330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 7), 'n')
    int_162331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 9), 'int')
    # Applying the binary operator '+' (line 318)
    result_add_162332 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 7), '+', n_162330, int_162331)
    
    slice_162333 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 4), result_add_162332, None, None)
    # Getting the type of 'zs' (line 318)
    zs_162334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'zs')
    # Obtaining the member '__getitem__' of a type (line 318)
    getitem___162335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 4), zs_162334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 318)
    subscript_call_result_162336 = invoke(stypy.reporting.localization.Localization(__file__, 318, 4), getitem___162335, slice_162333)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 318)
    n_162337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'n')
    int_162338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 22), 'int')
    # Applying the binary operator '+' (line 318)
    result_add_162339 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 20), '+', n_162337, int_162338)
    
    slice_162340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 16), result_add_162339, None, None)
    # Getting the type of 'div' (line 318)
    div_162341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'div')
    # Obtaining the member '__getitem__' of a type (line 318)
    getitem___162342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), div_162341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 318)
    subscript_call_result_162343 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), getitem___162342, slice_162340)
    
    # Applying the binary operator 'div=' (line 318)
    result_div_162344 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 4), 'div=', subscript_call_result_162336, subscript_call_result_162343)
    # Getting the type of 'zs' (line 318)
    zs_162345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'zs')
    # Getting the type of 'n' (line 318)
    n_162346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 7), 'n')
    int_162347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 9), 'int')
    # Applying the binary operator '+' (line 318)
    result_add_162348 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 7), '+', n_162346, int_162347)
    
    slice_162349 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 4), result_add_162348, None, None)
    # Storing an element on a container (line 318)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 4), zs_162345, (slice_162349, result_div_162344))
    
    
    # Assigning a Num to a Subscript (line 319):
    
    # Assigning a Num to a Subscript (line 319):
    int_162350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 12), 'int')
    # Getting the type of 'zs' (line 319)
    zs_162351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'zs')
    # Getting the type of 'n' (line 319)
    n_162352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 7), 'n')
    # Storing an element on a container (line 319)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 4), zs_162351, (n_162352, int_162350))
    # Getting the type of 'zs' (line 320)
    zs_162353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'zs')
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type', zs_162353)
    
    # ################# End of '_zseries_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zseries_int' in the type store
    # Getting the type of 'stypy_return_type' (line 289)
    stypy_return_type_162354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162354)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zseries_int'
    return stypy_return_type_162354

# Assigning a type to the variable '_zseries_int' (line 289)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), '_zseries_int', _zseries_int)

@norecursion
def poly2cheb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poly2cheb'
    module_type_store = module_type_store.open_function_context('poly2cheb', 327, 0, False)
    
    # Passed parameters checking function
    poly2cheb.stypy_localization = localization
    poly2cheb.stypy_type_of_self = None
    poly2cheb.stypy_type_store = module_type_store
    poly2cheb.stypy_function_name = 'poly2cheb'
    poly2cheb.stypy_param_names_list = ['pol']
    poly2cheb.stypy_varargs_param_name = None
    poly2cheb.stypy_kwargs_param_name = None
    poly2cheb.stypy_call_defaults = defaults
    poly2cheb.stypy_call_varargs = varargs
    poly2cheb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poly2cheb', ['pol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poly2cheb', localization, ['pol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poly2cheb(...)' code ##################

    str_162355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, (-1)), 'str', '\n    Convert a polynomial to a Chebyshev series.\n\n    Convert an array representing the coefficients of a polynomial (relative\n    to the "standard" basis) ordered from lowest degree to highest, to an\n    array of the coefficients of the equivalent Chebyshev series, ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    pol : array_like\n        1-D array containing the polynomial coefficients\n\n    Returns\n    -------\n    c : ndarray\n        1-D array containing the coefficients of the equivalent Chebyshev\n        series.\n\n    See Also\n    --------\n    cheb2poly\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> p = P.Polynomial(range(4))\n    >>> p\n    Polynomial([ 0.,  1.,  2.,  3.], [-1.,  1.])\n    >>> c = p.convert(kind=P.Chebyshev)\n    >>> c\n    Chebyshev([ 1.  ,  3.25,  1.  ,  0.75], [-1.,  1.])\n    >>> P.poly2cheb(range(4))\n    array([ 1.  ,  3.25,  1.  ,  0.75])\n\n    ')
    
    # Assigning a Call to a List (line 369):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 369)
    # Processing the call arguments (line 369)
    
    # Obtaining an instance of the builtin type 'list' (line 369)
    list_162358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 369)
    # Adding element type (line 369)
    # Getting the type of 'pol' (line 369)
    pol_162359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'pol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 25), list_162358, pol_162359)
    
    # Processing the call keyword arguments (line 369)
    kwargs_162360 = {}
    # Getting the type of 'pu' (line 369)
    pu_162356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 369)
    as_series_162357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), pu_162356, 'as_series')
    # Calling as_series(args, kwargs) (line 369)
    as_series_call_result_162361 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), as_series_162357, *[list_162358], **kwargs_162360)
    
    # Assigning a type to the variable 'call_assignment_161876' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'call_assignment_161876', as_series_call_result_162361)
    
    # Assigning a Call to a Name (line 369):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162365 = {}
    # Getting the type of 'call_assignment_161876' (line 369)
    call_assignment_161876_162362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'call_assignment_161876', False)
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___162363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 4), call_assignment_161876_162362, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162366 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162363, *[int_162364], **kwargs_162365)
    
    # Assigning a type to the variable 'call_assignment_161877' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'call_assignment_161877', getitem___call_result_162366)
    
    # Assigning a Name to a Name (line 369):
    # Getting the type of 'call_assignment_161877' (line 369)
    call_assignment_161877_162367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'call_assignment_161877')
    # Assigning a type to the variable 'pol' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 5), 'pol', call_assignment_161877_162367)
    
    # Assigning a BinOp to a Name (line 370):
    
    # Assigning a BinOp to a Name (line 370):
    
    # Call to len(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'pol' (line 370)
    pol_162369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 14), 'pol', False)
    # Processing the call keyword arguments (line 370)
    kwargs_162370 = {}
    # Getting the type of 'len' (line 370)
    len_162368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 10), 'len', False)
    # Calling len(args, kwargs) (line 370)
    len_call_result_162371 = invoke(stypy.reporting.localization.Localization(__file__, 370, 10), len_162368, *[pol_162369], **kwargs_162370)
    
    int_162372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 21), 'int')
    # Applying the binary operator '-' (line 370)
    result_sub_162373 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 10), '-', len_call_result_162371, int_162372)
    
    # Assigning a type to the variable 'deg' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'deg', result_sub_162373)
    
    # Assigning a Num to a Name (line 371):
    
    # Assigning a Num to a Name (line 371):
    int_162374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 10), 'int')
    # Assigning a type to the variable 'res' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'res', int_162374)
    
    
    # Call to range(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'deg' (line 372)
    deg_162376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 19), 'deg', False)
    int_162377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 24), 'int')
    int_162378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 28), 'int')
    # Processing the call keyword arguments (line 372)
    kwargs_162379 = {}
    # Getting the type of 'range' (line 372)
    range_162375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), 'range', False)
    # Calling range(args, kwargs) (line 372)
    range_call_result_162380 = invoke(stypy.reporting.localization.Localization(__file__, 372, 13), range_162375, *[deg_162376, int_162377, int_162378], **kwargs_162379)
    
    # Testing the type of a for loop iterable (line 372)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 372, 4), range_call_result_162380)
    # Getting the type of the for loop variable (line 372)
    for_loop_var_162381 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 372, 4), range_call_result_162380)
    # Assigning a type to the variable 'i' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'i', for_loop_var_162381)
    # SSA begins for a for statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to chebadd(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to chebmulx(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'res' (line 373)
    res_162384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'res', False)
    # Processing the call keyword arguments (line 373)
    kwargs_162385 = {}
    # Getting the type of 'chebmulx' (line 373)
    chebmulx_162383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'chebmulx', False)
    # Calling chebmulx(args, kwargs) (line 373)
    chebmulx_call_result_162386 = invoke(stypy.reporting.localization.Localization(__file__, 373, 22), chebmulx_162383, *[res_162384], **kwargs_162385)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 373)
    i_162387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 41), 'i', False)
    # Getting the type of 'pol' (line 373)
    pol_162388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 37), 'pol', False)
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___162389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 37), pol_162388, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_162390 = invoke(stypy.reporting.localization.Localization(__file__, 373, 37), getitem___162389, i_162387)
    
    # Processing the call keyword arguments (line 373)
    kwargs_162391 = {}
    # Getting the type of 'chebadd' (line 373)
    chebadd_162382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 14), 'chebadd', False)
    # Calling chebadd(args, kwargs) (line 373)
    chebadd_call_result_162392 = invoke(stypy.reporting.localization.Localization(__file__, 373, 14), chebadd_162382, *[chebmulx_call_result_162386, subscript_call_result_162390], **kwargs_162391)
    
    # Assigning a type to the variable 'res' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'res', chebadd_call_result_162392)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 374)
    res_162393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type', res_162393)
    
    # ################# End of 'poly2cheb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poly2cheb' in the type store
    # Getting the type of 'stypy_return_type' (line 327)
    stypy_return_type_162394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162394)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poly2cheb'
    return stypy_return_type_162394

# Assigning a type to the variable 'poly2cheb' (line 327)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), 'poly2cheb', poly2cheb)

@norecursion
def cheb2poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cheb2poly'
    module_type_store = module_type_store.open_function_context('cheb2poly', 377, 0, False)
    
    # Passed parameters checking function
    cheb2poly.stypy_localization = localization
    cheb2poly.stypy_type_of_self = None
    cheb2poly.stypy_type_store = module_type_store
    cheb2poly.stypy_function_name = 'cheb2poly'
    cheb2poly.stypy_param_names_list = ['c']
    cheb2poly.stypy_varargs_param_name = None
    cheb2poly.stypy_kwargs_param_name = None
    cheb2poly.stypy_call_defaults = defaults
    cheb2poly.stypy_call_varargs = varargs
    cheb2poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cheb2poly', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cheb2poly', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cheb2poly(...)' code ##################

    str_162395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, (-1)), 'str', '\n    Convert a Chebyshev series to a polynomial.\n\n    Convert an array representing the coefficients of a Chebyshev series,\n    ordered from lowest degree to highest, to an array of the coefficients\n    of the equivalent polynomial (relative to the "standard" basis) ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array containing the Chebyshev series coefficients, ordered\n        from lowest order term to highest.\n\n    Returns\n    -------\n    pol : ndarray\n        1-D array containing the coefficients of the equivalent polynomial\n        (relative to the "standard" basis) ordered from lowest order term\n        to highest.\n\n    See Also\n    --------\n    poly2cheb\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy import polynomial as P\n    >>> c = P.Chebyshev(range(4))\n    >>> c\n    Chebyshev([ 0.,  1.,  2.,  3.], [-1.,  1.])\n    >>> p = c.convert(kind=P.Polynomial)\n    >>> p\n    Polynomial([ -2.,  -8.,   4.,  12.], [-1.,  1.])\n    >>> P.cheb2poly(range(4))\n    array([ -2.,  -8.,   4.,  12.])\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 421, 4))
    
    # 'from numpy.polynomial.polynomial import polyadd, polysub, polymulx' statement (line 421)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_162396 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 421, 4), 'numpy.polynomial.polynomial')

    if (type(import_162396) is not StypyTypeError):

        if (import_162396 != 'pyd_module'):
            __import__(import_162396)
            sys_modules_162397 = sys.modules[import_162396]
            import_from_module(stypy.reporting.localization.Localization(__file__, 421, 4), 'numpy.polynomial.polynomial', sys_modules_162397.module_type_store, module_type_store, ['polyadd', 'polysub', 'polymulx'])
            nest_module(stypy.reporting.localization.Localization(__file__, 421, 4), __file__, sys_modules_162397, sys_modules_162397.module_type_store, module_type_store)
        else:
            from numpy.polynomial.polynomial import polyadd, polysub, polymulx

            import_from_module(stypy.reporting.localization.Localization(__file__, 421, 4), 'numpy.polynomial.polynomial', None, module_type_store, ['polyadd', 'polysub', 'polymulx'], [polyadd, polysub, polymulx])

    else:
        # Assigning a type to the variable 'numpy.polynomial.polynomial' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'numpy.polynomial.polynomial', import_162396)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Assigning a Call to a List (line 423):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 423)
    # Processing the call arguments (line 423)
    
    # Obtaining an instance of the builtin type 'list' (line 423)
    list_162400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'c' (line 423)
    c_162401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 23), list_162400, c_162401)
    
    # Processing the call keyword arguments (line 423)
    kwargs_162402 = {}
    # Getting the type of 'pu' (line 423)
    pu_162398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 423)
    as_series_162399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 10), pu_162398, 'as_series')
    # Calling as_series(args, kwargs) (line 423)
    as_series_call_result_162403 = invoke(stypy.reporting.localization.Localization(__file__, 423, 10), as_series_162399, *[list_162400], **kwargs_162402)
    
    # Assigning a type to the variable 'call_assignment_161878' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'call_assignment_161878', as_series_call_result_162403)
    
    # Assigning a Call to a Name (line 423):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162407 = {}
    # Getting the type of 'call_assignment_161878' (line 423)
    call_assignment_161878_162404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'call_assignment_161878', False)
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___162405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 4), call_assignment_161878_162404, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162408 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162405, *[int_162406], **kwargs_162407)
    
    # Assigning a type to the variable 'call_assignment_161879' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'call_assignment_161879', getitem___call_result_162408)
    
    # Assigning a Name to a Name (line 423):
    # Getting the type of 'call_assignment_161879' (line 423)
    call_assignment_161879_162409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'call_assignment_161879')
    # Assigning a type to the variable 'c' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 5), 'c', call_assignment_161879_162409)
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to len(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'c' (line 424)
    c_162411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'c', False)
    # Processing the call keyword arguments (line 424)
    kwargs_162412 = {}
    # Getting the type of 'len' (line 424)
    len_162410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'len', False)
    # Calling len(args, kwargs) (line 424)
    len_call_result_162413 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), len_162410, *[c_162411], **kwargs_162412)
    
    # Assigning a type to the variable 'n' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'n', len_call_result_162413)
    
    
    # Getting the type of 'n' (line 425)
    n_162414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 7), 'n')
    int_162415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 11), 'int')
    # Applying the binary operator '<' (line 425)
    result_lt_162416 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), '<', n_162414, int_162415)
    
    # Testing the type of an if condition (line 425)
    if_condition_162417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 4), result_lt_162416)
    # Assigning a type to the variable 'if_condition_162417' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'if_condition_162417', if_condition_162417)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 426)
    c_162418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type', c_162418)
    # SSA branch for the else part of an if statement (line 425)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 428):
    
    # Assigning a Subscript to a Name (line 428):
    
    # Obtaining the type of the subscript
    int_162419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 15), 'int')
    # Getting the type of 'c' (line 428)
    c_162420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___162421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 13), c_162420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_162422 = invoke(stypy.reporting.localization.Localization(__file__, 428, 13), getitem___162421, int_162419)
    
    # Assigning a type to the variable 'c0' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'c0', subscript_call_result_162422)
    
    # Assigning a Subscript to a Name (line 429):
    
    # Assigning a Subscript to a Name (line 429):
    
    # Obtaining the type of the subscript
    int_162423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 15), 'int')
    # Getting the type of 'c' (line 429)
    c_162424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 429)
    getitem___162425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 13), c_162424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 429)
    subscript_call_result_162426 = invoke(stypy.reporting.localization.Localization(__file__, 429, 13), getitem___162425, int_162423)
    
    # Assigning a type to the variable 'c1' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'c1', subscript_call_result_162426)
    
    
    # Call to range(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'n' (line 431)
    n_162428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'n', False)
    int_162429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 27), 'int')
    # Applying the binary operator '-' (line 431)
    result_sub_162430 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 23), '-', n_162428, int_162429)
    
    int_162431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 30), 'int')
    int_162432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 33), 'int')
    # Processing the call keyword arguments (line 431)
    kwargs_162433 = {}
    # Getting the type of 'range' (line 431)
    range_162427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 17), 'range', False)
    # Calling range(args, kwargs) (line 431)
    range_call_result_162434 = invoke(stypy.reporting.localization.Localization(__file__, 431, 17), range_162427, *[result_sub_162430, int_162431, int_162432], **kwargs_162433)
    
    # Testing the type of a for loop iterable (line 431)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 431, 8), range_call_result_162434)
    # Getting the type of the for loop variable (line 431)
    for_loop_var_162435 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 431, 8), range_call_result_162434)
    # Assigning a type to the variable 'i' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'i', for_loop_var_162435)
    # SSA begins for a for statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 432):
    
    # Assigning a Name to a Name (line 432):
    # Getting the type of 'c0' (line 432)
    c0_162436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'tmp', c0_162436)
    
    # Assigning a Call to a Name (line 433):
    
    # Assigning a Call to a Name (line 433):
    
    # Call to polysub(...): (line 433)
    # Processing the call arguments (line 433)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 433)
    i_162438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'i', False)
    int_162439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 31), 'int')
    # Applying the binary operator '-' (line 433)
    result_sub_162440 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 27), '-', i_162438, int_162439)
    
    # Getting the type of 'c' (line 433)
    c_162441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 25), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 433)
    getitem___162442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 25), c_162441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 433)
    subscript_call_result_162443 = invoke(stypy.reporting.localization.Localization(__file__, 433, 25), getitem___162442, result_sub_162440)
    
    # Getting the type of 'c1' (line 433)
    c1_162444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 35), 'c1', False)
    # Processing the call keyword arguments (line 433)
    kwargs_162445 = {}
    # Getting the type of 'polysub' (line 433)
    polysub_162437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), 'polysub', False)
    # Calling polysub(args, kwargs) (line 433)
    polysub_call_result_162446 = invoke(stypy.reporting.localization.Localization(__file__, 433, 17), polysub_162437, *[subscript_call_result_162443, c1_162444], **kwargs_162445)
    
    # Assigning a type to the variable 'c0' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'c0', polysub_call_result_162446)
    
    # Assigning a Call to a Name (line 434):
    
    # Assigning a Call to a Name (line 434):
    
    # Call to polyadd(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'tmp' (line 434)
    tmp_162448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'tmp', False)
    
    # Call to polymulx(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'c1' (line 434)
    c1_162450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 39), 'c1', False)
    # Processing the call keyword arguments (line 434)
    kwargs_162451 = {}
    # Getting the type of 'polymulx' (line 434)
    polymulx_162449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 30), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 434)
    polymulx_call_result_162452 = invoke(stypy.reporting.localization.Localization(__file__, 434, 30), polymulx_162449, *[c1_162450], **kwargs_162451)
    
    int_162453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 43), 'int')
    # Applying the binary operator '*' (line 434)
    result_mul_162454 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 30), '*', polymulx_call_result_162452, int_162453)
    
    # Processing the call keyword arguments (line 434)
    kwargs_162455 = {}
    # Getting the type of 'polyadd' (line 434)
    polyadd_162447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 434)
    polyadd_call_result_162456 = invoke(stypy.reporting.localization.Localization(__file__, 434, 17), polyadd_162447, *[tmp_162448, result_mul_162454], **kwargs_162455)
    
    # Assigning a type to the variable 'c1' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'c1', polyadd_call_result_162456)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to polyadd(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'c0' (line 435)
    c0_162458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 23), 'c0', False)
    
    # Call to polymulx(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'c1' (line 435)
    c1_162460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'c1', False)
    # Processing the call keyword arguments (line 435)
    kwargs_162461 = {}
    # Getting the type of 'polymulx' (line 435)
    polymulx_162459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 435)
    polymulx_call_result_162462 = invoke(stypy.reporting.localization.Localization(__file__, 435, 27), polymulx_162459, *[c1_162460], **kwargs_162461)
    
    # Processing the call keyword arguments (line 435)
    kwargs_162463 = {}
    # Getting the type of 'polyadd' (line 435)
    polyadd_162457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 435)
    polyadd_call_result_162464 = invoke(stypy.reporting.localization.Localization(__file__, 435, 15), polyadd_162457, *[c0_162458, polymulx_call_result_162462], **kwargs_162463)
    
    # Assigning a type to the variable 'stypy_return_type' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'stypy_return_type', polyadd_call_result_162464)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cheb2poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cheb2poly' in the type store
    # Getting the type of 'stypy_return_type' (line 377)
    stypy_return_type_162465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162465)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cheb2poly'
    return stypy_return_type_162465

# Assigning a type to the variable 'cheb2poly' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'cheb2poly', cheb2poly)

# Assigning a Call to a Name (line 444):

# Assigning a Call to a Name (line 444):

# Call to array(...): (line 444)
# Processing the call arguments (line 444)

# Obtaining an instance of the builtin type 'list' (line 444)
list_162468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 444)
# Adding element type (line 444)
int_162469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 22), list_162468, int_162469)
# Adding element type (line 444)
int_162470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 22), list_162468, int_162470)

# Processing the call keyword arguments (line 444)
kwargs_162471 = {}
# Getting the type of 'np' (line 444)
np_162466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 13), 'np', False)
# Obtaining the member 'array' of a type (line 444)
array_162467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 13), np_162466, 'array')
# Calling array(args, kwargs) (line 444)
array_call_result_162472 = invoke(stypy.reporting.localization.Localization(__file__, 444, 13), array_162467, *[list_162468], **kwargs_162471)

# Assigning a type to the variable 'chebdomain' (line 444)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 0), 'chebdomain', array_call_result_162472)

# Assigning a Call to a Name (line 447):

# Assigning a Call to a Name (line 447):

# Call to array(...): (line 447)
# Processing the call arguments (line 447)

# Obtaining an instance of the builtin type 'list' (line 447)
list_162475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 447)
# Adding element type (line 447)
int_162476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 20), list_162475, int_162476)

# Processing the call keyword arguments (line 447)
kwargs_162477 = {}
# Getting the type of 'np' (line 447)
np_162473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'np', False)
# Obtaining the member 'array' of a type (line 447)
array_162474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 11), np_162473, 'array')
# Calling array(args, kwargs) (line 447)
array_call_result_162478 = invoke(stypy.reporting.localization.Localization(__file__, 447, 11), array_162474, *[list_162475], **kwargs_162477)

# Assigning a type to the variable 'chebzero' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'chebzero', array_call_result_162478)

# Assigning a Call to a Name (line 450):

# Assigning a Call to a Name (line 450):

# Call to array(...): (line 450)
# Processing the call arguments (line 450)

# Obtaining an instance of the builtin type 'list' (line 450)
list_162481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 450)
# Adding element type (line 450)
int_162482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 19), list_162481, int_162482)

# Processing the call keyword arguments (line 450)
kwargs_162483 = {}
# Getting the type of 'np' (line 450)
np_162479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 10), 'np', False)
# Obtaining the member 'array' of a type (line 450)
array_162480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 10), np_162479, 'array')
# Calling array(args, kwargs) (line 450)
array_call_result_162484 = invoke(stypy.reporting.localization.Localization(__file__, 450, 10), array_162480, *[list_162481], **kwargs_162483)

# Assigning a type to the variable 'chebone' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'chebone', array_call_result_162484)

# Assigning a Call to a Name (line 453):

# Assigning a Call to a Name (line 453):

# Call to array(...): (line 453)
# Processing the call arguments (line 453)

# Obtaining an instance of the builtin type 'list' (line 453)
list_162487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 453)
# Adding element type (line 453)
int_162488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 17), list_162487, int_162488)
# Adding element type (line 453)
int_162489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 17), list_162487, int_162489)

# Processing the call keyword arguments (line 453)
kwargs_162490 = {}
# Getting the type of 'np' (line 453)
np_162485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'np', False)
# Obtaining the member 'array' of a type (line 453)
array_162486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), np_162485, 'array')
# Calling array(args, kwargs) (line 453)
array_call_result_162491 = invoke(stypy.reporting.localization.Localization(__file__, 453, 8), array_162486, *[list_162487], **kwargs_162490)

# Assigning a type to the variable 'chebx' (line 453)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'chebx', array_call_result_162491)

@norecursion
def chebline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebline'
    module_type_store = module_type_store.open_function_context('chebline', 456, 0, False)
    
    # Passed parameters checking function
    chebline.stypy_localization = localization
    chebline.stypy_type_of_self = None
    chebline.stypy_type_store = module_type_store
    chebline.stypy_function_name = 'chebline'
    chebline.stypy_param_names_list = ['off', 'scl']
    chebline.stypy_varargs_param_name = None
    chebline.stypy_kwargs_param_name = None
    chebline.stypy_call_defaults = defaults
    chebline.stypy_call_varargs = varargs
    chebline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebline', ['off', 'scl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebline', localization, ['off', 'scl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebline(...)' code ##################

    str_162492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, (-1)), 'str', "\n    Chebyshev series whose graph is a straight line.\n\n\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The specified line is given by ``off + scl*x``.\n\n    Returns\n    -------\n    y : ndarray\n        This module's representation of the Chebyshev series for\n        ``off + scl*x``.\n\n    See Also\n    --------\n    polyline\n\n    Examples\n    --------\n    >>> import numpy.polynomial.chebyshev as C\n    >>> C.chebline(3,2)\n    array([3, 2])\n    >>> C.chebval(-3, C.chebline(3,2)) # should be -3\n    -3.0\n\n    ")
    
    
    # Getting the type of 'scl' (line 486)
    scl_162493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 7), 'scl')
    int_162494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 14), 'int')
    # Applying the binary operator '!=' (line 486)
    result_ne_162495 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 7), '!=', scl_162493, int_162494)
    
    # Testing the type of an if condition (line 486)
    if_condition_162496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 4), result_ne_162495)
    # Assigning a type to the variable 'if_condition_162496' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'if_condition_162496', if_condition_162496)
    # SSA begins for if statement (line 486)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 487)
    # Processing the call arguments (line 487)
    
    # Obtaining an instance of the builtin type 'list' (line 487)
    list_162499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 487)
    # Adding element type (line 487)
    # Getting the type of 'off' (line 487)
    off_162500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 24), list_162499, off_162500)
    # Adding element type (line 487)
    # Getting the type of 'scl' (line 487)
    scl_162501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 30), 'scl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 24), list_162499, scl_162501)
    
    # Processing the call keyword arguments (line 487)
    kwargs_162502 = {}
    # Getting the type of 'np' (line 487)
    np_162497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 487)
    array_162498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 15), np_162497, 'array')
    # Calling array(args, kwargs) (line 487)
    array_call_result_162503 = invoke(stypy.reporting.localization.Localization(__file__, 487, 15), array_162498, *[list_162499], **kwargs_162502)
    
    # Assigning a type to the variable 'stypy_return_type' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'stypy_return_type', array_call_result_162503)
    # SSA branch for the else part of an if statement (line 486)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 489)
    # Processing the call arguments (line 489)
    
    # Obtaining an instance of the builtin type 'list' (line 489)
    list_162506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 489)
    # Adding element type (line 489)
    # Getting the type of 'off' (line 489)
    off_162507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 24), list_162506, off_162507)
    
    # Processing the call keyword arguments (line 489)
    kwargs_162508 = {}
    # Getting the type of 'np' (line 489)
    np_162504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 489)
    array_162505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), np_162504, 'array')
    # Calling array(args, kwargs) (line 489)
    array_call_result_162509 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), array_162505, *[list_162506], **kwargs_162508)
    
    # Assigning a type to the variable 'stypy_return_type' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', array_call_result_162509)
    # SSA join for if statement (line 486)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'chebline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebline' in the type store
    # Getting the type of 'stypy_return_type' (line 456)
    stypy_return_type_162510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162510)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebline'
    return stypy_return_type_162510

# Assigning a type to the variable 'chebline' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'chebline', chebline)

@norecursion
def chebfromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebfromroots'
    module_type_store = module_type_store.open_function_context('chebfromroots', 492, 0, False)
    
    # Passed parameters checking function
    chebfromroots.stypy_localization = localization
    chebfromroots.stypy_type_of_self = None
    chebfromroots.stypy_type_store = module_type_store
    chebfromroots.stypy_function_name = 'chebfromroots'
    chebfromroots.stypy_param_names_list = ['roots']
    chebfromroots.stypy_varargs_param_name = None
    chebfromroots.stypy_kwargs_param_name = None
    chebfromroots.stypy_call_defaults = defaults
    chebfromroots.stypy_call_varargs = varargs
    chebfromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebfromroots', ['roots'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebfromroots', localization, ['roots'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebfromroots(...)' code ##################

    str_162511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, (-1)), 'str', '\n    Generate a Chebyshev series with given roots.\n\n    The function returns the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    in Chebyshev form, where the `r_n` are the roots specified in `roots`.\n    If a zero has multiplicity n, then it must appear in `roots` n times.\n    For instance, if 2 is a root of multiplicity three and 3 is a root of\n    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The\n    roots can appear in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * T_1(x) + ... +  c_n * T_n(x)\n\n    The coefficient of the last term is not generally 1 for monic\n    polynomials in Chebyshev form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of coefficients.  If all roots are real then `out` is a\n        real array, if some of the roots are complex, then `out` is complex\n        even if all the coefficients in the result are real (see Examples\n        below).\n\n    See Also\n    --------\n    polyfromroots, legfromroots, lagfromroots, hermfromroots,\n    hermefromroots.\n\n    Examples\n    --------\n    >>> import numpy.polynomial.chebyshev as C\n    >>> C.chebfromroots((-1,0,1)) # x^3 - x relative to the standard basis\n    array([ 0.  , -0.25,  0.  ,  0.25])\n    >>> j = complex(0,1)\n    >>> C.chebfromroots((-j,j)) # x^2 + 1 relative to the standard basis\n    array([ 1.5+0.j,  0.0+0.j,  0.5+0.j])\n\n    ')
    
    
    
    # Call to len(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'roots' (line 541)
    roots_162513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 'roots', False)
    # Processing the call keyword arguments (line 541)
    kwargs_162514 = {}
    # Getting the type of 'len' (line 541)
    len_162512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 7), 'len', False)
    # Calling len(args, kwargs) (line 541)
    len_call_result_162515 = invoke(stypy.reporting.localization.Localization(__file__, 541, 7), len_162512, *[roots_162513], **kwargs_162514)
    
    int_162516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 21), 'int')
    # Applying the binary operator '==' (line 541)
    result_eq_162517 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 7), '==', len_call_result_162515, int_162516)
    
    # Testing the type of an if condition (line 541)
    if_condition_162518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 4), result_eq_162517)
    # Assigning a type to the variable 'if_condition_162518' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'if_condition_162518', if_condition_162518)
    # SSA begins for if statement (line 541)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 542)
    # Processing the call arguments (line 542)
    int_162521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 23), 'int')
    # Processing the call keyword arguments (line 542)
    kwargs_162522 = {}
    # Getting the type of 'np' (line 542)
    np_162519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 542)
    ones_162520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 15), np_162519, 'ones')
    # Calling ones(args, kwargs) (line 542)
    ones_call_result_162523 = invoke(stypy.reporting.localization.Localization(__file__, 542, 15), ones_162520, *[int_162521], **kwargs_162522)
    
    # Assigning a type to the variable 'stypy_return_type' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'stypy_return_type', ones_call_result_162523)
    # SSA branch for the else part of an if statement (line 541)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a List (line 544):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 544)
    # Processing the call arguments (line 544)
    
    # Obtaining an instance of the builtin type 'list' (line 544)
    list_162526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 544)
    # Adding element type (line 544)
    # Getting the type of 'roots' (line 544)
    roots_162527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 32), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 31), list_162526, roots_162527)
    
    # Processing the call keyword arguments (line 544)
    # Getting the type of 'False' (line 544)
    False_162528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 45), 'False', False)
    keyword_162529 = False_162528
    kwargs_162530 = {'trim': keyword_162529}
    # Getting the type of 'pu' (line 544)
    pu_162524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 18), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 544)
    as_series_162525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 18), pu_162524, 'as_series')
    # Calling as_series(args, kwargs) (line 544)
    as_series_call_result_162531 = invoke(stypy.reporting.localization.Localization(__file__, 544, 18), as_series_162525, *[list_162526], **kwargs_162530)
    
    # Assigning a type to the variable 'call_assignment_161880' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'call_assignment_161880', as_series_call_result_162531)
    
    # Assigning a Call to a Name (line 544):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 8), 'int')
    # Processing the call keyword arguments
    kwargs_162535 = {}
    # Getting the type of 'call_assignment_161880' (line 544)
    call_assignment_161880_162532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'call_assignment_161880', False)
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___162533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 8), call_assignment_161880_162532, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162536 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162533, *[int_162534], **kwargs_162535)
    
    # Assigning a type to the variable 'call_assignment_161881' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'call_assignment_161881', getitem___call_result_162536)
    
    # Assigning a Name to a Name (line 544):
    # Getting the type of 'call_assignment_161881' (line 544)
    call_assignment_161881_162537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'call_assignment_161881')
    # Assigning a type to the variable 'roots' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 9), 'roots', call_assignment_161881_162537)
    
    # Call to sort(...): (line 545)
    # Processing the call keyword arguments (line 545)
    kwargs_162540 = {}
    # Getting the type of 'roots' (line 545)
    roots_162538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'roots', False)
    # Obtaining the member 'sort' of a type (line 545)
    sort_162539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 8), roots_162538, 'sort')
    # Calling sort(args, kwargs) (line 545)
    sort_call_result_162541 = invoke(stypy.reporting.localization.Localization(__file__, 545, 8), sort_162539, *[], **kwargs_162540)
    
    
    # Assigning a ListComp to a Name (line 546):
    
    # Assigning a ListComp to a Name (line 546):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'roots' (line 546)
    roots_162548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 38), 'roots')
    comprehension_162549 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 13), roots_162548)
    # Assigning a type to the variable 'r' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), 'r', comprehension_162549)
    
    # Call to chebline(...): (line 546)
    # Processing the call arguments (line 546)
    
    # Getting the type of 'r' (line 546)
    r_162543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'r', False)
    # Applying the 'usub' unary operator (line 546)
    result___neg___162544 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 22), 'usub', r_162543)
    
    int_162545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 26), 'int')
    # Processing the call keyword arguments (line 546)
    kwargs_162546 = {}
    # Getting the type of 'chebline' (line 546)
    chebline_162542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), 'chebline', False)
    # Calling chebline(args, kwargs) (line 546)
    chebline_call_result_162547 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), chebline_162542, *[result___neg___162544, int_162545], **kwargs_162546)
    
    list_162550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 13), list_162550, chebline_call_result_162547)
    # Assigning a type to the variable 'p' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'p', list_162550)
    
    # Assigning a Call to a Name (line 547):
    
    # Assigning a Call to a Name (line 547):
    
    # Call to len(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'p' (line 547)
    p_162552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'p', False)
    # Processing the call keyword arguments (line 547)
    kwargs_162553 = {}
    # Getting the type of 'len' (line 547)
    len_162551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'len', False)
    # Calling len(args, kwargs) (line 547)
    len_call_result_162554 = invoke(stypy.reporting.localization.Localization(__file__, 547, 12), len_162551, *[p_162552], **kwargs_162553)
    
    # Assigning a type to the variable 'n' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'n', len_call_result_162554)
    
    
    # Getting the type of 'n' (line 548)
    n_162555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 14), 'n')
    int_162556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 18), 'int')
    # Applying the binary operator '>' (line 548)
    result_gt_162557 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 14), '>', n_162555, int_162556)
    
    # Testing the type of an if condition (line 548)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 8), result_gt_162557)
    # SSA begins for while statement (line 548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 549):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'n' (line 549)
    n_162559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 26), 'n', False)
    int_162560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 29), 'int')
    # Processing the call keyword arguments (line 549)
    kwargs_162561 = {}
    # Getting the type of 'divmod' (line 549)
    divmod_162558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 549)
    divmod_call_result_162562 = invoke(stypy.reporting.localization.Localization(__file__, 549, 19), divmod_162558, *[n_162559, int_162560], **kwargs_162561)
    
    # Assigning a type to the variable 'call_assignment_161882' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161882', divmod_call_result_162562)
    
    # Assigning a Call to a Name (line 549):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 12), 'int')
    # Processing the call keyword arguments
    kwargs_162566 = {}
    # Getting the type of 'call_assignment_161882' (line 549)
    call_assignment_161882_162563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161882', False)
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___162564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), call_assignment_161882_162563, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162567 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162564, *[int_162565], **kwargs_162566)
    
    # Assigning a type to the variable 'call_assignment_161883' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161883', getitem___call_result_162567)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'call_assignment_161883' (line 549)
    call_assignment_161883_162568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161883')
    # Assigning a type to the variable 'm' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'm', call_assignment_161883_162568)
    
    # Assigning a Call to a Name (line 549):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 12), 'int')
    # Processing the call keyword arguments
    kwargs_162572 = {}
    # Getting the type of 'call_assignment_161882' (line 549)
    call_assignment_161882_162569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161882', False)
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___162570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), call_assignment_161882_162569, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162573 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162570, *[int_162571], **kwargs_162572)
    
    # Assigning a type to the variable 'call_assignment_161884' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161884', getitem___call_result_162573)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'call_assignment_161884' (line 549)
    call_assignment_161884_162574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'call_assignment_161884')
    # Assigning a type to the variable 'r' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 15), 'r', call_assignment_161884_162574)
    
    # Assigning a ListComp to a Name (line 550):
    
    # Assigning a ListComp to a Name (line 550):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'm' (line 550)
    m_162589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 56), 'm', False)
    # Processing the call keyword arguments (line 550)
    kwargs_162590 = {}
    # Getting the type of 'range' (line 550)
    range_162588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 50), 'range', False)
    # Calling range(args, kwargs) (line 550)
    range_call_result_162591 = invoke(stypy.reporting.localization.Localization(__file__, 550, 50), range_162588, *[m_162589], **kwargs_162590)
    
    comprehension_162592 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 19), range_call_result_162591)
    # Assigning a type to the variable 'i' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'i', comprehension_162592)
    
    # Call to chebmul(...): (line 550)
    # Processing the call arguments (line 550)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 550)
    i_162576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 29), 'i', False)
    # Getting the type of 'p' (line 550)
    p_162577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 27), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___162578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 27), p_162577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 550)
    subscript_call_result_162579 = invoke(stypy.reporting.localization.Localization(__file__, 550, 27), getitem___162578, i_162576)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 550)
    i_162580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'i', False)
    # Getting the type of 'm' (line 550)
    m_162581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 37), 'm', False)
    # Applying the binary operator '+' (line 550)
    result_add_162582 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 35), '+', i_162580, m_162581)
    
    # Getting the type of 'p' (line 550)
    p_162583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 33), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___162584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 33), p_162583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 550)
    subscript_call_result_162585 = invoke(stypy.reporting.localization.Localization(__file__, 550, 33), getitem___162584, result_add_162582)
    
    # Processing the call keyword arguments (line 550)
    kwargs_162586 = {}
    # Getting the type of 'chebmul' (line 550)
    chebmul_162575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'chebmul', False)
    # Calling chebmul(args, kwargs) (line 550)
    chebmul_call_result_162587 = invoke(stypy.reporting.localization.Localization(__file__, 550, 19), chebmul_162575, *[subscript_call_result_162579, subscript_call_result_162585], **kwargs_162586)
    
    list_162593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 19), list_162593, chebmul_call_result_162587)
    # Assigning a type to the variable 'tmp' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'tmp', list_162593)
    
    # Getting the type of 'r' (line 551)
    r_162594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'r')
    # Testing the type of an if condition (line 551)
    if_condition_162595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 12), r_162594)
    # Assigning a type to the variable 'if_condition_162595' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'if_condition_162595', if_condition_162595)
    # SSA begins for if statement (line 551)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 552):
    
    # Assigning a Call to a Subscript (line 552):
    
    # Call to chebmul(...): (line 552)
    # Processing the call arguments (line 552)
    
    # Obtaining the type of the subscript
    int_162597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 37), 'int')
    # Getting the type of 'tmp' (line 552)
    tmp_162598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 33), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 552)
    getitem___162599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 33), tmp_162598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 552)
    subscript_call_result_162600 = invoke(stypy.reporting.localization.Localization(__file__, 552, 33), getitem___162599, int_162597)
    
    
    # Obtaining the type of the subscript
    int_162601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 43), 'int')
    # Getting the type of 'p' (line 552)
    p_162602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 41), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 552)
    getitem___162603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 41), p_162602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 552)
    subscript_call_result_162604 = invoke(stypy.reporting.localization.Localization(__file__, 552, 41), getitem___162603, int_162601)
    
    # Processing the call keyword arguments (line 552)
    kwargs_162605 = {}
    # Getting the type of 'chebmul' (line 552)
    chebmul_162596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 25), 'chebmul', False)
    # Calling chebmul(args, kwargs) (line 552)
    chebmul_call_result_162606 = invoke(stypy.reporting.localization.Localization(__file__, 552, 25), chebmul_162596, *[subscript_call_result_162600, subscript_call_result_162604], **kwargs_162605)
    
    # Getting the type of 'tmp' (line 552)
    tmp_162607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'tmp')
    int_162608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 20), 'int')
    # Storing an element on a container (line 552)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 16), tmp_162607, (int_162608, chebmul_call_result_162606))
    # SSA join for if statement (line 551)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 553):
    
    # Assigning a Name to a Name (line 553):
    # Getting the type of 'tmp' (line 553)
    tmp_162609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'tmp')
    # Assigning a type to the variable 'p' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'p', tmp_162609)
    
    # Assigning a Name to a Name (line 554):
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'm' (line 554)
    m_162610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'm')
    # Assigning a type to the variable 'n' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'n', m_162610)
    # SSA join for while statement (line 548)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_162611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 17), 'int')
    # Getting the type of 'p' (line 555)
    p_162612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'p')
    # Obtaining the member '__getitem__' of a type (line 555)
    getitem___162613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 15), p_162612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 555)
    subscript_call_result_162614 = invoke(stypy.reporting.localization.Localization(__file__, 555, 15), getitem___162613, int_162611)
    
    # Assigning a type to the variable 'stypy_return_type' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'stypy_return_type', subscript_call_result_162614)
    # SSA join for if statement (line 541)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'chebfromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebfromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 492)
    stypy_return_type_162615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebfromroots'
    return stypy_return_type_162615

# Assigning a type to the variable 'chebfromroots' (line 492)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'chebfromroots', chebfromroots)

@norecursion
def chebadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebadd'
    module_type_store = module_type_store.open_function_context('chebadd', 558, 0, False)
    
    # Passed parameters checking function
    chebadd.stypy_localization = localization
    chebadd.stypy_type_of_self = None
    chebadd.stypy_type_store = module_type_store
    chebadd.stypy_function_name = 'chebadd'
    chebadd.stypy_param_names_list = ['c1', 'c2']
    chebadd.stypy_varargs_param_name = None
    chebadd.stypy_kwargs_param_name = None
    chebadd.stypy_call_defaults = defaults
    chebadd.stypy_call_varargs = varargs
    chebadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebadd', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebadd', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebadd(...)' code ##################

    str_162616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, (-1)), 'str', '\n    Add one Chebyshev series to another.\n\n    Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments\n    are sequences of coefficients ordered from lowest order term to\n    highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Chebyshev series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the Chebyshev series of their sum.\n\n    See Also\n    --------\n    chebsub, chebmul, chebdiv, chebpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the sum of two Chebyshev series\n    is a Chebyshev series (without having to "reproject" the result onto\n    the basis set) so addition, just like that of "standard" polynomials,\n    is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial import chebyshev as C\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> C.chebadd(c1,c2)\n    array([ 4.,  4.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 598):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 598)
    # Processing the call arguments (line 598)
    
    # Obtaining an instance of the builtin type 'list' (line 598)
    list_162619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 598)
    # Adding element type (line 598)
    # Getting the type of 'c1' (line 598)
    c1_162620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 28), list_162619, c1_162620)
    # Adding element type (line 598)
    # Getting the type of 'c2' (line 598)
    c2_162621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 28), list_162619, c2_162621)
    
    # Processing the call keyword arguments (line 598)
    kwargs_162622 = {}
    # Getting the type of 'pu' (line 598)
    pu_162617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 598)
    as_series_162618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 15), pu_162617, 'as_series')
    # Calling as_series(args, kwargs) (line 598)
    as_series_call_result_162623 = invoke(stypy.reporting.localization.Localization(__file__, 598, 15), as_series_162618, *[list_162619], **kwargs_162622)
    
    # Assigning a type to the variable 'call_assignment_161885' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161885', as_series_call_result_162623)
    
    # Assigning a Call to a Name (line 598):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162627 = {}
    # Getting the type of 'call_assignment_161885' (line 598)
    call_assignment_161885_162624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161885', False)
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___162625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 4), call_assignment_161885_162624, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162628 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162625, *[int_162626], **kwargs_162627)
    
    # Assigning a type to the variable 'call_assignment_161886' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161886', getitem___call_result_162628)
    
    # Assigning a Name to a Name (line 598):
    # Getting the type of 'call_assignment_161886' (line 598)
    call_assignment_161886_162629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161886')
    # Assigning a type to the variable 'c1' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 5), 'c1', call_assignment_161886_162629)
    
    # Assigning a Call to a Name (line 598):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162633 = {}
    # Getting the type of 'call_assignment_161885' (line 598)
    call_assignment_161885_162630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161885', False)
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___162631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 4), call_assignment_161885_162630, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162634 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162631, *[int_162632], **kwargs_162633)
    
    # Assigning a type to the variable 'call_assignment_161887' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161887', getitem___call_result_162634)
    
    # Assigning a Name to a Name (line 598):
    # Getting the type of 'call_assignment_161887' (line 598)
    call_assignment_161887_162635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_161887')
    # Assigning a type to the variable 'c2' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 9), 'c2', call_assignment_161887_162635)
    
    
    
    # Call to len(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'c1' (line 599)
    c1_162637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'c1', False)
    # Processing the call keyword arguments (line 599)
    kwargs_162638 = {}
    # Getting the type of 'len' (line 599)
    len_162636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 7), 'len', False)
    # Calling len(args, kwargs) (line 599)
    len_call_result_162639 = invoke(stypy.reporting.localization.Localization(__file__, 599, 7), len_162636, *[c1_162637], **kwargs_162638)
    
    
    # Call to len(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'c2' (line 599)
    c2_162641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 21), 'c2', False)
    # Processing the call keyword arguments (line 599)
    kwargs_162642 = {}
    # Getting the type of 'len' (line 599)
    len_162640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'len', False)
    # Calling len(args, kwargs) (line 599)
    len_call_result_162643 = invoke(stypy.reporting.localization.Localization(__file__, 599, 17), len_162640, *[c2_162641], **kwargs_162642)
    
    # Applying the binary operator '>' (line 599)
    result_gt_162644 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 7), '>', len_call_result_162639, len_call_result_162643)
    
    # Testing the type of an if condition (line 599)
    if_condition_162645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 4), result_gt_162644)
    # Assigning a type to the variable 'if_condition_162645' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'if_condition_162645', if_condition_162645)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 600)
    c1_162646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 600)
    c2_162647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'c2')
    # Obtaining the member 'size' of a type (line 600)
    size_162648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), c2_162647, 'size')
    slice_162649 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 600, 8), None, size_162648, None)
    # Getting the type of 'c1' (line 600)
    c1_162650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___162651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 8), c1_162650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 600)
    subscript_call_result_162652 = invoke(stypy.reporting.localization.Localization(__file__, 600, 8), getitem___162651, slice_162649)
    
    # Getting the type of 'c2' (line 600)
    c2_162653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'c2')
    # Applying the binary operator '+=' (line 600)
    result_iadd_162654 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 8), '+=', subscript_call_result_162652, c2_162653)
    # Getting the type of 'c1' (line 600)
    c1_162655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'c1')
    # Getting the type of 'c2' (line 600)
    c2_162656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'c2')
    # Obtaining the member 'size' of a type (line 600)
    size_162657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), c2_162656, 'size')
    slice_162658 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 600, 8), None, size_162657, None)
    # Storing an element on a container (line 600)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 8), c1_162655, (slice_162658, result_iadd_162654))
    
    
    # Assigning a Name to a Name (line 601):
    
    # Assigning a Name to a Name (line 601):
    # Getting the type of 'c1' (line 601)
    c1_162659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'ret', c1_162659)
    # SSA branch for the else part of an if statement (line 599)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'c2' (line 603)
    c2_162660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 603)
    c1_162661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'c1')
    # Obtaining the member 'size' of a type (line 603)
    size_162662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 12), c1_162661, 'size')
    slice_162663 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 603, 8), None, size_162662, None)
    # Getting the type of 'c2' (line 603)
    c2_162664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___162665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 8), c2_162664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_162666 = invoke(stypy.reporting.localization.Localization(__file__, 603, 8), getitem___162665, slice_162663)
    
    # Getting the type of 'c1' (line 603)
    c1_162667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 24), 'c1')
    # Applying the binary operator '+=' (line 603)
    result_iadd_162668 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 8), '+=', subscript_call_result_162666, c1_162667)
    # Getting the type of 'c2' (line 603)
    c2_162669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'c2')
    # Getting the type of 'c1' (line 603)
    c1_162670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'c1')
    # Obtaining the member 'size' of a type (line 603)
    size_162671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 12), c1_162670, 'size')
    slice_162672 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 603, 8), None, size_162671, None)
    # Storing an element on a container (line 603)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 8), c2_162669, (slice_162672, result_iadd_162668))
    
    
    # Assigning a Name to a Name (line 604):
    
    # Assigning a Name to a Name (line 604):
    # Getting the type of 'c2' (line 604)
    c2_162673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'ret', c2_162673)
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 605)
    # Processing the call arguments (line 605)
    # Getting the type of 'ret' (line 605)
    ret_162676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 22), 'ret', False)
    # Processing the call keyword arguments (line 605)
    kwargs_162677 = {}
    # Getting the type of 'pu' (line 605)
    pu_162674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 605)
    trimseq_162675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 11), pu_162674, 'trimseq')
    # Calling trimseq(args, kwargs) (line 605)
    trimseq_call_result_162678 = invoke(stypy.reporting.localization.Localization(__file__, 605, 11), trimseq_162675, *[ret_162676], **kwargs_162677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'stypy_return_type', trimseq_call_result_162678)
    
    # ################# End of 'chebadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebadd' in the type store
    # Getting the type of 'stypy_return_type' (line 558)
    stypy_return_type_162679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162679)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebadd'
    return stypy_return_type_162679

# Assigning a type to the variable 'chebadd' (line 558)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), 'chebadd', chebadd)

@norecursion
def chebsub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebsub'
    module_type_store = module_type_store.open_function_context('chebsub', 608, 0, False)
    
    # Passed parameters checking function
    chebsub.stypy_localization = localization
    chebsub.stypy_type_of_self = None
    chebsub.stypy_type_store = module_type_store
    chebsub.stypy_function_name = 'chebsub'
    chebsub.stypy_param_names_list = ['c1', 'c2']
    chebsub.stypy_varargs_param_name = None
    chebsub.stypy_kwargs_param_name = None
    chebsub.stypy_call_defaults = defaults
    chebsub.stypy_call_varargs = varargs
    chebsub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebsub', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebsub', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebsub(...)' code ##################

    str_162680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, (-1)), 'str', '\n    Subtract one Chebyshev series from another.\n\n    Returns the difference of two Chebyshev series `c1` - `c2`.  The\n    sequences of coefficients are from lowest order term to highest, i.e.,\n    [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Chebyshev series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Chebyshev series coefficients representing their difference.\n\n    See Also\n    --------\n    chebadd, chebmul, chebdiv, chebpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the difference of two Chebyshev\n    series is a Chebyshev series (without having to "reproject" the result\n    onto the basis set) so subtraction, just like that of "standard"\n    polynomials, is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial import chebyshev as C\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> C.chebsub(c1,c2)\n    array([-2.,  0.,  2.])\n    >>> C.chebsub(c2,c1) # -C.chebsub(c1,c2)\n    array([ 2.,  0., -2.])\n\n    ')
    
    # Assigning a Call to a List (line 650):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 650)
    # Processing the call arguments (line 650)
    
    # Obtaining an instance of the builtin type 'list' (line 650)
    list_162683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 650)
    # Adding element type (line 650)
    # Getting the type of 'c1' (line 650)
    c1_162684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 28), list_162683, c1_162684)
    # Adding element type (line 650)
    # Getting the type of 'c2' (line 650)
    c2_162685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 28), list_162683, c2_162685)
    
    # Processing the call keyword arguments (line 650)
    kwargs_162686 = {}
    # Getting the type of 'pu' (line 650)
    pu_162681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 650)
    as_series_162682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 15), pu_162681, 'as_series')
    # Calling as_series(args, kwargs) (line 650)
    as_series_call_result_162687 = invoke(stypy.reporting.localization.Localization(__file__, 650, 15), as_series_162682, *[list_162683], **kwargs_162686)
    
    # Assigning a type to the variable 'call_assignment_161888' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161888', as_series_call_result_162687)
    
    # Assigning a Call to a Name (line 650):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162691 = {}
    # Getting the type of 'call_assignment_161888' (line 650)
    call_assignment_161888_162688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161888', False)
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___162689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 4), call_assignment_161888_162688, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162692 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162689, *[int_162690], **kwargs_162691)
    
    # Assigning a type to the variable 'call_assignment_161889' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161889', getitem___call_result_162692)
    
    # Assigning a Name to a Name (line 650):
    # Getting the type of 'call_assignment_161889' (line 650)
    call_assignment_161889_162693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161889')
    # Assigning a type to the variable 'c1' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 5), 'c1', call_assignment_161889_162693)
    
    # Assigning a Call to a Name (line 650):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162697 = {}
    # Getting the type of 'call_assignment_161888' (line 650)
    call_assignment_161888_162694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161888', False)
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___162695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 4), call_assignment_161888_162694, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162698 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162695, *[int_162696], **kwargs_162697)
    
    # Assigning a type to the variable 'call_assignment_161890' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161890', getitem___call_result_162698)
    
    # Assigning a Name to a Name (line 650):
    # Getting the type of 'call_assignment_161890' (line 650)
    call_assignment_161890_162699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'call_assignment_161890')
    # Assigning a type to the variable 'c2' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 9), 'c2', call_assignment_161890_162699)
    
    
    
    # Call to len(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'c1' (line 651)
    c1_162701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 11), 'c1', False)
    # Processing the call keyword arguments (line 651)
    kwargs_162702 = {}
    # Getting the type of 'len' (line 651)
    len_162700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 7), 'len', False)
    # Calling len(args, kwargs) (line 651)
    len_call_result_162703 = invoke(stypy.reporting.localization.Localization(__file__, 651, 7), len_162700, *[c1_162701], **kwargs_162702)
    
    
    # Call to len(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'c2' (line 651)
    c2_162705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 21), 'c2', False)
    # Processing the call keyword arguments (line 651)
    kwargs_162706 = {}
    # Getting the type of 'len' (line 651)
    len_162704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 17), 'len', False)
    # Calling len(args, kwargs) (line 651)
    len_call_result_162707 = invoke(stypy.reporting.localization.Localization(__file__, 651, 17), len_162704, *[c2_162705], **kwargs_162706)
    
    # Applying the binary operator '>' (line 651)
    result_gt_162708 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 7), '>', len_call_result_162703, len_call_result_162707)
    
    # Testing the type of an if condition (line 651)
    if_condition_162709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 4), result_gt_162708)
    # Assigning a type to the variable 'if_condition_162709' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'if_condition_162709', if_condition_162709)
    # SSA begins for if statement (line 651)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 652)
    c1_162710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 652)
    c2_162711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'c2')
    # Obtaining the member 'size' of a type (line 652)
    size_162712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 12), c2_162711, 'size')
    slice_162713 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 652, 8), None, size_162712, None)
    # Getting the type of 'c1' (line 652)
    c1_162714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___162715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 8), c1_162714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_162716 = invoke(stypy.reporting.localization.Localization(__file__, 652, 8), getitem___162715, slice_162713)
    
    # Getting the type of 'c2' (line 652)
    c2_162717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 24), 'c2')
    # Applying the binary operator '-=' (line 652)
    result_isub_162718 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 8), '-=', subscript_call_result_162716, c2_162717)
    # Getting the type of 'c1' (line 652)
    c1_162719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'c1')
    # Getting the type of 'c2' (line 652)
    c2_162720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'c2')
    # Obtaining the member 'size' of a type (line 652)
    size_162721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 12), c2_162720, 'size')
    slice_162722 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 652, 8), None, size_162721, None)
    # Storing an element on a container (line 652)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 8), c1_162719, (slice_162722, result_isub_162718))
    
    
    # Assigning a Name to a Name (line 653):
    
    # Assigning a Name to a Name (line 653):
    # Getting the type of 'c1' (line 653)
    c1_162723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'ret', c1_162723)
    # SSA branch for the else part of an if statement (line 651)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a UnaryOp to a Name (line 655):
    
    # Assigning a UnaryOp to a Name (line 655):
    
    # Getting the type of 'c2' (line 655)
    c2_162724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'c2')
    # Applying the 'usub' unary operator (line 655)
    result___neg___162725 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 13), 'usub', c2_162724)
    
    # Assigning a type to the variable 'c2' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'c2', result___neg___162725)
    
    # Getting the type of 'c2' (line 656)
    c2_162726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 656)
    c1_162727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'c1')
    # Obtaining the member 'size' of a type (line 656)
    size_162728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 12), c1_162727, 'size')
    slice_162729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 656, 8), None, size_162728, None)
    # Getting the type of 'c2' (line 656)
    c2_162730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 656)
    getitem___162731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 8), c2_162730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 656)
    subscript_call_result_162732 = invoke(stypy.reporting.localization.Localization(__file__, 656, 8), getitem___162731, slice_162729)
    
    # Getting the type of 'c1' (line 656)
    c1_162733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 24), 'c1')
    # Applying the binary operator '+=' (line 656)
    result_iadd_162734 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 8), '+=', subscript_call_result_162732, c1_162733)
    # Getting the type of 'c2' (line 656)
    c2_162735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'c2')
    # Getting the type of 'c1' (line 656)
    c1_162736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'c1')
    # Obtaining the member 'size' of a type (line 656)
    size_162737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 12), c1_162736, 'size')
    slice_162738 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 656, 8), None, size_162737, None)
    # Storing an element on a container (line 656)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 8), c2_162735, (slice_162738, result_iadd_162734))
    
    
    # Assigning a Name to a Name (line 657):
    
    # Assigning a Name to a Name (line 657):
    # Getting the type of 'c2' (line 657)
    c2_162739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'ret', c2_162739)
    # SSA join for if statement (line 651)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'ret' (line 658)
    ret_162742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 22), 'ret', False)
    # Processing the call keyword arguments (line 658)
    kwargs_162743 = {}
    # Getting the type of 'pu' (line 658)
    pu_162740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 658)
    trimseq_162741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 11), pu_162740, 'trimseq')
    # Calling trimseq(args, kwargs) (line 658)
    trimseq_call_result_162744 = invoke(stypy.reporting.localization.Localization(__file__, 658, 11), trimseq_162741, *[ret_162742], **kwargs_162743)
    
    # Assigning a type to the variable 'stypy_return_type' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'stypy_return_type', trimseq_call_result_162744)
    
    # ################# End of 'chebsub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebsub' in the type store
    # Getting the type of 'stypy_return_type' (line 608)
    stypy_return_type_162745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162745)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebsub'
    return stypy_return_type_162745

# Assigning a type to the variable 'chebsub' (line 608)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 0), 'chebsub', chebsub)

@norecursion
def chebmulx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebmulx'
    module_type_store = module_type_store.open_function_context('chebmulx', 661, 0, False)
    
    # Passed parameters checking function
    chebmulx.stypy_localization = localization
    chebmulx.stypy_type_of_self = None
    chebmulx.stypy_type_store = module_type_store
    chebmulx.stypy_function_name = 'chebmulx'
    chebmulx.stypy_param_names_list = ['c']
    chebmulx.stypy_varargs_param_name = None
    chebmulx.stypy_kwargs_param_name = None
    chebmulx.stypy_call_defaults = defaults
    chebmulx.stypy_call_varargs = varargs
    chebmulx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebmulx', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebmulx', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebmulx(...)' code ##################

    str_162746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, (-1)), 'str', 'Multiply a Chebyshev series by x.\n\n    Multiply the polynomial `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Chebyshev series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.5.0\n\n    ')
    
    # Assigning a Call to a List (line 686):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 686)
    # Processing the call arguments (line 686)
    
    # Obtaining an instance of the builtin type 'list' (line 686)
    list_162749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 686)
    # Adding element type (line 686)
    # Getting the type of 'c' (line 686)
    c_162750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 23), list_162749, c_162750)
    
    # Processing the call keyword arguments (line 686)
    kwargs_162751 = {}
    # Getting the type of 'pu' (line 686)
    pu_162747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 686)
    as_series_162748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 10), pu_162747, 'as_series')
    # Calling as_series(args, kwargs) (line 686)
    as_series_call_result_162752 = invoke(stypy.reporting.localization.Localization(__file__, 686, 10), as_series_162748, *[list_162749], **kwargs_162751)
    
    # Assigning a type to the variable 'call_assignment_161891' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'call_assignment_161891', as_series_call_result_162752)
    
    # Assigning a Call to a Name (line 686):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162756 = {}
    # Getting the type of 'call_assignment_161891' (line 686)
    call_assignment_161891_162753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'call_assignment_161891', False)
    # Obtaining the member '__getitem__' of a type (line 686)
    getitem___162754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 4), call_assignment_161891_162753, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162757 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162754, *[int_162755], **kwargs_162756)
    
    # Assigning a type to the variable 'call_assignment_161892' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'call_assignment_161892', getitem___call_result_162757)
    
    # Assigning a Name to a Name (line 686):
    # Getting the type of 'call_assignment_161892' (line 686)
    call_assignment_161892_162758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'call_assignment_161892')
    # Assigning a type to the variable 'c' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 5), 'c', call_assignment_161892_162758)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 688)
    # Processing the call arguments (line 688)
    # Getting the type of 'c' (line 688)
    c_162760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 11), 'c', False)
    # Processing the call keyword arguments (line 688)
    kwargs_162761 = {}
    # Getting the type of 'len' (line 688)
    len_162759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 7), 'len', False)
    # Calling len(args, kwargs) (line 688)
    len_call_result_162762 = invoke(stypy.reporting.localization.Localization(__file__, 688, 7), len_162759, *[c_162760], **kwargs_162761)
    
    int_162763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 17), 'int')
    # Applying the binary operator '==' (line 688)
    result_eq_162764 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 7), '==', len_call_result_162762, int_162763)
    
    
    
    # Obtaining the type of the subscript
    int_162765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 25), 'int')
    # Getting the type of 'c' (line 688)
    c_162766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 23), 'c')
    # Obtaining the member '__getitem__' of a type (line 688)
    getitem___162767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 23), c_162766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 688)
    subscript_call_result_162768 = invoke(stypy.reporting.localization.Localization(__file__, 688, 23), getitem___162767, int_162765)
    
    int_162769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 31), 'int')
    # Applying the binary operator '==' (line 688)
    result_eq_162770 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 23), '==', subscript_call_result_162768, int_162769)
    
    # Applying the binary operator 'and' (line 688)
    result_and_keyword_162771 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 7), 'and', result_eq_162764, result_eq_162770)
    
    # Testing the type of an if condition (line 688)
    if_condition_162772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 4), result_and_keyword_162771)
    # Assigning a type to the variable 'if_condition_162772' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'if_condition_162772', if_condition_162772)
    # SSA begins for if statement (line 688)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 689)
    c_162773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'stypy_return_type', c_162773)
    # SSA join for if statement (line 688)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 691):
    
    # Assigning a Call to a Name (line 691):
    
    # Call to empty(...): (line 691)
    # Processing the call arguments (line 691)
    
    # Call to len(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'c' (line 691)
    c_162777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 23), 'c', False)
    # Processing the call keyword arguments (line 691)
    kwargs_162778 = {}
    # Getting the type of 'len' (line 691)
    len_162776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 19), 'len', False)
    # Calling len(args, kwargs) (line 691)
    len_call_result_162779 = invoke(stypy.reporting.localization.Localization(__file__, 691, 19), len_162776, *[c_162777], **kwargs_162778)
    
    int_162780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 28), 'int')
    # Applying the binary operator '+' (line 691)
    result_add_162781 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 19), '+', len_call_result_162779, int_162780)
    
    # Processing the call keyword arguments (line 691)
    # Getting the type of 'c' (line 691)
    c_162782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 37), 'c', False)
    # Obtaining the member 'dtype' of a type (line 691)
    dtype_162783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 37), c_162782, 'dtype')
    keyword_162784 = dtype_162783
    kwargs_162785 = {'dtype': keyword_162784}
    # Getting the type of 'np' (line 691)
    np_162774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 691)
    empty_162775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 10), np_162774, 'empty')
    # Calling empty(args, kwargs) (line 691)
    empty_call_result_162786 = invoke(stypy.reporting.localization.Localization(__file__, 691, 10), empty_162775, *[result_add_162781], **kwargs_162785)
    
    # Assigning a type to the variable 'prd' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'prd', empty_call_result_162786)
    
    # Assigning a BinOp to a Subscript (line 692):
    
    # Assigning a BinOp to a Subscript (line 692):
    
    # Obtaining the type of the subscript
    int_162787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 15), 'int')
    # Getting the type of 'c' (line 692)
    c_162788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 692)
    getitem___162789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 13), c_162788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 692)
    subscript_call_result_162790 = invoke(stypy.reporting.localization.Localization(__file__, 692, 13), getitem___162789, int_162787)
    
    int_162791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 18), 'int')
    # Applying the binary operator '*' (line 692)
    result_mul_162792 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 13), '*', subscript_call_result_162790, int_162791)
    
    # Getting the type of 'prd' (line 692)
    prd_162793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'prd')
    int_162794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 8), 'int')
    # Storing an element on a container (line 692)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 4), prd_162793, (int_162794, result_mul_162792))
    
    # Assigning a Subscript to a Subscript (line 693):
    
    # Assigning a Subscript to a Subscript (line 693):
    
    # Obtaining the type of the subscript
    int_162795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 15), 'int')
    # Getting the type of 'c' (line 693)
    c_162796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___162797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 13), c_162796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_162798 = invoke(stypy.reporting.localization.Localization(__file__, 693, 13), getitem___162797, int_162795)
    
    # Getting the type of 'prd' (line 693)
    prd_162799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'prd')
    int_162800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 8), 'int')
    # Storing an element on a container (line 693)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 693, 4), prd_162799, (int_162800, subscript_call_result_162798))
    
    
    
    # Call to len(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'c' (line 694)
    c_162802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 11), 'c', False)
    # Processing the call keyword arguments (line 694)
    kwargs_162803 = {}
    # Getting the type of 'len' (line 694)
    len_162801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 7), 'len', False)
    # Calling len(args, kwargs) (line 694)
    len_call_result_162804 = invoke(stypy.reporting.localization.Localization(__file__, 694, 7), len_162801, *[c_162802], **kwargs_162803)
    
    int_162805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 16), 'int')
    # Applying the binary operator '>' (line 694)
    result_gt_162806 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 7), '>', len_call_result_162804, int_162805)
    
    # Testing the type of an if condition (line 694)
    if_condition_162807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 4), result_gt_162806)
    # Assigning a type to the variable 'if_condition_162807' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'if_condition_162807', if_condition_162807)
    # SSA begins for if statement (line 694)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 695):
    
    # Assigning a BinOp to a Name (line 695):
    
    # Obtaining the type of the subscript
    int_162808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 16), 'int')
    slice_162809 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 695, 14), int_162808, None, None)
    # Getting the type of 'c' (line 695)
    c_162810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 14), 'c')
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___162811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 14), c_162810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_162812 = invoke(stypy.reporting.localization.Localization(__file__, 695, 14), getitem___162811, slice_162809)
    
    int_162813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 20), 'int')
    # Applying the binary operator 'div' (line 695)
    result_div_162814 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 14), 'div', subscript_call_result_162812, int_162813)
    
    # Assigning a type to the variable 'tmp' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'tmp', result_div_162814)
    
    # Assigning a Name to a Subscript (line 696):
    
    # Assigning a Name to a Subscript (line 696):
    # Getting the type of 'tmp' (line 696)
    tmp_162815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 18), 'tmp')
    # Getting the type of 'prd' (line 696)
    prd_162816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'prd')
    int_162817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 12), 'int')
    slice_162818 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 696, 8), int_162817, None, None)
    # Storing an element on a container (line 696)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 8), prd_162816, (slice_162818, tmp_162815))
    
    # Getting the type of 'prd' (line 697)
    prd_162819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'prd')
    
    # Obtaining the type of the subscript
    int_162820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 12), 'int')
    int_162821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 14), 'int')
    slice_162822 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 697, 8), int_162820, int_162821, None)
    # Getting the type of 'prd' (line 697)
    prd_162823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'prd')
    # Obtaining the member '__getitem__' of a type (line 697)
    getitem___162824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), prd_162823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 697)
    subscript_call_result_162825 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), getitem___162824, slice_162822)
    
    # Getting the type of 'tmp' (line 697)
    tmp_162826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 21), 'tmp')
    # Applying the binary operator '+=' (line 697)
    result_iadd_162827 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 8), '+=', subscript_call_result_162825, tmp_162826)
    # Getting the type of 'prd' (line 697)
    prd_162828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'prd')
    int_162829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 12), 'int')
    int_162830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 14), 'int')
    slice_162831 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 697, 8), int_162829, int_162830, None)
    # Storing an element on a container (line 697)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 8), prd_162828, (slice_162831, result_iadd_162827))
    
    # SSA join for if statement (line 694)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 698)
    prd_162832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 11), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'stypy_return_type', prd_162832)
    
    # ################# End of 'chebmulx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebmulx' in the type store
    # Getting the type of 'stypy_return_type' (line 661)
    stypy_return_type_162833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162833)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebmulx'
    return stypy_return_type_162833

# Assigning a type to the variable 'chebmulx' (line 661)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 0), 'chebmulx', chebmulx)

@norecursion
def chebmul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebmul'
    module_type_store = module_type_store.open_function_context('chebmul', 701, 0, False)
    
    # Passed parameters checking function
    chebmul.stypy_localization = localization
    chebmul.stypy_type_of_self = None
    chebmul.stypy_type_store = module_type_store
    chebmul.stypy_function_name = 'chebmul'
    chebmul.stypy_param_names_list = ['c1', 'c2']
    chebmul.stypy_varargs_param_name = None
    chebmul.stypy_kwargs_param_name = None
    chebmul.stypy_call_defaults = defaults
    chebmul.stypy_call_varargs = varargs
    chebmul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebmul', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebmul', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebmul(...)' code ##################

    str_162834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, (-1)), 'str', '\n    Multiply one Chebyshev series by another.\n\n    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments\n    are sequences of coefficients, from lowest order "term" to highest,\n    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Chebyshev series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Chebyshev series coefficients representing their product.\n\n    See Also\n    --------\n    chebadd, chebsub, chebdiv, chebpow\n\n    Notes\n    -----\n    In general, the (polynomial) product of two C-series results in terms\n    that are not in the Chebyshev polynomial basis set.  Thus, to express\n    the product as a C-series, it is typically necessary to "reproject"\n    the product onto said basis set, which typically produces\n    "unintuitive live" (but correct) results; see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import chebyshev as C\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"\n    array([  6.5,  12. ,  12. ,   4. ,   1.5])\n\n    ')
    
    # Assigning a Call to a List (line 742):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 742)
    # Processing the call arguments (line 742)
    
    # Obtaining an instance of the builtin type 'list' (line 742)
    list_162837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 742)
    # Adding element type (line 742)
    # Getting the type of 'c1' (line 742)
    c1_162838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 28), list_162837, c1_162838)
    # Adding element type (line 742)
    # Getting the type of 'c2' (line 742)
    c2_162839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 28), list_162837, c2_162839)
    
    # Processing the call keyword arguments (line 742)
    kwargs_162840 = {}
    # Getting the type of 'pu' (line 742)
    pu_162835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 742)
    as_series_162836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 15), pu_162835, 'as_series')
    # Calling as_series(args, kwargs) (line 742)
    as_series_call_result_162841 = invoke(stypy.reporting.localization.Localization(__file__, 742, 15), as_series_162836, *[list_162837], **kwargs_162840)
    
    # Assigning a type to the variable 'call_assignment_161893' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161893', as_series_call_result_162841)
    
    # Assigning a Call to a Name (line 742):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162845 = {}
    # Getting the type of 'call_assignment_161893' (line 742)
    call_assignment_161893_162842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161893', False)
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___162843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 4), call_assignment_161893_162842, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162846 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162843, *[int_162844], **kwargs_162845)
    
    # Assigning a type to the variable 'call_assignment_161894' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161894', getitem___call_result_162846)
    
    # Assigning a Name to a Name (line 742):
    # Getting the type of 'call_assignment_161894' (line 742)
    call_assignment_161894_162847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161894')
    # Assigning a type to the variable 'c1' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 5), 'c1', call_assignment_161894_162847)
    
    # Assigning a Call to a Name (line 742):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162851 = {}
    # Getting the type of 'call_assignment_161893' (line 742)
    call_assignment_161893_162848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161893', False)
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___162849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 4), call_assignment_161893_162848, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162852 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162849, *[int_162850], **kwargs_162851)
    
    # Assigning a type to the variable 'call_assignment_161895' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161895', getitem___call_result_162852)
    
    # Assigning a Name to a Name (line 742):
    # Getting the type of 'call_assignment_161895' (line 742)
    call_assignment_161895_162853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'call_assignment_161895')
    # Assigning a type to the variable 'c2' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 9), 'c2', call_assignment_161895_162853)
    
    # Assigning a Call to a Name (line 743):
    
    # Assigning a Call to a Name (line 743):
    
    # Call to _cseries_to_zseries(...): (line 743)
    # Processing the call arguments (line 743)
    # Getting the type of 'c1' (line 743)
    c1_162855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 29), 'c1', False)
    # Processing the call keyword arguments (line 743)
    kwargs_162856 = {}
    # Getting the type of '_cseries_to_zseries' (line 743)
    _cseries_to_zseries_162854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 9), '_cseries_to_zseries', False)
    # Calling _cseries_to_zseries(args, kwargs) (line 743)
    _cseries_to_zseries_call_result_162857 = invoke(stypy.reporting.localization.Localization(__file__, 743, 9), _cseries_to_zseries_162854, *[c1_162855], **kwargs_162856)
    
    # Assigning a type to the variable 'z1' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 4), 'z1', _cseries_to_zseries_call_result_162857)
    
    # Assigning a Call to a Name (line 744):
    
    # Assigning a Call to a Name (line 744):
    
    # Call to _cseries_to_zseries(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 'c2' (line 744)
    c2_162859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 29), 'c2', False)
    # Processing the call keyword arguments (line 744)
    kwargs_162860 = {}
    # Getting the type of '_cseries_to_zseries' (line 744)
    _cseries_to_zseries_162858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 9), '_cseries_to_zseries', False)
    # Calling _cseries_to_zseries(args, kwargs) (line 744)
    _cseries_to_zseries_call_result_162861 = invoke(stypy.reporting.localization.Localization(__file__, 744, 9), _cseries_to_zseries_162858, *[c2_162859], **kwargs_162860)
    
    # Assigning a type to the variable 'z2' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'z2', _cseries_to_zseries_call_result_162861)
    
    # Assigning a Call to a Name (line 745):
    
    # Assigning a Call to a Name (line 745):
    
    # Call to _zseries_mul(...): (line 745)
    # Processing the call arguments (line 745)
    # Getting the type of 'z1' (line 745)
    z1_162863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 23), 'z1', False)
    # Getting the type of 'z2' (line 745)
    z2_162864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 27), 'z2', False)
    # Processing the call keyword arguments (line 745)
    kwargs_162865 = {}
    # Getting the type of '_zseries_mul' (line 745)
    _zseries_mul_162862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 10), '_zseries_mul', False)
    # Calling _zseries_mul(args, kwargs) (line 745)
    _zseries_mul_call_result_162866 = invoke(stypy.reporting.localization.Localization(__file__, 745, 10), _zseries_mul_162862, *[z1_162863, z2_162864], **kwargs_162865)
    
    # Assigning a type to the variable 'prd' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'prd', _zseries_mul_call_result_162866)
    
    # Assigning a Call to a Name (line 746):
    
    # Assigning a Call to a Name (line 746):
    
    # Call to _zseries_to_cseries(...): (line 746)
    # Processing the call arguments (line 746)
    # Getting the type of 'prd' (line 746)
    prd_162868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 30), 'prd', False)
    # Processing the call keyword arguments (line 746)
    kwargs_162869 = {}
    # Getting the type of '_zseries_to_cseries' (line 746)
    _zseries_to_cseries_162867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 10), '_zseries_to_cseries', False)
    # Calling _zseries_to_cseries(args, kwargs) (line 746)
    _zseries_to_cseries_call_result_162870 = invoke(stypy.reporting.localization.Localization(__file__, 746, 10), _zseries_to_cseries_162867, *[prd_162868], **kwargs_162869)
    
    # Assigning a type to the variable 'ret' (line 746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 4), 'ret', _zseries_to_cseries_call_result_162870)
    
    # Call to trimseq(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'ret' (line 747)
    ret_162873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 22), 'ret', False)
    # Processing the call keyword arguments (line 747)
    kwargs_162874 = {}
    # Getting the type of 'pu' (line 747)
    pu_162871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 747)
    trimseq_162872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 11), pu_162871, 'trimseq')
    # Calling trimseq(args, kwargs) (line 747)
    trimseq_call_result_162875 = invoke(stypy.reporting.localization.Localization(__file__, 747, 11), trimseq_162872, *[ret_162873], **kwargs_162874)
    
    # Assigning a type to the variable 'stypy_return_type' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'stypy_return_type', trimseq_call_result_162875)
    
    # ################# End of 'chebmul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebmul' in the type store
    # Getting the type of 'stypy_return_type' (line 701)
    stypy_return_type_162876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162876)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebmul'
    return stypy_return_type_162876

# Assigning a type to the variable 'chebmul' (line 701)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 0), 'chebmul', chebmul)

@norecursion
def chebdiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebdiv'
    module_type_store = module_type_store.open_function_context('chebdiv', 750, 0, False)
    
    # Passed parameters checking function
    chebdiv.stypy_localization = localization
    chebdiv.stypy_type_of_self = None
    chebdiv.stypy_type_store = module_type_store
    chebdiv.stypy_function_name = 'chebdiv'
    chebdiv.stypy_param_names_list = ['c1', 'c2']
    chebdiv.stypy_varargs_param_name = None
    chebdiv.stypy_kwargs_param_name = None
    chebdiv.stypy_call_defaults = defaults
    chebdiv.stypy_call_varargs = varargs
    chebdiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebdiv', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebdiv', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebdiv(...)' code ##################

    str_162877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, (-1)), 'str', '\n    Divide one Chebyshev series by another.\n\n    Returns the quotient-with-remainder of two Chebyshev series\n    `c1` / `c2`.  The arguments are sequences of coefficients from lowest\n    order "term" to highest, e.g., [1,2,3] represents the series\n    ``T_0 + 2*T_1 + 3*T_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Chebyshev series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    [quo, rem] : ndarrays\n        Of Chebyshev series coefficients representing the quotient and\n        remainder.\n\n    See Also\n    --------\n    chebadd, chebsub, chebmul, chebpow\n\n    Notes\n    -----\n    In general, the (polynomial) division of one C-series by another\n    results in quotient and remainder terms that are not in the Chebyshev\n    polynomial basis set.  Thus, to express these results as C-series, it\n    is typically necessary to "reproject" the results onto said basis\n    set, which typically produces "unintuitive" (but correct) results;\n    see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import chebyshev as C\n    >>> c1 = (1,2,3)\n    >>> c2 = (3,2,1)\n    >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not\n    (array([ 3.]), array([-8., -4.]))\n    >>> c2 = (0,1,2,3)\n    >>> C.chebdiv(c2,c1) # neither "intuitive"\n    (array([ 0.,  2.]), array([-2., -4.]))\n\n    ')
    
    # Assigning a Call to a List (line 797):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 797)
    # Processing the call arguments (line 797)
    
    # Obtaining an instance of the builtin type 'list' (line 797)
    list_162880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 797)
    # Adding element type (line 797)
    # Getting the type of 'c1' (line 797)
    c1_162881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 28), list_162880, c1_162881)
    # Adding element type (line 797)
    # Getting the type of 'c2' (line 797)
    c2_162882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 28), list_162880, c2_162882)
    
    # Processing the call keyword arguments (line 797)
    kwargs_162883 = {}
    # Getting the type of 'pu' (line 797)
    pu_162878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 797)
    as_series_162879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 15), pu_162878, 'as_series')
    # Calling as_series(args, kwargs) (line 797)
    as_series_call_result_162884 = invoke(stypy.reporting.localization.Localization(__file__, 797, 15), as_series_162879, *[list_162880], **kwargs_162883)
    
    # Assigning a type to the variable 'call_assignment_161896' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161896', as_series_call_result_162884)
    
    # Assigning a Call to a Name (line 797):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162888 = {}
    # Getting the type of 'call_assignment_161896' (line 797)
    call_assignment_161896_162885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161896', False)
    # Obtaining the member '__getitem__' of a type (line 797)
    getitem___162886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 4), call_assignment_161896_162885, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162889 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162886, *[int_162887], **kwargs_162888)
    
    # Assigning a type to the variable 'call_assignment_161897' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161897', getitem___call_result_162889)
    
    # Assigning a Name to a Name (line 797):
    # Getting the type of 'call_assignment_161897' (line 797)
    call_assignment_161897_162890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161897')
    # Assigning a type to the variable 'c1' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 5), 'c1', call_assignment_161897_162890)
    
    # Assigning a Call to a Name (line 797):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162894 = {}
    # Getting the type of 'call_assignment_161896' (line 797)
    call_assignment_161896_162891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161896', False)
    # Obtaining the member '__getitem__' of a type (line 797)
    getitem___162892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 4), call_assignment_161896_162891, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162895 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162892, *[int_162893], **kwargs_162894)
    
    # Assigning a type to the variable 'call_assignment_161898' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161898', getitem___call_result_162895)
    
    # Assigning a Name to a Name (line 797):
    # Getting the type of 'call_assignment_161898' (line 797)
    call_assignment_161898_162896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'call_assignment_161898')
    # Assigning a type to the variable 'c2' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 9), 'c2', call_assignment_161898_162896)
    
    
    
    # Obtaining the type of the subscript
    int_162897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 10), 'int')
    # Getting the type of 'c2' (line 798)
    c2_162898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 7), 'c2')
    # Obtaining the member '__getitem__' of a type (line 798)
    getitem___162899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 7), c2_162898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 798)
    subscript_call_result_162900 = invoke(stypy.reporting.localization.Localization(__file__, 798, 7), getitem___162899, int_162897)
    
    int_162901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 17), 'int')
    # Applying the binary operator '==' (line 798)
    result_eq_162902 = python_operator(stypy.reporting.localization.Localization(__file__, 798, 7), '==', subscript_call_result_162900, int_162901)
    
    # Testing the type of an if condition (line 798)
    if_condition_162903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 4), result_eq_162902)
    # Assigning a type to the variable 'if_condition_162903' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'if_condition_162903', if_condition_162903)
    # SSA begins for if statement (line 798)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ZeroDivisionError(...): (line 799)
    # Processing the call keyword arguments (line 799)
    kwargs_162905 = {}
    # Getting the type of 'ZeroDivisionError' (line 799)
    ZeroDivisionError_162904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 14), 'ZeroDivisionError', False)
    # Calling ZeroDivisionError(args, kwargs) (line 799)
    ZeroDivisionError_call_result_162906 = invoke(stypy.reporting.localization.Localization(__file__, 799, 14), ZeroDivisionError_162904, *[], **kwargs_162905)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 799, 8), ZeroDivisionError_call_result_162906, 'raise parameter', BaseException)
    # SSA join for if statement (line 798)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 801):
    
    # Assigning a Call to a Name (line 801):
    
    # Call to len(...): (line 801)
    # Processing the call arguments (line 801)
    # Getting the type of 'c1' (line 801)
    c1_162908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 14), 'c1', False)
    # Processing the call keyword arguments (line 801)
    kwargs_162909 = {}
    # Getting the type of 'len' (line 801)
    len_162907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 10), 'len', False)
    # Calling len(args, kwargs) (line 801)
    len_call_result_162910 = invoke(stypy.reporting.localization.Localization(__file__, 801, 10), len_162907, *[c1_162908], **kwargs_162909)
    
    # Assigning a type to the variable 'lc1' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'lc1', len_call_result_162910)
    
    # Assigning a Call to a Name (line 802):
    
    # Assigning a Call to a Name (line 802):
    
    # Call to len(...): (line 802)
    # Processing the call arguments (line 802)
    # Getting the type of 'c2' (line 802)
    c2_162912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 14), 'c2', False)
    # Processing the call keyword arguments (line 802)
    kwargs_162913 = {}
    # Getting the type of 'len' (line 802)
    len_162911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 10), 'len', False)
    # Calling len(args, kwargs) (line 802)
    len_call_result_162914 = invoke(stypy.reporting.localization.Localization(__file__, 802, 10), len_162911, *[c2_162912], **kwargs_162913)
    
    # Assigning a type to the variable 'lc2' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'lc2', len_call_result_162914)
    
    
    # Getting the type of 'lc1' (line 803)
    lc1_162915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 7), 'lc1')
    # Getting the type of 'lc2' (line 803)
    lc2_162916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 13), 'lc2')
    # Applying the binary operator '<' (line 803)
    result_lt_162917 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 7), '<', lc1_162915, lc2_162916)
    
    # Testing the type of an if condition (line 803)
    if_condition_162918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 803, 4), result_lt_162917)
    # Assigning a type to the variable 'if_condition_162918' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 'if_condition_162918', if_condition_162918)
    # SSA begins for if statement (line 803)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 804)
    tuple_162919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 804)
    # Adding element type (line 804)
    
    # Obtaining the type of the subscript
    int_162920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 19), 'int')
    slice_162921 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 804, 15), None, int_162920, None)
    # Getting the type of 'c1' (line 804)
    c1_162922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 804)
    getitem___162923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 15), c1_162922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 804)
    subscript_call_result_162924 = invoke(stypy.reporting.localization.Localization(__file__, 804, 15), getitem___162923, slice_162921)
    
    int_162925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 22), 'int')
    # Applying the binary operator '*' (line 804)
    result_mul_162926 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 15), '*', subscript_call_result_162924, int_162925)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 15), tuple_162919, result_mul_162926)
    # Adding element type (line 804)
    # Getting the type of 'c1' (line 804)
    c1_162927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 25), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 15), tuple_162919, c1_162927)
    
    # Assigning a type to the variable 'stypy_return_type' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 8), 'stypy_return_type', tuple_162919)
    # SSA branch for the else part of an if statement (line 803)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lc2' (line 805)
    lc2_162928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 9), 'lc2')
    int_162929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 16), 'int')
    # Applying the binary operator '==' (line 805)
    result_eq_162930 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 9), '==', lc2_162928, int_162929)
    
    # Testing the type of an if condition (line 805)
    if_condition_162931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 805, 9), result_eq_162930)
    # Assigning a type to the variable 'if_condition_162931' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 9), 'if_condition_162931', if_condition_162931)
    # SSA begins for if statement (line 805)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 806)
    tuple_162932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 806)
    # Adding element type (line 806)
    # Getting the type of 'c1' (line 806)
    c1_162933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 15), 'c1')
    
    # Obtaining the type of the subscript
    int_162934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 21), 'int')
    # Getting the type of 'c2' (line 806)
    c2_162935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 18), 'c2')
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___162936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 18), c2_162935, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_162937 = invoke(stypy.reporting.localization.Localization(__file__, 806, 18), getitem___162936, int_162934)
    
    # Applying the binary operator 'div' (line 806)
    result_div_162938 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 15), 'div', c1_162933, subscript_call_result_162937)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), tuple_162932, result_div_162938)
    # Adding element type (line 806)
    
    # Obtaining the type of the subscript
    int_162939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 30), 'int')
    slice_162940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 806, 26), None, int_162939, None)
    # Getting the type of 'c1' (line 806)
    c1_162941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___162942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 26), c1_162941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_162943 = invoke(stypy.reporting.localization.Localization(__file__, 806, 26), getitem___162942, slice_162940)
    
    int_162944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 33), 'int')
    # Applying the binary operator '*' (line 806)
    result_mul_162945 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 26), '*', subscript_call_result_162943, int_162944)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), tuple_162932, result_mul_162945)
    
    # Assigning a type to the variable 'stypy_return_type' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'stypy_return_type', tuple_162932)
    # SSA branch for the else part of an if statement (line 805)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 808):
    
    # Assigning a Call to a Name (line 808):
    
    # Call to _cseries_to_zseries(...): (line 808)
    # Processing the call arguments (line 808)
    # Getting the type of 'c1' (line 808)
    c1_162947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 33), 'c1', False)
    # Processing the call keyword arguments (line 808)
    kwargs_162948 = {}
    # Getting the type of '_cseries_to_zseries' (line 808)
    _cseries_to_zseries_162946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 13), '_cseries_to_zseries', False)
    # Calling _cseries_to_zseries(args, kwargs) (line 808)
    _cseries_to_zseries_call_result_162949 = invoke(stypy.reporting.localization.Localization(__file__, 808, 13), _cseries_to_zseries_162946, *[c1_162947], **kwargs_162948)
    
    # Assigning a type to the variable 'z1' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'z1', _cseries_to_zseries_call_result_162949)
    
    # Assigning a Call to a Name (line 809):
    
    # Assigning a Call to a Name (line 809):
    
    # Call to _cseries_to_zseries(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'c2' (line 809)
    c2_162951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 33), 'c2', False)
    # Processing the call keyword arguments (line 809)
    kwargs_162952 = {}
    # Getting the type of '_cseries_to_zseries' (line 809)
    _cseries_to_zseries_162950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 13), '_cseries_to_zseries', False)
    # Calling _cseries_to_zseries(args, kwargs) (line 809)
    _cseries_to_zseries_call_result_162953 = invoke(stypy.reporting.localization.Localization(__file__, 809, 13), _cseries_to_zseries_162950, *[c2_162951], **kwargs_162952)
    
    # Assigning a type to the variable 'z2' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'z2', _cseries_to_zseries_call_result_162953)
    
    # Assigning a Call to a Tuple (line 810):
    
    # Assigning a Call to a Name:
    
    # Call to _zseries_div(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'z1' (line 810)
    z1_162955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 32), 'z1', False)
    # Getting the type of 'z2' (line 810)
    z2_162956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 36), 'z2', False)
    # Processing the call keyword arguments (line 810)
    kwargs_162957 = {}
    # Getting the type of '_zseries_div' (line 810)
    _zseries_div_162954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 19), '_zseries_div', False)
    # Calling _zseries_div(args, kwargs) (line 810)
    _zseries_div_call_result_162958 = invoke(stypy.reporting.localization.Localization(__file__, 810, 19), _zseries_div_162954, *[z1_162955, z2_162956], **kwargs_162957)
    
    # Assigning a type to the variable 'call_assignment_161899' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161899', _zseries_div_call_result_162958)
    
    # Assigning a Call to a Name (line 810):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 8), 'int')
    # Processing the call keyword arguments
    kwargs_162962 = {}
    # Getting the type of 'call_assignment_161899' (line 810)
    call_assignment_161899_162959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161899', False)
    # Obtaining the member '__getitem__' of a type (line 810)
    getitem___162960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), call_assignment_161899_162959, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162963 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162960, *[int_162961], **kwargs_162962)
    
    # Assigning a type to the variable 'call_assignment_161900' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161900', getitem___call_result_162963)
    
    # Assigning a Name to a Name (line 810):
    # Getting the type of 'call_assignment_161900' (line 810)
    call_assignment_161900_162964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161900')
    # Assigning a type to the variable 'quo' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'quo', call_assignment_161900_162964)
    
    # Assigning a Call to a Name (line 810):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_162967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 8), 'int')
    # Processing the call keyword arguments
    kwargs_162968 = {}
    # Getting the type of 'call_assignment_161899' (line 810)
    call_assignment_161899_162965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161899', False)
    # Obtaining the member '__getitem__' of a type (line 810)
    getitem___162966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), call_assignment_161899_162965, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_162969 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___162966, *[int_162967], **kwargs_162968)
    
    # Assigning a type to the variable 'call_assignment_161901' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161901', getitem___call_result_162969)
    
    # Assigning a Name to a Name (line 810):
    # Getting the type of 'call_assignment_161901' (line 810)
    call_assignment_161901_162970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'call_assignment_161901')
    # Assigning a type to the variable 'rem' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 13), 'rem', call_assignment_161901_162970)
    
    # Assigning a Call to a Name (line 811):
    
    # Assigning a Call to a Name (line 811):
    
    # Call to trimseq(...): (line 811)
    # Processing the call arguments (line 811)
    
    # Call to _zseries_to_cseries(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'quo' (line 811)
    quo_162974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 45), 'quo', False)
    # Processing the call keyword arguments (line 811)
    kwargs_162975 = {}
    # Getting the type of '_zseries_to_cseries' (line 811)
    _zseries_to_cseries_162973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 25), '_zseries_to_cseries', False)
    # Calling _zseries_to_cseries(args, kwargs) (line 811)
    _zseries_to_cseries_call_result_162976 = invoke(stypy.reporting.localization.Localization(__file__, 811, 25), _zseries_to_cseries_162973, *[quo_162974], **kwargs_162975)
    
    # Processing the call keyword arguments (line 811)
    kwargs_162977 = {}
    # Getting the type of 'pu' (line 811)
    pu_162971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 811)
    trimseq_162972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 14), pu_162971, 'trimseq')
    # Calling trimseq(args, kwargs) (line 811)
    trimseq_call_result_162978 = invoke(stypy.reporting.localization.Localization(__file__, 811, 14), trimseq_162972, *[_zseries_to_cseries_call_result_162976], **kwargs_162977)
    
    # Assigning a type to the variable 'quo' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'quo', trimseq_call_result_162978)
    
    # Assigning a Call to a Name (line 812):
    
    # Assigning a Call to a Name (line 812):
    
    # Call to trimseq(...): (line 812)
    # Processing the call arguments (line 812)
    
    # Call to _zseries_to_cseries(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'rem' (line 812)
    rem_162982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 45), 'rem', False)
    # Processing the call keyword arguments (line 812)
    kwargs_162983 = {}
    # Getting the type of '_zseries_to_cseries' (line 812)
    _zseries_to_cseries_162981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), '_zseries_to_cseries', False)
    # Calling _zseries_to_cseries(args, kwargs) (line 812)
    _zseries_to_cseries_call_result_162984 = invoke(stypy.reporting.localization.Localization(__file__, 812, 25), _zseries_to_cseries_162981, *[rem_162982], **kwargs_162983)
    
    # Processing the call keyword arguments (line 812)
    kwargs_162985 = {}
    # Getting the type of 'pu' (line 812)
    pu_162979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 14), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 812)
    trimseq_162980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 14), pu_162979, 'trimseq')
    # Calling trimseq(args, kwargs) (line 812)
    trimseq_call_result_162986 = invoke(stypy.reporting.localization.Localization(__file__, 812, 14), trimseq_162980, *[_zseries_to_cseries_call_result_162984], **kwargs_162985)
    
    # Assigning a type to the variable 'rem' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'rem', trimseq_call_result_162986)
    
    # Obtaining an instance of the builtin type 'tuple' (line 813)
    tuple_162987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 813)
    # Adding element type (line 813)
    # Getting the type of 'quo' (line 813)
    quo_162988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 15), 'quo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 15), tuple_162987, quo_162988)
    # Adding element type (line 813)
    # Getting the type of 'rem' (line 813)
    rem_162989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 20), 'rem')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 15), tuple_162987, rem_162989)
    
    # Assigning a type to the variable 'stypy_return_type' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'stypy_return_type', tuple_162987)
    # SSA join for if statement (line 805)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 803)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'chebdiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebdiv' in the type store
    # Getting the type of 'stypy_return_type' (line 750)
    stypy_return_type_162990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebdiv'
    return stypy_return_type_162990

# Assigning a type to the variable 'chebdiv' (line 750)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 0), 'chebdiv', chebdiv)

@norecursion
def chebpow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_162991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 29), 'int')
    defaults = [int_162991]
    # Create a new context for function 'chebpow'
    module_type_store = module_type_store.open_function_context('chebpow', 816, 0, False)
    
    # Passed parameters checking function
    chebpow.stypy_localization = localization
    chebpow.stypy_type_of_self = None
    chebpow.stypy_type_store = module_type_store
    chebpow.stypy_function_name = 'chebpow'
    chebpow.stypy_param_names_list = ['c', 'pow', 'maxpower']
    chebpow.stypy_varargs_param_name = None
    chebpow.stypy_kwargs_param_name = None
    chebpow.stypy_call_defaults = defaults
    chebpow.stypy_call_varargs = varargs
    chebpow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebpow', ['c', 'pow', 'maxpower'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebpow', localization, ['c', 'pow', 'maxpower'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebpow(...)' code ##################

    str_162992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, (-1)), 'str', 'Raise a Chebyshev series to a power.\n\n    Returns the Chebyshev series `c` raised to the power `pow`. The\n    argument `c` is a sequence of coefficients ordered from low to high.\n    i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Chebyshev series coefficients ordered from low to\n        high.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Chebyshev series of power.\n\n    See Also\n    --------\n    chebadd, chebsub, chebmul, chebdiv\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a Call to a List (line 848):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 848)
    # Processing the call arguments (line 848)
    
    # Obtaining an instance of the builtin type 'list' (line 848)
    list_162995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 848)
    # Adding element type (line 848)
    # Getting the type of 'c' (line 848)
    c_162996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 848, 23), list_162995, c_162996)
    
    # Processing the call keyword arguments (line 848)
    kwargs_162997 = {}
    # Getting the type of 'pu' (line 848)
    pu_162993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 848)
    as_series_162994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 10), pu_162993, 'as_series')
    # Calling as_series(args, kwargs) (line 848)
    as_series_call_result_162998 = invoke(stypy.reporting.localization.Localization(__file__, 848, 10), as_series_162994, *[list_162995], **kwargs_162997)
    
    # Assigning a type to the variable 'call_assignment_161902' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'call_assignment_161902', as_series_call_result_162998)
    
    # Assigning a Call to a Name (line 848):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_163001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 4), 'int')
    # Processing the call keyword arguments
    kwargs_163002 = {}
    # Getting the type of 'call_assignment_161902' (line 848)
    call_assignment_161902_162999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'call_assignment_161902', False)
    # Obtaining the member '__getitem__' of a type (line 848)
    getitem___163000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 4), call_assignment_161902_162999, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163003 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___163000, *[int_163001], **kwargs_163002)
    
    # Assigning a type to the variable 'call_assignment_161903' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'call_assignment_161903', getitem___call_result_163003)
    
    # Assigning a Name to a Name (line 848):
    # Getting the type of 'call_assignment_161903' (line 848)
    call_assignment_161903_163004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'call_assignment_161903')
    # Assigning a type to the variable 'c' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 5), 'c', call_assignment_161903_163004)
    
    # Assigning a Call to a Name (line 849):
    
    # Assigning a Call to a Name (line 849):
    
    # Call to int(...): (line 849)
    # Processing the call arguments (line 849)
    # Getting the type of 'pow' (line 849)
    pow_163006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 16), 'pow', False)
    # Processing the call keyword arguments (line 849)
    kwargs_163007 = {}
    # Getting the type of 'int' (line 849)
    int_163005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 12), 'int', False)
    # Calling int(args, kwargs) (line 849)
    int_call_result_163008 = invoke(stypy.reporting.localization.Localization(__file__, 849, 12), int_163005, *[pow_163006], **kwargs_163007)
    
    # Assigning a type to the variable 'power' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 4), 'power', int_call_result_163008)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'power' (line 850)
    power_163009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 7), 'power')
    # Getting the type of 'pow' (line 850)
    pow_163010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 16), 'pow')
    # Applying the binary operator '!=' (line 850)
    result_ne_163011 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 7), '!=', power_163009, pow_163010)
    
    
    # Getting the type of 'power' (line 850)
    power_163012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 23), 'power')
    int_163013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 31), 'int')
    # Applying the binary operator '<' (line 850)
    result_lt_163014 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 23), '<', power_163012, int_163013)
    
    # Applying the binary operator 'or' (line 850)
    result_or_keyword_163015 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 7), 'or', result_ne_163011, result_lt_163014)
    
    # Testing the type of an if condition (line 850)
    if_condition_163016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 850, 4), result_or_keyword_163015)
    # Assigning a type to the variable 'if_condition_163016' (line 850)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 4), 'if_condition_163016', if_condition_163016)
    # SSA begins for if statement (line 850)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 851)
    # Processing the call arguments (line 851)
    str_163018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 25), 'str', 'Power must be a non-negative integer.')
    # Processing the call keyword arguments (line 851)
    kwargs_163019 = {}
    # Getting the type of 'ValueError' (line 851)
    ValueError_163017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 851)
    ValueError_call_result_163020 = invoke(stypy.reporting.localization.Localization(__file__, 851, 14), ValueError_163017, *[str_163018], **kwargs_163019)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 851, 8), ValueError_call_result_163020, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 850)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxpower' (line 852)
    maxpower_163021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 9), 'maxpower')
    # Getting the type of 'None' (line 852)
    None_163022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 25), 'None')
    # Applying the binary operator 'isnot' (line 852)
    result_is_not_163023 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 9), 'isnot', maxpower_163021, None_163022)
    
    
    # Getting the type of 'power' (line 852)
    power_163024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 34), 'power')
    # Getting the type of 'maxpower' (line 852)
    maxpower_163025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 42), 'maxpower')
    # Applying the binary operator '>' (line 852)
    result_gt_163026 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 34), '>', power_163024, maxpower_163025)
    
    # Applying the binary operator 'and' (line 852)
    result_and_keyword_163027 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 9), 'and', result_is_not_163023, result_gt_163026)
    
    # Testing the type of an if condition (line 852)
    if_condition_163028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 852, 9), result_and_keyword_163027)
    # Assigning a type to the variable 'if_condition_163028' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 9), 'if_condition_163028', if_condition_163028)
    # SSA begins for if statement (line 852)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 853)
    # Processing the call arguments (line 853)
    str_163030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 25), 'str', 'Power is too large')
    # Processing the call keyword arguments (line 853)
    kwargs_163031 = {}
    # Getting the type of 'ValueError' (line 853)
    ValueError_163029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 853)
    ValueError_call_result_163032 = invoke(stypy.reporting.localization.Localization(__file__, 853, 14), ValueError_163029, *[str_163030], **kwargs_163031)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 853, 8), ValueError_call_result_163032, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 852)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 854)
    power_163033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 9), 'power')
    int_163034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 18), 'int')
    # Applying the binary operator '==' (line 854)
    result_eq_163035 = python_operator(stypy.reporting.localization.Localization(__file__, 854, 9), '==', power_163033, int_163034)
    
    # Testing the type of an if condition (line 854)
    if_condition_163036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 854, 9), result_eq_163035)
    # Assigning a type to the variable 'if_condition_163036' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 9), 'if_condition_163036', if_condition_163036)
    # SSA begins for if statement (line 854)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 855)
    # Processing the call arguments (line 855)
    
    # Obtaining an instance of the builtin type 'list' (line 855)
    list_163039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 855)
    # Adding element type (line 855)
    int_163040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 855, 24), list_163039, int_163040)
    
    # Processing the call keyword arguments (line 855)
    # Getting the type of 'c' (line 855)
    c_163041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 35), 'c', False)
    # Obtaining the member 'dtype' of a type (line 855)
    dtype_163042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 35), c_163041, 'dtype')
    keyword_163043 = dtype_163042
    kwargs_163044 = {'dtype': keyword_163043}
    # Getting the type of 'np' (line 855)
    np_163037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 855)
    array_163038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 15), np_163037, 'array')
    # Calling array(args, kwargs) (line 855)
    array_call_result_163045 = invoke(stypy.reporting.localization.Localization(__file__, 855, 15), array_163038, *[list_163039], **kwargs_163044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'stypy_return_type', array_call_result_163045)
    # SSA branch for the else part of an if statement (line 854)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 856)
    power_163046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 9), 'power')
    int_163047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 18), 'int')
    # Applying the binary operator '==' (line 856)
    result_eq_163048 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 9), '==', power_163046, int_163047)
    
    # Testing the type of an if condition (line 856)
    if_condition_163049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 856, 9), result_eq_163048)
    # Assigning a type to the variable 'if_condition_163049' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 9), 'if_condition_163049', if_condition_163049)
    # SSA begins for if statement (line 856)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 857)
    c_163050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'stypy_return_type', c_163050)
    # SSA branch for the else part of an if statement (line 856)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 861):
    
    # Assigning a Call to a Name (line 861):
    
    # Call to _cseries_to_zseries(...): (line 861)
    # Processing the call arguments (line 861)
    # Getting the type of 'c' (line 861)
    c_163052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 33), 'c', False)
    # Processing the call keyword arguments (line 861)
    kwargs_163053 = {}
    # Getting the type of '_cseries_to_zseries' (line 861)
    _cseries_to_zseries_163051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 13), '_cseries_to_zseries', False)
    # Calling _cseries_to_zseries(args, kwargs) (line 861)
    _cseries_to_zseries_call_result_163054 = invoke(stypy.reporting.localization.Localization(__file__, 861, 13), _cseries_to_zseries_163051, *[c_163052], **kwargs_163053)
    
    # Assigning a type to the variable 'zs' (line 861)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 8), 'zs', _cseries_to_zseries_call_result_163054)
    
    # Assigning a Name to a Name (line 862):
    
    # Assigning a Name to a Name (line 862):
    # Getting the type of 'zs' (line 862)
    zs_163055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 14), 'zs')
    # Assigning a type to the variable 'prd' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'prd', zs_163055)
    
    
    # Call to range(...): (line 863)
    # Processing the call arguments (line 863)
    int_163057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 23), 'int')
    # Getting the type of 'power' (line 863)
    power_163058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 26), 'power', False)
    int_163059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 34), 'int')
    # Applying the binary operator '+' (line 863)
    result_add_163060 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 26), '+', power_163058, int_163059)
    
    # Processing the call keyword arguments (line 863)
    kwargs_163061 = {}
    # Getting the type of 'range' (line 863)
    range_163056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 17), 'range', False)
    # Calling range(args, kwargs) (line 863)
    range_call_result_163062 = invoke(stypy.reporting.localization.Localization(__file__, 863, 17), range_163056, *[int_163057, result_add_163060], **kwargs_163061)
    
    # Testing the type of a for loop iterable (line 863)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 863, 8), range_call_result_163062)
    # Getting the type of the for loop variable (line 863)
    for_loop_var_163063 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 863, 8), range_call_result_163062)
    # Assigning a type to the variable 'i' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'i', for_loop_var_163063)
    # SSA begins for a for statement (line 863)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 864):
    
    # Assigning a Call to a Name (line 864):
    
    # Call to convolve(...): (line 864)
    # Processing the call arguments (line 864)
    # Getting the type of 'prd' (line 864)
    prd_163066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 30), 'prd', False)
    # Getting the type of 'zs' (line 864)
    zs_163067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 35), 'zs', False)
    # Processing the call keyword arguments (line 864)
    kwargs_163068 = {}
    # Getting the type of 'np' (line 864)
    np_163064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 18), 'np', False)
    # Obtaining the member 'convolve' of a type (line 864)
    convolve_163065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 18), np_163064, 'convolve')
    # Calling convolve(args, kwargs) (line 864)
    convolve_call_result_163069 = invoke(stypy.reporting.localization.Localization(__file__, 864, 18), convolve_163065, *[prd_163066, zs_163067], **kwargs_163068)
    
    # Assigning a type to the variable 'prd' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 12), 'prd', convolve_call_result_163069)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _zseries_to_cseries(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'prd' (line 865)
    prd_163071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 35), 'prd', False)
    # Processing the call keyword arguments (line 865)
    kwargs_163072 = {}
    # Getting the type of '_zseries_to_cseries' (line 865)
    _zseries_to_cseries_163070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 15), '_zseries_to_cseries', False)
    # Calling _zseries_to_cseries(args, kwargs) (line 865)
    _zseries_to_cseries_call_result_163073 = invoke(stypy.reporting.localization.Localization(__file__, 865, 15), _zseries_to_cseries_163070, *[prd_163071], **kwargs_163072)
    
    # Assigning a type to the variable 'stypy_return_type' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'stypy_return_type', _zseries_to_cseries_call_result_163073)
    # SSA join for if statement (line 856)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 854)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 852)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 850)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'chebpow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebpow' in the type store
    # Getting the type of 'stypy_return_type' (line 816)
    stypy_return_type_163074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebpow'
    return stypy_return_type_163074

# Assigning a type to the variable 'chebpow' (line 816)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 0), 'chebpow', chebpow)

@norecursion
def chebder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_163075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 17), 'int')
    int_163076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 24), 'int')
    int_163077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 32), 'int')
    defaults = [int_163075, int_163076, int_163077]
    # Create a new context for function 'chebder'
    module_type_store = module_type_store.open_function_context('chebder', 868, 0, False)
    
    # Passed parameters checking function
    chebder.stypy_localization = localization
    chebder.stypy_type_of_self = None
    chebder.stypy_type_store = module_type_store
    chebder.stypy_function_name = 'chebder'
    chebder.stypy_param_names_list = ['c', 'm', 'scl', 'axis']
    chebder.stypy_varargs_param_name = None
    chebder.stypy_kwargs_param_name = None
    chebder.stypy_call_defaults = defaults
    chebder.stypy_call_varargs = varargs
    chebder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebder', ['c', 'm', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebder', localization, ['c', 'm', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebder(...)' code ##################

    str_163078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, (-1)), 'str', '\n    Differentiate a Chebyshev series.\n\n    Returns the Chebyshev series coefficients `c` differentiated `m` times\n    along `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable). The argument\n    `c` is an array of coefficients from low to high degree along each\n    axis, e.g., [1,2,3] represents the series ``1*T_0 + 2*T_1 + 3*T_2``\n    while [[1,2],[1,2]] represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) +\n    2*T_0(x)*T_1(y) + 2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is\n    ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Chebyshev series coefficients. If c is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change of\n        variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Chebyshev series of the derivative.\n\n    See Also\n    --------\n    chebint\n\n    Notes\n    -----\n    In general, the result of differentiating a C-series needs to be\n    "reprojected" onto the C-series basis set. Thus, typically, the\n    result of this function is "unintuitive," albeit correct; see Examples\n    section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import chebyshev as C\n    >>> c = (1,2,3,4)\n    >>> C.chebder(c)\n    array([ 14.,  12.,  24.])\n    >>> C.chebder(c,3)\n    array([ 96.])\n    >>> C.chebder(c,scl=-1)\n    array([-14., -12., -24.])\n    >>> C.chebder(c,2,-1)\n    array([ 12.,  96.])\n\n    ')
    
    # Assigning a Call to a Name (line 928):
    
    # Assigning a Call to a Name (line 928):
    
    # Call to array(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 'c' (line 928)
    c_163081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 17), 'c', False)
    # Processing the call keyword arguments (line 928)
    int_163082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 26), 'int')
    keyword_163083 = int_163082
    int_163084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 34), 'int')
    keyword_163085 = int_163084
    kwargs_163086 = {'copy': keyword_163085, 'ndmin': keyword_163083}
    # Getting the type of 'np' (line 928)
    np_163079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 928)
    array_163080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 8), np_163079, 'array')
    # Calling array(args, kwargs) (line 928)
    array_call_result_163087 = invoke(stypy.reporting.localization.Localization(__file__, 928, 8), array_163080, *[c_163081], **kwargs_163086)
    
    # Assigning a type to the variable 'c' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 4), 'c', array_call_result_163087)
    
    
    # Getting the type of 'c' (line 929)
    c_163088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 929)
    dtype_163089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 7), c_163088, 'dtype')
    # Obtaining the member 'char' of a type (line 929)
    char_163090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 7), dtype_163089, 'char')
    str_163091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 929)
    result_contains_163092 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 7), 'in', char_163090, str_163091)
    
    # Testing the type of an if condition (line 929)
    if_condition_163093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 929, 4), result_contains_163092)
    # Assigning a type to the variable 'if_condition_163093' (line 929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 4), 'if_condition_163093', if_condition_163093)
    # SSA begins for if statement (line 929)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 930):
    
    # Assigning a Call to a Name (line 930):
    
    # Call to astype(...): (line 930)
    # Processing the call arguments (line 930)
    # Getting the type of 'np' (line 930)
    np_163096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 930)
    double_163097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 21), np_163096, 'double')
    # Processing the call keyword arguments (line 930)
    kwargs_163098 = {}
    # Getting the type of 'c' (line 930)
    c_163094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 930)
    astype_163095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 12), c_163094, 'astype')
    # Calling astype(args, kwargs) (line 930)
    astype_call_result_163099 = invoke(stypy.reporting.localization.Localization(__file__, 930, 12), astype_163095, *[double_163097], **kwargs_163098)
    
    # Assigning a type to the variable 'c' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'c', astype_call_result_163099)
    # SSA join for if statement (line 929)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 931):
    
    # Assigning a Subscript to a Name (line 931):
    
    # Obtaining the type of the subscript
    int_163100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 931)
    list_163105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 931)
    # Adding element type (line 931)
    # Getting the type of 'm' (line 931)
    m_163106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 34), list_163105, m_163106)
    # Adding element type (line 931)
    # Getting the type of 'axis' (line 931)
    axis_163107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 34), list_163105, axis_163107)
    
    comprehension_163108 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 18), list_163105)
    # Assigning a type to the variable 't' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 18), 't', comprehension_163108)
    
    # Call to int(...): (line 931)
    # Processing the call arguments (line 931)
    # Getting the type of 't' (line 931)
    t_163102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 22), 't', False)
    # Processing the call keyword arguments (line 931)
    kwargs_163103 = {}
    # Getting the type of 'int' (line 931)
    int_163101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 18), 'int', False)
    # Calling int(args, kwargs) (line 931)
    int_call_result_163104 = invoke(stypy.reporting.localization.Localization(__file__, 931, 18), int_163101, *[t_163102], **kwargs_163103)
    
    list_163109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 18), list_163109, int_call_result_163104)
    # Obtaining the member '__getitem__' of a type (line 931)
    getitem___163110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 4), list_163109, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 931)
    subscript_call_result_163111 = invoke(stypy.reporting.localization.Localization(__file__, 931, 4), getitem___163110, int_163100)
    
    # Assigning a type to the variable 'tuple_var_assignment_161904' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'tuple_var_assignment_161904', subscript_call_result_163111)
    
    # Assigning a Subscript to a Name (line 931):
    
    # Obtaining the type of the subscript
    int_163112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 931)
    list_163117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 931)
    # Adding element type (line 931)
    # Getting the type of 'm' (line 931)
    m_163118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 34), list_163117, m_163118)
    # Adding element type (line 931)
    # Getting the type of 'axis' (line 931)
    axis_163119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 34), list_163117, axis_163119)
    
    comprehension_163120 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 18), list_163117)
    # Assigning a type to the variable 't' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 18), 't', comprehension_163120)
    
    # Call to int(...): (line 931)
    # Processing the call arguments (line 931)
    # Getting the type of 't' (line 931)
    t_163114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 22), 't', False)
    # Processing the call keyword arguments (line 931)
    kwargs_163115 = {}
    # Getting the type of 'int' (line 931)
    int_163113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 18), 'int', False)
    # Calling int(args, kwargs) (line 931)
    int_call_result_163116 = invoke(stypy.reporting.localization.Localization(__file__, 931, 18), int_163113, *[t_163114], **kwargs_163115)
    
    list_163121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 18), list_163121, int_call_result_163116)
    # Obtaining the member '__getitem__' of a type (line 931)
    getitem___163122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 4), list_163121, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 931)
    subscript_call_result_163123 = invoke(stypy.reporting.localization.Localization(__file__, 931, 4), getitem___163122, int_163112)
    
    # Assigning a type to the variable 'tuple_var_assignment_161905' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'tuple_var_assignment_161905', subscript_call_result_163123)
    
    # Assigning a Name to a Name (line 931):
    # Getting the type of 'tuple_var_assignment_161904' (line 931)
    tuple_var_assignment_161904_163124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'tuple_var_assignment_161904')
    # Assigning a type to the variable 'cnt' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'cnt', tuple_var_assignment_161904_163124)
    
    # Assigning a Name to a Name (line 931):
    # Getting the type of 'tuple_var_assignment_161905' (line 931)
    tuple_var_assignment_161905_163125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'tuple_var_assignment_161905')
    # Assigning a type to the variable 'iaxis' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 9), 'iaxis', tuple_var_assignment_161905_163125)
    
    
    # Getting the type of 'cnt' (line 933)
    cnt_163126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 7), 'cnt')
    # Getting the type of 'm' (line 933)
    m_163127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 14), 'm')
    # Applying the binary operator '!=' (line 933)
    result_ne_163128 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 7), '!=', cnt_163126, m_163127)
    
    # Testing the type of an if condition (line 933)
    if_condition_163129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 933, 4), result_ne_163128)
    # Assigning a type to the variable 'if_condition_163129' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'if_condition_163129', if_condition_163129)
    # SSA begins for if statement (line 933)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 934)
    # Processing the call arguments (line 934)
    str_163131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 25), 'str', 'The order of derivation must be integer')
    # Processing the call keyword arguments (line 934)
    kwargs_163132 = {}
    # Getting the type of 'ValueError' (line 934)
    ValueError_163130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 934)
    ValueError_call_result_163133 = invoke(stypy.reporting.localization.Localization(__file__, 934, 14), ValueError_163130, *[str_163131], **kwargs_163132)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 934, 8), ValueError_call_result_163133, 'raise parameter', BaseException)
    # SSA join for if statement (line 933)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 935)
    cnt_163134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 7), 'cnt')
    int_163135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 13), 'int')
    # Applying the binary operator '<' (line 935)
    result_lt_163136 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 7), '<', cnt_163134, int_163135)
    
    # Testing the type of an if condition (line 935)
    if_condition_163137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 935, 4), result_lt_163136)
    # Assigning a type to the variable 'if_condition_163137' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 4), 'if_condition_163137', if_condition_163137)
    # SSA begins for if statement (line 935)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 936)
    # Processing the call arguments (line 936)
    str_163139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 25), 'str', 'The order of derivation must be non-negative')
    # Processing the call keyword arguments (line 936)
    kwargs_163140 = {}
    # Getting the type of 'ValueError' (line 936)
    ValueError_163138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 936)
    ValueError_call_result_163141 = invoke(stypy.reporting.localization.Localization(__file__, 936, 14), ValueError_163138, *[str_163139], **kwargs_163140)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 936, 8), ValueError_call_result_163141, 'raise parameter', BaseException)
    # SSA join for if statement (line 935)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 937)
    iaxis_163142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 7), 'iaxis')
    # Getting the type of 'axis' (line 937)
    axis_163143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 16), 'axis')
    # Applying the binary operator '!=' (line 937)
    result_ne_163144 = python_operator(stypy.reporting.localization.Localization(__file__, 937, 7), '!=', iaxis_163142, axis_163143)
    
    # Testing the type of an if condition (line 937)
    if_condition_163145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 937, 4), result_ne_163144)
    # Assigning a type to the variable 'if_condition_163145' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'if_condition_163145', if_condition_163145)
    # SSA begins for if statement (line 937)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 938)
    # Processing the call arguments (line 938)
    str_163147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 938)
    kwargs_163148 = {}
    # Getting the type of 'ValueError' (line 938)
    ValueError_163146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 938)
    ValueError_call_result_163149 = invoke(stypy.reporting.localization.Localization(__file__, 938, 14), ValueError_163146, *[str_163147], **kwargs_163148)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 938, 8), ValueError_call_result_163149, 'raise parameter', BaseException)
    # SSA join for if statement (line 937)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 939)
    c_163150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 939)
    ndim_163151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 12), c_163150, 'ndim')
    # Applying the 'usub' unary operator (line 939)
    result___neg___163152 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 11), 'usub', ndim_163151)
    
    # Getting the type of 'iaxis' (line 939)
    iaxis_163153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 22), 'iaxis')
    # Applying the binary operator '<=' (line 939)
    result_le_163154 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 11), '<=', result___neg___163152, iaxis_163153)
    # Getting the type of 'c' (line 939)
    c_163155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 939)
    ndim_163156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 30), c_163155, 'ndim')
    # Applying the binary operator '<' (line 939)
    result_lt_163157 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 11), '<', iaxis_163153, ndim_163156)
    # Applying the binary operator '&' (line 939)
    result_and__163158 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 11), '&', result_le_163154, result_lt_163157)
    
    # Applying the 'not' unary operator (line 939)
    result_not__163159 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 7), 'not', result_and__163158)
    
    # Testing the type of an if condition (line 939)
    if_condition_163160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 939, 4), result_not__163159)
    # Assigning a type to the variable 'if_condition_163160' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'if_condition_163160', if_condition_163160)
    # SSA begins for if statement (line 939)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 940)
    # Processing the call arguments (line 940)
    str_163162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 940)
    kwargs_163163 = {}
    # Getting the type of 'ValueError' (line 940)
    ValueError_163161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 940)
    ValueError_call_result_163164 = invoke(stypy.reporting.localization.Localization(__file__, 940, 14), ValueError_163161, *[str_163162], **kwargs_163163)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 940, 8), ValueError_call_result_163164, 'raise parameter', BaseException)
    # SSA join for if statement (line 939)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 941)
    iaxis_163165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 7), 'iaxis')
    int_163166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 15), 'int')
    # Applying the binary operator '<' (line 941)
    result_lt_163167 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 7), '<', iaxis_163165, int_163166)
    
    # Testing the type of an if condition (line 941)
    if_condition_163168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 941, 4), result_lt_163167)
    # Assigning a type to the variable 'if_condition_163168' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'if_condition_163168', if_condition_163168)
    # SSA begins for if statement (line 941)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 942)
    iaxis_163169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 8), 'iaxis')
    # Getting the type of 'c' (line 942)
    c_163170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 942)
    ndim_163171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 17), c_163170, 'ndim')
    # Applying the binary operator '+=' (line 942)
    result_iadd_163172 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 8), '+=', iaxis_163169, ndim_163171)
    # Assigning a type to the variable 'iaxis' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 8), 'iaxis', result_iadd_163172)
    
    # SSA join for if statement (line 941)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 944)
    cnt_163173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 7), 'cnt')
    int_163174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 14), 'int')
    # Applying the binary operator '==' (line 944)
    result_eq_163175 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 7), '==', cnt_163173, int_163174)
    
    # Testing the type of an if condition (line 944)
    if_condition_163176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 944, 4), result_eq_163175)
    # Assigning a type to the variable 'if_condition_163176' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'if_condition_163176', if_condition_163176)
    # SSA begins for if statement (line 944)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 945)
    c_163177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 8), 'stypy_return_type', c_163177)
    # SSA join for if statement (line 944)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 947):
    
    # Assigning a Call to a Name (line 947):
    
    # Call to rollaxis(...): (line 947)
    # Processing the call arguments (line 947)
    # Getting the type of 'c' (line 947)
    c_163180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 20), 'c', False)
    # Getting the type of 'iaxis' (line 947)
    iaxis_163181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 947)
    kwargs_163182 = {}
    # Getting the type of 'np' (line 947)
    np_163178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 947)
    rollaxis_163179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 947, 8), np_163178, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 947)
    rollaxis_call_result_163183 = invoke(stypy.reporting.localization.Localization(__file__, 947, 8), rollaxis_163179, *[c_163180, iaxis_163181], **kwargs_163182)
    
    # Assigning a type to the variable 'c' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'c', rollaxis_call_result_163183)
    
    # Assigning a Call to a Name (line 948):
    
    # Assigning a Call to a Name (line 948):
    
    # Call to len(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'c' (line 948)
    c_163185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 12), 'c', False)
    # Processing the call keyword arguments (line 948)
    kwargs_163186 = {}
    # Getting the type of 'len' (line 948)
    len_163184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'len', False)
    # Calling len(args, kwargs) (line 948)
    len_call_result_163187 = invoke(stypy.reporting.localization.Localization(__file__, 948, 8), len_163184, *[c_163185], **kwargs_163186)
    
    # Assigning a type to the variable 'n' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'n', len_call_result_163187)
    
    
    # Getting the type of 'cnt' (line 949)
    cnt_163188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 7), 'cnt')
    # Getting the type of 'n' (line 949)
    n_163189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 14), 'n')
    # Applying the binary operator '>=' (line 949)
    result_ge_163190 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 7), '>=', cnt_163188, n_163189)
    
    # Testing the type of an if condition (line 949)
    if_condition_163191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 949, 4), result_ge_163190)
    # Assigning a type to the variable 'if_condition_163191' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 4), 'if_condition_163191', if_condition_163191)
    # SSA begins for if statement (line 949)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 950):
    
    # Assigning a BinOp to a Name (line 950):
    
    # Obtaining the type of the subscript
    int_163192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 15), 'int')
    slice_163193 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 950, 12), None, int_163192, None)
    # Getting the type of 'c' (line 950)
    c_163194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 950)
    getitem___163195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 12), c_163194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 950)
    subscript_call_result_163196 = invoke(stypy.reporting.localization.Localization(__file__, 950, 12), getitem___163195, slice_163193)
    
    int_163197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 18), 'int')
    # Applying the binary operator '*' (line 950)
    result_mul_163198 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 12), '*', subscript_call_result_163196, int_163197)
    
    # Assigning a type to the variable 'c' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'c', result_mul_163198)
    # SSA branch for the else part of an if statement (line 949)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 952)
    # Processing the call arguments (line 952)
    # Getting the type of 'cnt' (line 952)
    cnt_163200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 23), 'cnt', False)
    # Processing the call keyword arguments (line 952)
    kwargs_163201 = {}
    # Getting the type of 'range' (line 952)
    range_163199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 17), 'range', False)
    # Calling range(args, kwargs) (line 952)
    range_call_result_163202 = invoke(stypy.reporting.localization.Localization(__file__, 952, 17), range_163199, *[cnt_163200], **kwargs_163201)
    
    # Testing the type of a for loop iterable (line 952)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 952, 8), range_call_result_163202)
    # Getting the type of the for loop variable (line 952)
    for_loop_var_163203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 952, 8), range_call_result_163202)
    # Assigning a type to the variable 'i' (line 952)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'i', for_loop_var_163203)
    # SSA begins for a for statement (line 952)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 953):
    
    # Assigning a BinOp to a Name (line 953):
    # Getting the type of 'n' (line 953)
    n_163204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 16), 'n')
    int_163205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 20), 'int')
    # Applying the binary operator '-' (line 953)
    result_sub_163206 = python_operator(stypy.reporting.localization.Localization(__file__, 953, 16), '-', n_163204, int_163205)
    
    # Assigning a type to the variable 'n' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 12), 'n', result_sub_163206)
    
    # Getting the type of 'c' (line 954)
    c_163207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 12), 'c')
    # Getting the type of 'scl' (line 954)
    scl_163208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 17), 'scl')
    # Applying the binary operator '*=' (line 954)
    result_imul_163209 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 12), '*=', c_163207, scl_163208)
    # Assigning a type to the variable 'c' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 12), 'c', result_imul_163209)
    
    
    # Assigning a Call to a Name (line 955):
    
    # Assigning a Call to a Name (line 955):
    
    # Call to empty(...): (line 955)
    # Processing the call arguments (line 955)
    
    # Obtaining an instance of the builtin type 'tuple' (line 955)
    tuple_163212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 955)
    # Adding element type (line 955)
    # Getting the type of 'n' (line 955)
    n_163213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 955, 28), tuple_163212, n_163213)
    
    
    # Obtaining the type of the subscript
    int_163214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 42), 'int')
    slice_163215 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 955, 34), int_163214, None, None)
    # Getting the type of 'c' (line 955)
    c_163216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 955)
    shape_163217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 34), c_163216, 'shape')
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___163218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 34), shape_163217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_163219 = invoke(stypy.reporting.localization.Localization(__file__, 955, 34), getitem___163218, slice_163215)
    
    # Applying the binary operator '+' (line 955)
    result_add_163220 = python_operator(stypy.reporting.localization.Localization(__file__, 955, 27), '+', tuple_163212, subscript_call_result_163219)
    
    # Processing the call keyword arguments (line 955)
    # Getting the type of 'c' (line 955)
    c_163221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 955)
    dtype_163222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 53), c_163221, 'dtype')
    keyword_163223 = dtype_163222
    kwargs_163224 = {'dtype': keyword_163223}
    # Getting the type of 'np' (line 955)
    np_163210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 955)
    empty_163211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 18), np_163210, 'empty')
    # Calling empty(args, kwargs) (line 955)
    empty_call_result_163225 = invoke(stypy.reporting.localization.Localization(__file__, 955, 18), empty_163211, *[result_add_163220], **kwargs_163224)
    
    # Assigning a type to the variable 'der' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 12), 'der', empty_call_result_163225)
    
    
    # Call to range(...): (line 956)
    # Processing the call arguments (line 956)
    # Getting the type of 'n' (line 956)
    n_163227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 27), 'n', False)
    int_163228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 30), 'int')
    int_163229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 33), 'int')
    # Processing the call keyword arguments (line 956)
    kwargs_163230 = {}
    # Getting the type of 'range' (line 956)
    range_163226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 21), 'range', False)
    # Calling range(args, kwargs) (line 956)
    range_call_result_163231 = invoke(stypy.reporting.localization.Localization(__file__, 956, 21), range_163226, *[n_163227, int_163228, int_163229], **kwargs_163230)
    
    # Testing the type of a for loop iterable (line 956)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 956, 12), range_call_result_163231)
    # Getting the type of the for loop variable (line 956)
    for_loop_var_163232 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 956, 12), range_call_result_163231)
    # Assigning a type to the variable 'j' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 12), 'j', for_loop_var_163232)
    # SSA begins for a for statement (line 956)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 957):
    
    # Assigning a BinOp to a Subscript (line 957):
    int_163233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 30), 'int')
    # Getting the type of 'j' (line 957)
    j_163234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 32), 'j')
    # Applying the binary operator '*' (line 957)
    result_mul_163235 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 30), '*', int_163233, j_163234)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 957)
    j_163236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 37), 'j')
    # Getting the type of 'c' (line 957)
    c_163237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 35), 'c')
    # Obtaining the member '__getitem__' of a type (line 957)
    getitem___163238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 35), c_163237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 957)
    subscript_call_result_163239 = invoke(stypy.reporting.localization.Localization(__file__, 957, 35), getitem___163238, j_163236)
    
    # Applying the binary operator '*' (line 957)
    result_mul_163240 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 29), '*', result_mul_163235, subscript_call_result_163239)
    
    # Getting the type of 'der' (line 957)
    der_163241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 16), 'der')
    # Getting the type of 'j' (line 957)
    j_163242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 20), 'j')
    int_163243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 24), 'int')
    # Applying the binary operator '-' (line 957)
    result_sub_163244 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 20), '-', j_163242, int_163243)
    
    # Storing an element on a container (line 957)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 957, 16), der_163241, (result_sub_163244, result_mul_163240))
    
    # Getting the type of 'c' (line 958)
    c_163245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 16), 'c')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 958)
    j_163246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 18), 'j')
    int_163247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 22), 'int')
    # Applying the binary operator '-' (line 958)
    result_sub_163248 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 18), '-', j_163246, int_163247)
    
    # Getting the type of 'c' (line 958)
    c_163249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 16), 'c')
    # Obtaining the member '__getitem__' of a type (line 958)
    getitem___163250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 16), c_163249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 958)
    subscript_call_result_163251 = invoke(stypy.reporting.localization.Localization(__file__, 958, 16), getitem___163250, result_sub_163248)
    
    # Getting the type of 'j' (line 958)
    j_163252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 29), 'j')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 958)
    j_163253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 33), 'j')
    # Getting the type of 'c' (line 958)
    c_163254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 31), 'c')
    # Obtaining the member '__getitem__' of a type (line 958)
    getitem___163255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 31), c_163254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 958)
    subscript_call_result_163256 = invoke(stypy.reporting.localization.Localization(__file__, 958, 31), getitem___163255, j_163253)
    
    # Applying the binary operator '*' (line 958)
    result_mul_163257 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 29), '*', j_163252, subscript_call_result_163256)
    
    # Getting the type of 'j' (line 958)
    j_163258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 38), 'j')
    int_163259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 42), 'int')
    # Applying the binary operator '-' (line 958)
    result_sub_163260 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 38), '-', j_163258, int_163259)
    
    # Applying the binary operator 'div' (line 958)
    result_div_163261 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 28), 'div', result_mul_163257, result_sub_163260)
    
    # Applying the binary operator '+=' (line 958)
    result_iadd_163262 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 16), '+=', subscript_call_result_163251, result_div_163261)
    # Getting the type of 'c' (line 958)
    c_163263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 16), 'c')
    # Getting the type of 'j' (line 958)
    j_163264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 18), 'j')
    int_163265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 22), 'int')
    # Applying the binary operator '-' (line 958)
    result_sub_163266 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 18), '-', j_163264, int_163265)
    
    # Storing an element on a container (line 958)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 16), c_163263, (result_sub_163266, result_iadd_163262))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 959)
    n_163267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 15), 'n')
    int_163268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 19), 'int')
    # Applying the binary operator '>' (line 959)
    result_gt_163269 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 15), '>', n_163267, int_163268)
    
    # Testing the type of an if condition (line 959)
    if_condition_163270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 959, 12), result_gt_163269)
    # Assigning a type to the variable 'if_condition_163270' (line 959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 12), 'if_condition_163270', if_condition_163270)
    # SSA begins for if statement (line 959)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 960):
    
    # Assigning a BinOp to a Subscript (line 960):
    int_163271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 25), 'int')
    
    # Obtaining the type of the subscript
    int_163272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 29), 'int')
    # Getting the type of 'c' (line 960)
    c_163273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 27), 'c')
    # Obtaining the member '__getitem__' of a type (line 960)
    getitem___163274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 27), c_163273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 960)
    subscript_call_result_163275 = invoke(stypy.reporting.localization.Localization(__file__, 960, 27), getitem___163274, int_163272)
    
    # Applying the binary operator '*' (line 960)
    result_mul_163276 = python_operator(stypy.reporting.localization.Localization(__file__, 960, 25), '*', int_163271, subscript_call_result_163275)
    
    # Getting the type of 'der' (line 960)
    der_163277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 16), 'der')
    int_163278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 20), 'int')
    # Storing an element on a container (line 960)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 16), der_163277, (int_163278, result_mul_163276))
    # SSA join for if statement (line 959)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 961):
    
    # Assigning a Subscript to a Subscript (line 961):
    
    # Obtaining the type of the subscript
    int_163279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 23), 'int')
    # Getting the type of 'c' (line 961)
    c_163280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 961)
    getitem___163281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 21), c_163280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 961)
    subscript_call_result_163282 = invoke(stypy.reporting.localization.Localization(__file__, 961, 21), getitem___163281, int_163279)
    
    # Getting the type of 'der' (line 961)
    der_163283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 12), 'der')
    int_163284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 16), 'int')
    # Storing an element on a container (line 961)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 12), der_163283, (int_163284, subscript_call_result_163282))
    
    # Assigning a Name to a Name (line 962):
    
    # Assigning a Name to a Name (line 962):
    # Getting the type of 'der' (line 962)
    der_163285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 16), 'der')
    # Assigning a type to the variable 'c' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 12), 'c', der_163285)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 949)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 963):
    
    # Assigning a Call to a Name (line 963):
    
    # Call to rollaxis(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'c' (line 963)
    c_163288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 20), 'c', False)
    int_163289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 23), 'int')
    # Getting the type of 'iaxis' (line 963)
    iaxis_163290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 26), 'iaxis', False)
    int_163291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 34), 'int')
    # Applying the binary operator '+' (line 963)
    result_add_163292 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 26), '+', iaxis_163290, int_163291)
    
    # Processing the call keyword arguments (line 963)
    kwargs_163293 = {}
    # Getting the type of 'np' (line 963)
    np_163286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 963)
    rollaxis_163287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 8), np_163286, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 963)
    rollaxis_call_result_163294 = invoke(stypy.reporting.localization.Localization(__file__, 963, 8), rollaxis_163287, *[c_163288, int_163289, result_add_163292], **kwargs_163293)
    
    # Assigning a type to the variable 'c' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'c', rollaxis_call_result_163294)
    # Getting the type of 'c' (line 964)
    c_163295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'stypy_return_type', c_163295)
    
    # ################# End of 'chebder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebder' in the type store
    # Getting the type of 'stypy_return_type' (line 868)
    stypy_return_type_163296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebder'
    return stypy_return_type_163296

# Assigning a type to the variable 'chebder' (line 868)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 0), 'chebder', chebder)

@norecursion
def chebint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_163297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 17), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 967)
    list_163298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 967)
    
    int_163299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 31), 'int')
    int_163300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 38), 'int')
    int_163301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 46), 'int')
    defaults = [int_163297, list_163298, int_163299, int_163300, int_163301]
    # Create a new context for function 'chebint'
    module_type_store = module_type_store.open_function_context('chebint', 967, 0, False)
    
    # Passed parameters checking function
    chebint.stypy_localization = localization
    chebint.stypy_type_of_self = None
    chebint.stypy_type_store = module_type_store
    chebint.stypy_function_name = 'chebint'
    chebint.stypy_param_names_list = ['c', 'm', 'k', 'lbnd', 'scl', 'axis']
    chebint.stypy_varargs_param_name = None
    chebint.stypy_kwargs_param_name = None
    chebint.stypy_call_defaults = defaults
    chebint.stypy_call_varargs = varargs
    chebint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebint', ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebint', localization, ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebint(...)' code ##################

    str_163302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, (-1)), 'str', '\n    Integrate a Chebyshev series.\n\n    Returns the Chebyshev series coefficients `c` integrated `m` times from\n    `lbnd` along `axis`. At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.)  The argument `c` is an array of\n    coefficients from low to high degree along each axis, e.g., [1,2,3]\n    represents the series ``T_0 + 2*T_1 + 3*T_2`` while [[1,2],[1,2]]\n    represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) + 2*T_0(x)*T_1(y) +\n    2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Chebyshev series coefficients. If c is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at zero\n        is the first value in the list, the value of the second integral\n        at zero is the second value, etc.  If ``k == []`` (the default),\n        all constants are set to zero.  If ``m == 1``, a single scalar can\n        be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        C-series coefficients of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 1``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or\n        ``np.isscalar(scl) == False``.\n\n    See Also\n    --------\n    chebder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.\n    Why is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`.  Then\n    .. math::`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a`- perhaps not what one would have first thought.\n\n    Also note that, in general, the result of integrating a C-series needs\n    to be "reprojected" onto the C-series basis set.  Thus, typically,\n    the result of this function is "unintuitive," albeit correct; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial import chebyshev as C\n    >>> c = (1,2,3)\n    >>> C.chebint(c)\n    array([ 0.5, -0.5,  0.5,  0.5])\n    >>> C.chebint(c,3)\n    array([ 0.03125   , -0.1875    ,  0.04166667, -0.05208333,  0.01041667,\n            0.00625   ])\n    >>> C.chebint(c, k=3)\n    array([ 3.5, -0.5,  0.5,  0.5])\n    >>> C.chebint(c,lbnd=-2)\n    array([ 8.5, -0.5,  0.5,  0.5])\n    >>> C.chebint(c,scl=-2)\n    array([-1.,  1., -1., -1.])\n\n    ')
    
    # Assigning a Call to a Name (line 1052):
    
    # Assigning a Call to a Name (line 1052):
    
    # Call to array(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'c' (line 1052)
    c_163305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 17), 'c', False)
    # Processing the call keyword arguments (line 1052)
    int_163306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 26), 'int')
    keyword_163307 = int_163306
    int_163308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 34), 'int')
    keyword_163309 = int_163308
    kwargs_163310 = {'copy': keyword_163309, 'ndmin': keyword_163307}
    # Getting the type of 'np' (line 1052)
    np_163303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1052)
    array_163304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 8), np_163303, 'array')
    # Calling array(args, kwargs) (line 1052)
    array_call_result_163311 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 8), array_163304, *[c_163305], **kwargs_163310)
    
    # Assigning a type to the variable 'c' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'c', array_call_result_163311)
    
    
    # Getting the type of 'c' (line 1053)
    c_163312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 1053)
    dtype_163313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 7), c_163312, 'dtype')
    # Obtaining the member 'char' of a type (line 1053)
    char_163314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 7), dtype_163313, 'char')
    str_163315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 1053)
    result_contains_163316 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 7), 'in', char_163314, str_163315)
    
    # Testing the type of an if condition (line 1053)
    if_condition_163317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1053, 4), result_contains_163316)
    # Assigning a type to the variable 'if_condition_163317' (line 1053)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 4), 'if_condition_163317', if_condition_163317)
    # SSA begins for if statement (line 1053)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1054):
    
    # Assigning a Call to a Name (line 1054):
    
    # Call to astype(...): (line 1054)
    # Processing the call arguments (line 1054)
    # Getting the type of 'np' (line 1054)
    np_163320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 1054)
    double_163321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 21), np_163320, 'double')
    # Processing the call keyword arguments (line 1054)
    kwargs_163322 = {}
    # Getting the type of 'c' (line 1054)
    c_163318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 1054)
    astype_163319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 12), c_163318, 'astype')
    # Calling astype(args, kwargs) (line 1054)
    astype_call_result_163323 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 12), astype_163319, *[double_163321], **kwargs_163322)
    
    # Assigning a type to the variable 'c' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'c', astype_call_result_163323)
    # SSA join for if statement (line 1053)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to iterable(...): (line 1055)
    # Processing the call arguments (line 1055)
    # Getting the type of 'k' (line 1055)
    k_163326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 23), 'k', False)
    # Processing the call keyword arguments (line 1055)
    kwargs_163327 = {}
    # Getting the type of 'np' (line 1055)
    np_163324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 11), 'np', False)
    # Obtaining the member 'iterable' of a type (line 1055)
    iterable_163325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 11), np_163324, 'iterable')
    # Calling iterable(args, kwargs) (line 1055)
    iterable_call_result_163328 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 11), iterable_163325, *[k_163326], **kwargs_163327)
    
    # Applying the 'not' unary operator (line 1055)
    result_not__163329 = python_operator(stypy.reporting.localization.Localization(__file__, 1055, 7), 'not', iterable_call_result_163328)
    
    # Testing the type of an if condition (line 1055)
    if_condition_163330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1055, 4), result_not__163329)
    # Assigning a type to the variable 'if_condition_163330' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 4), 'if_condition_163330', if_condition_163330)
    # SSA begins for if statement (line 1055)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 1056):
    
    # Assigning a List to a Name (line 1056):
    
    # Obtaining an instance of the builtin type 'list' (line 1056)
    list_163331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1056)
    # Adding element type (line 1056)
    # Getting the type of 'k' (line 1056)
    k_163332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1056, 12), list_163331, k_163332)
    
    # Assigning a type to the variable 'k' (line 1056)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 8), 'k', list_163331)
    # SSA join for if statement (line 1055)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 1057):
    
    # Assigning a Subscript to a Name (line 1057):
    
    # Obtaining the type of the subscript
    int_163333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 1057)
    list_163338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1057)
    # Adding element type (line 1057)
    # Getting the type of 'm' (line 1057)
    m_163339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 34), list_163338, m_163339)
    # Adding element type (line 1057)
    # Getting the type of 'axis' (line 1057)
    axis_163340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 34), list_163338, axis_163340)
    
    comprehension_163341 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 18), list_163338)
    # Assigning a type to the variable 't' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 18), 't', comprehension_163341)
    
    # Call to int(...): (line 1057)
    # Processing the call arguments (line 1057)
    # Getting the type of 't' (line 1057)
    t_163335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 22), 't', False)
    # Processing the call keyword arguments (line 1057)
    kwargs_163336 = {}
    # Getting the type of 'int' (line 1057)
    int_163334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 18), 'int', False)
    # Calling int(args, kwargs) (line 1057)
    int_call_result_163337 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 18), int_163334, *[t_163335], **kwargs_163336)
    
    list_163342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 18), list_163342, int_call_result_163337)
    # Obtaining the member '__getitem__' of a type (line 1057)
    getitem___163343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 4), list_163342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1057)
    subscript_call_result_163344 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 4), getitem___163343, int_163333)
    
    # Assigning a type to the variable 'tuple_var_assignment_161906' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'tuple_var_assignment_161906', subscript_call_result_163344)
    
    # Assigning a Subscript to a Name (line 1057):
    
    # Obtaining the type of the subscript
    int_163345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 1057)
    list_163350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1057)
    # Adding element type (line 1057)
    # Getting the type of 'm' (line 1057)
    m_163351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 34), list_163350, m_163351)
    # Adding element type (line 1057)
    # Getting the type of 'axis' (line 1057)
    axis_163352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 34), list_163350, axis_163352)
    
    comprehension_163353 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 18), list_163350)
    # Assigning a type to the variable 't' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 18), 't', comprehension_163353)
    
    # Call to int(...): (line 1057)
    # Processing the call arguments (line 1057)
    # Getting the type of 't' (line 1057)
    t_163347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 22), 't', False)
    # Processing the call keyword arguments (line 1057)
    kwargs_163348 = {}
    # Getting the type of 'int' (line 1057)
    int_163346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 18), 'int', False)
    # Calling int(args, kwargs) (line 1057)
    int_call_result_163349 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 18), int_163346, *[t_163347], **kwargs_163348)
    
    list_163354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 18), list_163354, int_call_result_163349)
    # Obtaining the member '__getitem__' of a type (line 1057)
    getitem___163355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 4), list_163354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1057)
    subscript_call_result_163356 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 4), getitem___163355, int_163345)
    
    # Assigning a type to the variable 'tuple_var_assignment_161907' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'tuple_var_assignment_161907', subscript_call_result_163356)
    
    # Assigning a Name to a Name (line 1057):
    # Getting the type of 'tuple_var_assignment_161906' (line 1057)
    tuple_var_assignment_161906_163357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'tuple_var_assignment_161906')
    # Assigning a type to the variable 'cnt' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'cnt', tuple_var_assignment_161906_163357)
    
    # Assigning a Name to a Name (line 1057):
    # Getting the type of 'tuple_var_assignment_161907' (line 1057)
    tuple_var_assignment_161907_163358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'tuple_var_assignment_161907')
    # Assigning a type to the variable 'iaxis' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 9), 'iaxis', tuple_var_assignment_161907_163358)
    
    
    # Getting the type of 'cnt' (line 1059)
    cnt_163359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 7), 'cnt')
    # Getting the type of 'm' (line 1059)
    m_163360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 14), 'm')
    # Applying the binary operator '!=' (line 1059)
    result_ne_163361 = python_operator(stypy.reporting.localization.Localization(__file__, 1059, 7), '!=', cnt_163359, m_163360)
    
    # Testing the type of an if condition (line 1059)
    if_condition_163362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1059, 4), result_ne_163361)
    # Assigning a type to the variable 'if_condition_163362' (line 1059)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 4), 'if_condition_163362', if_condition_163362)
    # SSA begins for if statement (line 1059)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1060)
    # Processing the call arguments (line 1060)
    str_163364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 25), 'str', 'The order of integration must be integer')
    # Processing the call keyword arguments (line 1060)
    kwargs_163365 = {}
    # Getting the type of 'ValueError' (line 1060)
    ValueError_163363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1060)
    ValueError_call_result_163366 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 14), ValueError_163363, *[str_163364], **kwargs_163365)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1060, 8), ValueError_call_result_163366, 'raise parameter', BaseException)
    # SSA join for if statement (line 1059)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 1061)
    cnt_163367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 7), 'cnt')
    int_163368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 13), 'int')
    # Applying the binary operator '<' (line 1061)
    result_lt_163369 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 7), '<', cnt_163367, int_163368)
    
    # Testing the type of an if condition (line 1061)
    if_condition_163370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1061, 4), result_lt_163369)
    # Assigning a type to the variable 'if_condition_163370' (line 1061)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 4), 'if_condition_163370', if_condition_163370)
    # SSA begins for if statement (line 1061)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1062)
    # Processing the call arguments (line 1062)
    str_163372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 25), 'str', 'The order of integration must be non-negative')
    # Processing the call keyword arguments (line 1062)
    kwargs_163373 = {}
    # Getting the type of 'ValueError' (line 1062)
    ValueError_163371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1062)
    ValueError_call_result_163374 = invoke(stypy.reporting.localization.Localization(__file__, 1062, 14), ValueError_163371, *[str_163372], **kwargs_163373)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1062, 8), ValueError_call_result_163374, 'raise parameter', BaseException)
    # SSA join for if statement (line 1061)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1063)
    # Processing the call arguments (line 1063)
    # Getting the type of 'k' (line 1063)
    k_163376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 11), 'k', False)
    # Processing the call keyword arguments (line 1063)
    kwargs_163377 = {}
    # Getting the type of 'len' (line 1063)
    len_163375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 7), 'len', False)
    # Calling len(args, kwargs) (line 1063)
    len_call_result_163378 = invoke(stypy.reporting.localization.Localization(__file__, 1063, 7), len_163375, *[k_163376], **kwargs_163377)
    
    # Getting the type of 'cnt' (line 1063)
    cnt_163379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 16), 'cnt')
    # Applying the binary operator '>' (line 1063)
    result_gt_163380 = python_operator(stypy.reporting.localization.Localization(__file__, 1063, 7), '>', len_call_result_163378, cnt_163379)
    
    # Testing the type of an if condition (line 1063)
    if_condition_163381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1063, 4), result_gt_163380)
    # Assigning a type to the variable 'if_condition_163381' (line 1063)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 4), 'if_condition_163381', if_condition_163381)
    # SSA begins for if statement (line 1063)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1064)
    # Processing the call arguments (line 1064)
    str_163383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 25), 'str', 'Too many integration constants')
    # Processing the call keyword arguments (line 1064)
    kwargs_163384 = {}
    # Getting the type of 'ValueError' (line 1064)
    ValueError_163382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1064)
    ValueError_call_result_163385 = invoke(stypy.reporting.localization.Localization(__file__, 1064, 14), ValueError_163382, *[str_163383], **kwargs_163384)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1064, 8), ValueError_call_result_163385, 'raise parameter', BaseException)
    # SSA join for if statement (line 1063)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 1065)
    iaxis_163386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 7), 'iaxis')
    # Getting the type of 'axis' (line 1065)
    axis_163387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 16), 'axis')
    # Applying the binary operator '!=' (line 1065)
    result_ne_163388 = python_operator(stypy.reporting.localization.Localization(__file__, 1065, 7), '!=', iaxis_163386, axis_163387)
    
    # Testing the type of an if condition (line 1065)
    if_condition_163389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1065, 4), result_ne_163388)
    # Assigning a type to the variable 'if_condition_163389' (line 1065)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 4), 'if_condition_163389', if_condition_163389)
    # SSA begins for if statement (line 1065)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1066)
    # Processing the call arguments (line 1066)
    str_163391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 1066)
    kwargs_163392 = {}
    # Getting the type of 'ValueError' (line 1066)
    ValueError_163390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1066)
    ValueError_call_result_163393 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 14), ValueError_163390, *[str_163391], **kwargs_163392)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1066, 8), ValueError_call_result_163393, 'raise parameter', BaseException)
    # SSA join for if statement (line 1065)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 1067)
    c_163394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 1067)
    ndim_163395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 12), c_163394, 'ndim')
    # Applying the 'usub' unary operator (line 1067)
    result___neg___163396 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 11), 'usub', ndim_163395)
    
    # Getting the type of 'iaxis' (line 1067)
    iaxis_163397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 22), 'iaxis')
    # Applying the binary operator '<=' (line 1067)
    result_le_163398 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 11), '<=', result___neg___163396, iaxis_163397)
    # Getting the type of 'c' (line 1067)
    c_163399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 1067)
    ndim_163400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 30), c_163399, 'ndim')
    # Applying the binary operator '<' (line 1067)
    result_lt_163401 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 11), '<', iaxis_163397, ndim_163400)
    # Applying the binary operator '&' (line 1067)
    result_and__163402 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 11), '&', result_le_163398, result_lt_163401)
    
    # Applying the 'not' unary operator (line 1067)
    result_not__163403 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 7), 'not', result_and__163402)
    
    # Testing the type of an if condition (line 1067)
    if_condition_163404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1067, 4), result_not__163403)
    # Assigning a type to the variable 'if_condition_163404' (line 1067)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 4), 'if_condition_163404', if_condition_163404)
    # SSA begins for if statement (line 1067)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1068)
    # Processing the call arguments (line 1068)
    str_163406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 1068)
    kwargs_163407 = {}
    # Getting the type of 'ValueError' (line 1068)
    ValueError_163405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1068)
    ValueError_call_result_163408 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 14), ValueError_163405, *[str_163406], **kwargs_163407)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1068, 8), ValueError_call_result_163408, 'raise parameter', BaseException)
    # SSA join for if statement (line 1067)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 1069)
    iaxis_163409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 7), 'iaxis')
    int_163410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 15), 'int')
    # Applying the binary operator '<' (line 1069)
    result_lt_163411 = python_operator(stypy.reporting.localization.Localization(__file__, 1069, 7), '<', iaxis_163409, int_163410)
    
    # Testing the type of an if condition (line 1069)
    if_condition_163412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1069, 4), result_lt_163411)
    # Assigning a type to the variable 'if_condition_163412' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'if_condition_163412', if_condition_163412)
    # SSA begins for if statement (line 1069)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 1070)
    iaxis_163413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 8), 'iaxis')
    # Getting the type of 'c' (line 1070)
    c_163414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 1070)
    ndim_163415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 17), c_163414, 'ndim')
    # Applying the binary operator '+=' (line 1070)
    result_iadd_163416 = python_operator(stypy.reporting.localization.Localization(__file__, 1070, 8), '+=', iaxis_163413, ndim_163415)
    # Assigning a type to the variable 'iaxis' (line 1070)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1070, 8), 'iaxis', result_iadd_163416)
    
    # SSA join for if statement (line 1069)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 1072)
    cnt_163417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 7), 'cnt')
    int_163418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 14), 'int')
    # Applying the binary operator '==' (line 1072)
    result_eq_163419 = python_operator(stypy.reporting.localization.Localization(__file__, 1072, 7), '==', cnt_163417, int_163418)
    
    # Testing the type of an if condition (line 1072)
    if_condition_163420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1072, 4), result_eq_163419)
    # Assigning a type to the variable 'if_condition_163420' (line 1072)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'if_condition_163420', if_condition_163420)
    # SSA begins for if statement (line 1072)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 1073)
    c_163421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1073)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'stypy_return_type', c_163421)
    # SSA join for if statement (line 1072)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1075):
    
    # Assigning a Call to a Name (line 1075):
    
    # Call to rollaxis(...): (line 1075)
    # Processing the call arguments (line 1075)
    # Getting the type of 'c' (line 1075)
    c_163424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 20), 'c', False)
    # Getting the type of 'iaxis' (line 1075)
    iaxis_163425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 1075)
    kwargs_163426 = {}
    # Getting the type of 'np' (line 1075)
    np_163422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1075)
    rollaxis_163423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), np_163422, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1075)
    rollaxis_call_result_163427 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 8), rollaxis_163423, *[c_163424, iaxis_163425], **kwargs_163426)
    
    # Assigning a type to the variable 'c' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 4), 'c', rollaxis_call_result_163427)
    
    # Assigning a BinOp to a Name (line 1076):
    
    # Assigning a BinOp to a Name (line 1076):
    
    # Call to list(...): (line 1076)
    # Processing the call arguments (line 1076)
    # Getting the type of 'k' (line 1076)
    k_163429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 13), 'k', False)
    # Processing the call keyword arguments (line 1076)
    kwargs_163430 = {}
    # Getting the type of 'list' (line 1076)
    list_163428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 8), 'list', False)
    # Calling list(args, kwargs) (line 1076)
    list_call_result_163431 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 8), list_163428, *[k_163429], **kwargs_163430)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1076)
    list_163432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1076)
    # Adding element type (line 1076)
    int_163433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1076, 18), list_163432, int_163433)
    
    # Getting the type of 'cnt' (line 1076)
    cnt_163434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 23), 'cnt')
    
    # Call to len(...): (line 1076)
    # Processing the call arguments (line 1076)
    # Getting the type of 'k' (line 1076)
    k_163436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 33), 'k', False)
    # Processing the call keyword arguments (line 1076)
    kwargs_163437 = {}
    # Getting the type of 'len' (line 1076)
    len_163435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 29), 'len', False)
    # Calling len(args, kwargs) (line 1076)
    len_call_result_163438 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 29), len_163435, *[k_163436], **kwargs_163437)
    
    # Applying the binary operator '-' (line 1076)
    result_sub_163439 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 23), '-', cnt_163434, len_call_result_163438)
    
    # Applying the binary operator '*' (line 1076)
    result_mul_163440 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 18), '*', list_163432, result_sub_163439)
    
    # Applying the binary operator '+' (line 1076)
    result_add_163441 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 8), '+', list_call_result_163431, result_mul_163440)
    
    # Assigning a type to the variable 'k' (line 1076)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1076, 4), 'k', result_add_163441)
    
    
    # Call to range(...): (line 1077)
    # Processing the call arguments (line 1077)
    # Getting the type of 'cnt' (line 1077)
    cnt_163443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 19), 'cnt', False)
    # Processing the call keyword arguments (line 1077)
    kwargs_163444 = {}
    # Getting the type of 'range' (line 1077)
    range_163442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 13), 'range', False)
    # Calling range(args, kwargs) (line 1077)
    range_call_result_163445 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 13), range_163442, *[cnt_163443], **kwargs_163444)
    
    # Testing the type of a for loop iterable (line 1077)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1077, 4), range_call_result_163445)
    # Getting the type of the for loop variable (line 1077)
    for_loop_var_163446 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1077, 4), range_call_result_163445)
    # Assigning a type to the variable 'i' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'i', for_loop_var_163446)
    # SSA begins for a for statement (line 1077)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1078):
    
    # Assigning a Call to a Name (line 1078):
    
    # Call to len(...): (line 1078)
    # Processing the call arguments (line 1078)
    # Getting the type of 'c' (line 1078)
    c_163448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 16), 'c', False)
    # Processing the call keyword arguments (line 1078)
    kwargs_163449 = {}
    # Getting the type of 'len' (line 1078)
    len_163447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 12), 'len', False)
    # Calling len(args, kwargs) (line 1078)
    len_call_result_163450 = invoke(stypy.reporting.localization.Localization(__file__, 1078, 12), len_163447, *[c_163448], **kwargs_163449)
    
    # Assigning a type to the variable 'n' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 8), 'n', len_call_result_163450)
    
    # Getting the type of 'c' (line 1079)
    c_163451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'c')
    # Getting the type of 'scl' (line 1079)
    scl_163452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 13), 'scl')
    # Applying the binary operator '*=' (line 1079)
    result_imul_163453 = python_operator(stypy.reporting.localization.Localization(__file__, 1079, 8), '*=', c_163451, scl_163452)
    # Assigning a type to the variable 'c' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'c', result_imul_163453)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 1080)
    n_163454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 11), 'n')
    int_163455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 16), 'int')
    # Applying the binary operator '==' (line 1080)
    result_eq_163456 = python_operator(stypy.reporting.localization.Localization(__file__, 1080, 11), '==', n_163454, int_163455)
    
    
    # Call to all(...): (line 1080)
    # Processing the call arguments (line 1080)
    
    
    # Obtaining the type of the subscript
    int_163459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 31), 'int')
    # Getting the type of 'c' (line 1080)
    c_163460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1080)
    getitem___163461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1080, 29), c_163460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1080)
    subscript_call_result_163462 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 29), getitem___163461, int_163459)
    
    int_163463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 37), 'int')
    # Applying the binary operator '==' (line 1080)
    result_eq_163464 = python_operator(stypy.reporting.localization.Localization(__file__, 1080, 29), '==', subscript_call_result_163462, int_163463)
    
    # Processing the call keyword arguments (line 1080)
    kwargs_163465 = {}
    # Getting the type of 'np' (line 1080)
    np_163457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 22), 'np', False)
    # Obtaining the member 'all' of a type (line 1080)
    all_163458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1080, 22), np_163457, 'all')
    # Calling all(args, kwargs) (line 1080)
    all_call_result_163466 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 22), all_163458, *[result_eq_163464], **kwargs_163465)
    
    # Applying the binary operator 'and' (line 1080)
    result_and_keyword_163467 = python_operator(stypy.reporting.localization.Localization(__file__, 1080, 11), 'and', result_eq_163456, all_call_result_163466)
    
    # Testing the type of an if condition (line 1080)
    if_condition_163468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1080, 8), result_and_keyword_163467)
    # Assigning a type to the variable 'if_condition_163468' (line 1080)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 8), 'if_condition_163468', if_condition_163468)
    # SSA begins for if statement (line 1080)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 1081)
    c_163469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'c')
    
    # Obtaining the type of the subscript
    int_163470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 14), 'int')
    # Getting the type of 'c' (line 1081)
    c_163471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 1081)
    getitem___163472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 12), c_163471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1081)
    subscript_call_result_163473 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 12), getitem___163472, int_163470)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1081)
    i_163474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 22), 'i')
    # Getting the type of 'k' (line 1081)
    k_163475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 20), 'k')
    # Obtaining the member '__getitem__' of a type (line 1081)
    getitem___163476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 20), k_163475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1081)
    subscript_call_result_163477 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 20), getitem___163476, i_163474)
    
    # Applying the binary operator '+=' (line 1081)
    result_iadd_163478 = python_operator(stypy.reporting.localization.Localization(__file__, 1081, 12), '+=', subscript_call_result_163473, subscript_call_result_163477)
    # Getting the type of 'c' (line 1081)
    c_163479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'c')
    int_163480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 14), 'int')
    # Storing an element on a container (line 1081)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1081, 12), c_163479, (int_163480, result_iadd_163478))
    
    # SSA branch for the else part of an if statement (line 1080)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1083):
    
    # Assigning a Call to a Name (line 1083):
    
    # Call to empty(...): (line 1083)
    # Processing the call arguments (line 1083)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1083)
    tuple_163483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1083)
    # Adding element type (line 1083)
    # Getting the type of 'n' (line 1083)
    n_163484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 28), 'n', False)
    int_163485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 32), 'int')
    # Applying the binary operator '+' (line 1083)
    result_add_163486 = python_operator(stypy.reporting.localization.Localization(__file__, 1083, 28), '+', n_163484, int_163485)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1083, 28), tuple_163483, result_add_163486)
    
    
    # Obtaining the type of the subscript
    int_163487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 46), 'int')
    slice_163488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1083, 38), int_163487, None, None)
    # Getting the type of 'c' (line 1083)
    c_163489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 1083)
    shape_163490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 38), c_163489, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1083)
    getitem___163491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 38), shape_163490, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1083)
    subscript_call_result_163492 = invoke(stypy.reporting.localization.Localization(__file__, 1083, 38), getitem___163491, slice_163488)
    
    # Applying the binary operator '+' (line 1083)
    result_add_163493 = python_operator(stypy.reporting.localization.Localization(__file__, 1083, 27), '+', tuple_163483, subscript_call_result_163492)
    
    # Processing the call keyword arguments (line 1083)
    # Getting the type of 'c' (line 1083)
    c_163494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 57), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1083)
    dtype_163495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 57), c_163494, 'dtype')
    keyword_163496 = dtype_163495
    kwargs_163497 = {'dtype': keyword_163496}
    # Getting the type of 'np' (line 1083)
    np_163481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 1083)
    empty_163482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 18), np_163481, 'empty')
    # Calling empty(args, kwargs) (line 1083)
    empty_call_result_163498 = invoke(stypy.reporting.localization.Localization(__file__, 1083, 18), empty_163482, *[result_add_163493], **kwargs_163497)
    
    # Assigning a type to the variable 'tmp' (line 1083)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1083, 12), 'tmp', empty_call_result_163498)
    
    # Assigning a BinOp to a Subscript (line 1084):
    
    # Assigning a BinOp to a Subscript (line 1084):
    
    # Obtaining the type of the subscript
    int_163499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1084, 23), 'int')
    # Getting the type of 'c' (line 1084)
    c_163500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 1084)
    getitem___163501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 21), c_163500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1084)
    subscript_call_result_163502 = invoke(stypy.reporting.localization.Localization(__file__, 1084, 21), getitem___163501, int_163499)
    
    int_163503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1084, 26), 'int')
    # Applying the binary operator '*' (line 1084)
    result_mul_163504 = python_operator(stypy.reporting.localization.Localization(__file__, 1084, 21), '*', subscript_call_result_163502, int_163503)
    
    # Getting the type of 'tmp' (line 1084)
    tmp_163505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 12), 'tmp')
    int_163506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1084, 16), 'int')
    # Storing an element on a container (line 1084)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 12), tmp_163505, (int_163506, result_mul_163504))
    
    # Assigning a Subscript to a Subscript (line 1085):
    
    # Assigning a Subscript to a Subscript (line 1085):
    
    # Obtaining the type of the subscript
    int_163507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 23), 'int')
    # Getting the type of 'c' (line 1085)
    c_163508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 1085)
    getitem___163509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 21), c_163508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1085)
    subscript_call_result_163510 = invoke(stypy.reporting.localization.Localization(__file__, 1085, 21), getitem___163509, int_163507)
    
    # Getting the type of 'tmp' (line 1085)
    tmp_163511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 12), 'tmp')
    int_163512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 16), 'int')
    # Storing an element on a container (line 1085)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1085, 12), tmp_163511, (int_163512, subscript_call_result_163510))
    
    
    # Getting the type of 'n' (line 1086)
    n_163513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 15), 'n')
    int_163514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 19), 'int')
    # Applying the binary operator '>' (line 1086)
    result_gt_163515 = python_operator(stypy.reporting.localization.Localization(__file__, 1086, 15), '>', n_163513, int_163514)
    
    # Testing the type of an if condition (line 1086)
    if_condition_163516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1086, 12), result_gt_163515)
    # Assigning a type to the variable 'if_condition_163516' (line 1086)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 12), 'if_condition_163516', if_condition_163516)
    # SSA begins for if statement (line 1086)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 1087):
    
    # Assigning a BinOp to a Subscript (line 1087):
    
    # Obtaining the type of the subscript
    int_163517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 27), 'int')
    # Getting the type of 'c' (line 1087)
    c_163518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 25), 'c')
    # Obtaining the member '__getitem__' of a type (line 1087)
    getitem___163519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 25), c_163518, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1087)
    subscript_call_result_163520 = invoke(stypy.reporting.localization.Localization(__file__, 1087, 25), getitem___163519, int_163517)
    
    int_163521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 30), 'int')
    # Applying the binary operator 'div' (line 1087)
    result_div_163522 = python_operator(stypy.reporting.localization.Localization(__file__, 1087, 25), 'div', subscript_call_result_163520, int_163521)
    
    # Getting the type of 'tmp' (line 1087)
    tmp_163523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 16), 'tmp')
    int_163524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 20), 'int')
    # Storing an element on a container (line 1087)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1087, 16), tmp_163523, (int_163524, result_div_163522))
    # SSA join for if statement (line 1086)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 1088)
    # Processing the call arguments (line 1088)
    int_163526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 27), 'int')
    # Getting the type of 'n' (line 1088)
    n_163527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 30), 'n', False)
    # Processing the call keyword arguments (line 1088)
    kwargs_163528 = {}
    # Getting the type of 'range' (line 1088)
    range_163525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 21), 'range', False)
    # Calling range(args, kwargs) (line 1088)
    range_call_result_163529 = invoke(stypy.reporting.localization.Localization(__file__, 1088, 21), range_163525, *[int_163526, n_163527], **kwargs_163528)
    
    # Testing the type of a for loop iterable (line 1088)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1088, 12), range_call_result_163529)
    # Getting the type of the for loop variable (line 1088)
    for_loop_var_163530 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1088, 12), range_call_result_163529)
    # Assigning a type to the variable 'j' (line 1088)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 12), 'j', for_loop_var_163530)
    # SSA begins for a for statement (line 1088)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 1089):
    
    # Assigning a BinOp to a Name (line 1089):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 1089)
    j_163531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 22), 'j')
    # Getting the type of 'c' (line 1089)
    c_163532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 20), 'c')
    # Obtaining the member '__getitem__' of a type (line 1089)
    getitem___163533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 20), c_163532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1089)
    subscript_call_result_163534 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 20), getitem___163533, j_163531)
    
    int_163535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 26), 'int')
    # Getting the type of 'j' (line 1089)
    j_163536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 28), 'j')
    # Applying the binary operator '*' (line 1089)
    result_mul_163537 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 26), '*', int_163535, j_163536)
    
    int_163538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 32), 'int')
    # Applying the binary operator '+' (line 1089)
    result_add_163539 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 26), '+', result_mul_163537, int_163538)
    
    # Applying the binary operator 'div' (line 1089)
    result_div_163540 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 20), 'div', subscript_call_result_163534, result_add_163539)
    
    # Assigning a type to the variable 't' (line 1089)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 16), 't', result_div_163540)
    
    # Assigning a BinOp to a Subscript (line 1090):
    
    # Assigning a BinOp to a Subscript (line 1090):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 1090)
    j_163541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 31), 'j')
    # Getting the type of 'c' (line 1090)
    c_163542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 29), 'c')
    # Obtaining the member '__getitem__' of a type (line 1090)
    getitem___163543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 29), c_163542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1090)
    subscript_call_result_163544 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 29), getitem___163543, j_163541)
    
    int_163545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 35), 'int')
    # Getting the type of 'j' (line 1090)
    j_163546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 38), 'j')
    int_163547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 42), 'int')
    # Applying the binary operator '+' (line 1090)
    result_add_163548 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 38), '+', j_163546, int_163547)
    
    # Applying the binary operator '*' (line 1090)
    result_mul_163549 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 35), '*', int_163545, result_add_163548)
    
    # Applying the binary operator 'div' (line 1090)
    result_div_163550 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 29), 'div', subscript_call_result_163544, result_mul_163549)
    
    # Getting the type of 'tmp' (line 1090)
    tmp_163551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 16), 'tmp')
    # Getting the type of 'j' (line 1090)
    j_163552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 20), 'j')
    int_163553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 24), 'int')
    # Applying the binary operator '+' (line 1090)
    result_add_163554 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 20), '+', j_163552, int_163553)
    
    # Storing an element on a container (line 1090)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1090, 16), tmp_163551, (result_add_163554, result_div_163550))
    
    # Getting the type of 'tmp' (line 1091)
    tmp_163555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 16), 'tmp')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 1091)
    j_163556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 20), 'j')
    int_163557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 24), 'int')
    # Applying the binary operator '-' (line 1091)
    result_sub_163558 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 20), '-', j_163556, int_163557)
    
    # Getting the type of 'tmp' (line 1091)
    tmp_163559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 16), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 1091)
    getitem___163560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 16), tmp_163559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1091)
    subscript_call_result_163561 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 16), getitem___163560, result_sub_163558)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 1091)
    j_163562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 32), 'j')
    # Getting the type of 'c' (line 1091)
    c_163563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 30), 'c')
    # Obtaining the member '__getitem__' of a type (line 1091)
    getitem___163564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 30), c_163563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1091)
    subscript_call_result_163565 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 30), getitem___163564, j_163562)
    
    int_163566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 36), 'int')
    # Getting the type of 'j' (line 1091)
    j_163567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 39), 'j')
    int_163568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 43), 'int')
    # Applying the binary operator '-' (line 1091)
    result_sub_163569 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 39), '-', j_163567, int_163568)
    
    # Applying the binary operator '*' (line 1091)
    result_mul_163570 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 36), '*', int_163566, result_sub_163569)
    
    # Applying the binary operator 'div' (line 1091)
    result_div_163571 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 30), 'div', subscript_call_result_163565, result_mul_163570)
    
    # Applying the binary operator '-=' (line 1091)
    result_isub_163572 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 16), '-=', subscript_call_result_163561, result_div_163571)
    # Getting the type of 'tmp' (line 1091)
    tmp_163573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 16), 'tmp')
    # Getting the type of 'j' (line 1091)
    j_163574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 20), 'j')
    int_163575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 24), 'int')
    # Applying the binary operator '-' (line 1091)
    result_sub_163576 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 20), '-', j_163574, int_163575)
    
    # Storing an element on a container (line 1091)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1091, 16), tmp_163573, (result_sub_163576, result_isub_163572))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tmp' (line 1092)
    tmp_163577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 12), 'tmp')
    
    # Obtaining the type of the subscript
    int_163578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 16), 'int')
    # Getting the type of 'tmp' (line 1092)
    tmp_163579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 12), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 1092)
    getitem___163580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1092, 12), tmp_163579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1092)
    subscript_call_result_163581 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 12), getitem___163580, int_163578)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1092)
    i_163582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 24), 'i')
    # Getting the type of 'k' (line 1092)
    k_163583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 22), 'k')
    # Obtaining the member '__getitem__' of a type (line 1092)
    getitem___163584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1092, 22), k_163583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1092)
    subscript_call_result_163585 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 22), getitem___163584, i_163582)
    
    
    # Call to chebval(...): (line 1092)
    # Processing the call arguments (line 1092)
    # Getting the type of 'lbnd' (line 1092)
    lbnd_163587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 37), 'lbnd', False)
    # Getting the type of 'tmp' (line 1092)
    tmp_163588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 43), 'tmp', False)
    # Processing the call keyword arguments (line 1092)
    kwargs_163589 = {}
    # Getting the type of 'chebval' (line 1092)
    chebval_163586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 29), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1092)
    chebval_call_result_163590 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 29), chebval_163586, *[lbnd_163587, tmp_163588], **kwargs_163589)
    
    # Applying the binary operator '-' (line 1092)
    result_sub_163591 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 22), '-', subscript_call_result_163585, chebval_call_result_163590)
    
    # Applying the binary operator '+=' (line 1092)
    result_iadd_163592 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 12), '+=', subscript_call_result_163581, result_sub_163591)
    # Getting the type of 'tmp' (line 1092)
    tmp_163593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 12), 'tmp')
    int_163594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 16), 'int')
    # Storing an element on a container (line 1092)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1092, 12), tmp_163593, (int_163594, result_iadd_163592))
    
    
    # Assigning a Name to a Name (line 1093):
    
    # Assigning a Name to a Name (line 1093):
    # Getting the type of 'tmp' (line 1093)
    tmp_163595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 16), 'tmp')
    # Assigning a type to the variable 'c' (line 1093)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 12), 'c', tmp_163595)
    # SSA join for if statement (line 1080)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1094):
    
    # Assigning a Call to a Name (line 1094):
    
    # Call to rollaxis(...): (line 1094)
    # Processing the call arguments (line 1094)
    # Getting the type of 'c' (line 1094)
    c_163598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 20), 'c', False)
    int_163599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 23), 'int')
    # Getting the type of 'iaxis' (line 1094)
    iaxis_163600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 26), 'iaxis', False)
    int_163601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 34), 'int')
    # Applying the binary operator '+' (line 1094)
    result_add_163602 = python_operator(stypy.reporting.localization.Localization(__file__, 1094, 26), '+', iaxis_163600, int_163601)
    
    # Processing the call keyword arguments (line 1094)
    kwargs_163603 = {}
    # Getting the type of 'np' (line 1094)
    np_163596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1094)
    rollaxis_163597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 8), np_163596, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1094)
    rollaxis_call_result_163604 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 8), rollaxis_163597, *[c_163598, int_163599, result_add_163602], **kwargs_163603)
    
    # Assigning a type to the variable 'c' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 4), 'c', rollaxis_call_result_163604)
    # Getting the type of 'c' (line 1095)
    c_163605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1095)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 4), 'stypy_return_type', c_163605)
    
    # ################# End of 'chebint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebint' in the type store
    # Getting the type of 'stypy_return_type' (line 967)
    stypy_return_type_163606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163606)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebint'
    return stypy_return_type_163606

# Assigning a type to the variable 'chebint' (line 967)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 0), 'chebint', chebint)

@norecursion
def chebval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1098)
    True_163607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 25), 'True')
    defaults = [True_163607]
    # Create a new context for function 'chebval'
    module_type_store = module_type_store.open_function_context('chebval', 1098, 0, False)
    
    # Passed parameters checking function
    chebval.stypy_localization = localization
    chebval.stypy_type_of_self = None
    chebval.stypy_type_store = module_type_store
    chebval.stypy_function_name = 'chebval'
    chebval.stypy_param_names_list = ['x', 'c', 'tensor']
    chebval.stypy_varargs_param_name = None
    chebval.stypy_kwargs_param_name = None
    chebval.stypy_call_defaults = defaults
    chebval.stypy_call_varargs = varargs
    chebval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebval', ['x', 'c', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebval', localization, ['x', 'c', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebval(...)' code ##################

    str_163608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, (-1)), 'str', '\n    Evaluate a Chebyshev series at points x.\n\n    If `c` is of length `n + 1`, this function returns the value:\n\n    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, algebra_like\n        The shape of the return value is described above.\n\n    See Also\n    --------\n    chebval2d, chebgrid2d, chebval3d, chebgrid3d\n\n    Notes\n    -----\n    The evaluation uses Clenshaw recursion, aka synthetic division.\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a Call to a Name (line 1160):
    
    # Assigning a Call to a Name (line 1160):
    
    # Call to array(...): (line 1160)
    # Processing the call arguments (line 1160)
    # Getting the type of 'c' (line 1160)
    c_163611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 17), 'c', False)
    # Processing the call keyword arguments (line 1160)
    int_163612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 26), 'int')
    keyword_163613 = int_163612
    int_163614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 34), 'int')
    keyword_163615 = int_163614
    kwargs_163616 = {'copy': keyword_163615, 'ndmin': keyword_163613}
    # Getting the type of 'np' (line 1160)
    np_163609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1160)
    array_163610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1160, 8), np_163609, 'array')
    # Calling array(args, kwargs) (line 1160)
    array_call_result_163617 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 8), array_163610, *[c_163611], **kwargs_163616)
    
    # Assigning a type to the variable 'c' (line 1160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1160, 4), 'c', array_call_result_163617)
    
    
    # Getting the type of 'c' (line 1161)
    c_163618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 1161)
    dtype_163619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 7), c_163618, 'dtype')
    # Obtaining the member 'char' of a type (line 1161)
    char_163620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 7), dtype_163619, 'char')
    str_163621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 1161)
    result_contains_163622 = python_operator(stypy.reporting.localization.Localization(__file__, 1161, 7), 'in', char_163620, str_163621)
    
    # Testing the type of an if condition (line 1161)
    if_condition_163623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1161, 4), result_contains_163622)
    # Assigning a type to the variable 'if_condition_163623' (line 1161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 4), 'if_condition_163623', if_condition_163623)
    # SSA begins for if statement (line 1161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1162):
    
    # Assigning a Call to a Name (line 1162):
    
    # Call to astype(...): (line 1162)
    # Processing the call arguments (line 1162)
    # Getting the type of 'np' (line 1162)
    np_163626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 1162)
    double_163627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 21), np_163626, 'double')
    # Processing the call keyword arguments (line 1162)
    kwargs_163628 = {}
    # Getting the type of 'c' (line 1162)
    c_163624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 1162)
    astype_163625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 12), c_163624, 'astype')
    # Calling astype(args, kwargs) (line 1162)
    astype_call_result_163629 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 12), astype_163625, *[double_163627], **kwargs_163628)
    
    # Assigning a type to the variable 'c' (line 1162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 8), 'c', astype_call_result_163629)
    # SSA join for if statement (line 1161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 1163)
    # Processing the call arguments (line 1163)
    # Getting the type of 'x' (line 1163)
    x_163631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 18), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1163)
    tuple_163632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1163)
    # Adding element type (line 1163)
    # Getting the type of 'tuple' (line 1163)
    tuple_163633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 22), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1163, 22), tuple_163632, tuple_163633)
    # Adding element type (line 1163)
    # Getting the type of 'list' (line 1163)
    list_163634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1163, 22), tuple_163632, list_163634)
    
    # Processing the call keyword arguments (line 1163)
    kwargs_163635 = {}
    # Getting the type of 'isinstance' (line 1163)
    isinstance_163630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1163)
    isinstance_call_result_163636 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 7), isinstance_163630, *[x_163631, tuple_163632], **kwargs_163635)
    
    # Testing the type of an if condition (line 1163)
    if_condition_163637 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1163, 4), isinstance_call_result_163636)
    # Assigning a type to the variable 'if_condition_163637' (line 1163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1163, 4), 'if_condition_163637', if_condition_163637)
    # SSA begins for if statement (line 1163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1164):
    
    # Assigning a Call to a Name (line 1164):
    
    # Call to asarray(...): (line 1164)
    # Processing the call arguments (line 1164)
    # Getting the type of 'x' (line 1164)
    x_163640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 23), 'x', False)
    # Processing the call keyword arguments (line 1164)
    kwargs_163641 = {}
    # Getting the type of 'np' (line 1164)
    np_163638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1164)
    asarray_163639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 12), np_163638, 'asarray')
    # Calling asarray(args, kwargs) (line 1164)
    asarray_call_result_163642 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 12), asarray_163639, *[x_163640], **kwargs_163641)
    
    # Assigning a type to the variable 'x' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 8), 'x', asarray_call_result_163642)
    # SSA join for if statement (line 1163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 1165)
    # Processing the call arguments (line 1165)
    # Getting the type of 'x' (line 1165)
    x_163644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 18), 'x', False)
    # Getting the type of 'np' (line 1165)
    np_163645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 1165)
    ndarray_163646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 21), np_163645, 'ndarray')
    # Processing the call keyword arguments (line 1165)
    kwargs_163647 = {}
    # Getting the type of 'isinstance' (line 1165)
    isinstance_163643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1165)
    isinstance_call_result_163648 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 7), isinstance_163643, *[x_163644, ndarray_163646], **kwargs_163647)
    
    # Getting the type of 'tensor' (line 1165)
    tensor_163649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 37), 'tensor')
    # Applying the binary operator 'and' (line 1165)
    result_and_keyword_163650 = python_operator(stypy.reporting.localization.Localization(__file__, 1165, 7), 'and', isinstance_call_result_163648, tensor_163649)
    
    # Testing the type of an if condition (line 1165)
    if_condition_163651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1165, 4), result_and_keyword_163650)
    # Assigning a type to the variable 'if_condition_163651' (line 1165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 4), 'if_condition_163651', if_condition_163651)
    # SSA begins for if statement (line 1165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1166):
    
    # Assigning a Call to a Name (line 1166):
    
    # Call to reshape(...): (line 1166)
    # Processing the call arguments (line 1166)
    # Getting the type of 'c' (line 1166)
    c_163654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 22), 'c', False)
    # Obtaining the member 'shape' of a type (line 1166)
    shape_163655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 22), c_163654, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1166)
    tuple_163656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1166, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1166)
    # Adding element type (line 1166)
    int_163657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1166, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1166, 33), tuple_163656, int_163657)
    
    # Getting the type of 'x' (line 1166)
    x_163658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 37), 'x', False)
    # Obtaining the member 'ndim' of a type (line 1166)
    ndim_163659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 37), x_163658, 'ndim')
    # Applying the binary operator '*' (line 1166)
    result_mul_163660 = python_operator(stypy.reporting.localization.Localization(__file__, 1166, 32), '*', tuple_163656, ndim_163659)
    
    # Applying the binary operator '+' (line 1166)
    result_add_163661 = python_operator(stypy.reporting.localization.Localization(__file__, 1166, 22), '+', shape_163655, result_mul_163660)
    
    # Processing the call keyword arguments (line 1166)
    kwargs_163662 = {}
    # Getting the type of 'c' (line 1166)
    c_163652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 1166)
    reshape_163653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 12), c_163652, 'reshape')
    # Calling reshape(args, kwargs) (line 1166)
    reshape_call_result_163663 = invoke(stypy.reporting.localization.Localization(__file__, 1166, 12), reshape_163653, *[result_add_163661], **kwargs_163662)
    
    # Assigning a type to the variable 'c' (line 1166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1166, 8), 'c', reshape_call_result_163663)
    # SSA join for if statement (line 1165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1168)
    # Processing the call arguments (line 1168)
    # Getting the type of 'c' (line 1168)
    c_163665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 11), 'c', False)
    # Processing the call keyword arguments (line 1168)
    kwargs_163666 = {}
    # Getting the type of 'len' (line 1168)
    len_163664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 7), 'len', False)
    # Calling len(args, kwargs) (line 1168)
    len_call_result_163667 = invoke(stypy.reporting.localization.Localization(__file__, 1168, 7), len_163664, *[c_163665], **kwargs_163666)
    
    int_163668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 17), 'int')
    # Applying the binary operator '==' (line 1168)
    result_eq_163669 = python_operator(stypy.reporting.localization.Localization(__file__, 1168, 7), '==', len_call_result_163667, int_163668)
    
    # Testing the type of an if condition (line 1168)
    if_condition_163670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1168, 4), result_eq_163669)
    # Assigning a type to the variable 'if_condition_163670' (line 1168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1168, 4), 'if_condition_163670', if_condition_163670)
    # SSA begins for if statement (line 1168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1169):
    
    # Assigning a Subscript to a Name (line 1169):
    
    # Obtaining the type of the subscript
    int_163671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, 15), 'int')
    # Getting the type of 'c' (line 1169)
    c_163672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 1169)
    getitem___163673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 13), c_163672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1169)
    subscript_call_result_163674 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 13), getitem___163673, int_163671)
    
    # Assigning a type to the variable 'c0' (line 1169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 8), 'c0', subscript_call_result_163674)
    
    # Assigning a Num to a Name (line 1170):
    
    # Assigning a Num to a Name (line 1170):
    int_163675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1170, 13), 'int')
    # Assigning a type to the variable 'c1' (line 1170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 8), 'c1', int_163675)
    # SSA branch for the else part of an if statement (line 1168)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 1171)
    # Processing the call arguments (line 1171)
    # Getting the type of 'c' (line 1171)
    c_163677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 13), 'c', False)
    # Processing the call keyword arguments (line 1171)
    kwargs_163678 = {}
    # Getting the type of 'len' (line 1171)
    len_163676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 9), 'len', False)
    # Calling len(args, kwargs) (line 1171)
    len_call_result_163679 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 9), len_163676, *[c_163677], **kwargs_163678)
    
    int_163680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 19), 'int')
    # Applying the binary operator '==' (line 1171)
    result_eq_163681 = python_operator(stypy.reporting.localization.Localization(__file__, 1171, 9), '==', len_call_result_163679, int_163680)
    
    # Testing the type of an if condition (line 1171)
    if_condition_163682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1171, 9), result_eq_163681)
    # Assigning a type to the variable 'if_condition_163682' (line 1171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 9), 'if_condition_163682', if_condition_163682)
    # SSA begins for if statement (line 1171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1172):
    
    # Assigning a Subscript to a Name (line 1172):
    
    # Obtaining the type of the subscript
    int_163683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 15), 'int')
    # Getting the type of 'c' (line 1172)
    c_163684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 1172)
    getitem___163685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1172, 13), c_163684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1172)
    subscript_call_result_163686 = invoke(stypy.reporting.localization.Localization(__file__, 1172, 13), getitem___163685, int_163683)
    
    # Assigning a type to the variable 'c0' (line 1172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 8), 'c0', subscript_call_result_163686)
    
    # Assigning a Subscript to a Name (line 1173):
    
    # Assigning a Subscript to a Name (line 1173):
    
    # Obtaining the type of the subscript
    int_163687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1173, 15), 'int')
    # Getting the type of 'c' (line 1173)
    c_163688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 1173)
    getitem___163689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1173, 13), c_163688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1173)
    subscript_call_result_163690 = invoke(stypy.reporting.localization.Localization(__file__, 1173, 13), getitem___163689, int_163687)
    
    # Assigning a type to the variable 'c1' (line 1173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 8), 'c1', subscript_call_result_163690)
    # SSA branch for the else part of an if statement (line 1171)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 1175):
    
    # Assigning a BinOp to a Name (line 1175):
    int_163691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1175, 13), 'int')
    # Getting the type of 'x' (line 1175)
    x_163692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 15), 'x')
    # Applying the binary operator '*' (line 1175)
    result_mul_163693 = python_operator(stypy.reporting.localization.Localization(__file__, 1175, 13), '*', int_163691, x_163692)
    
    # Assigning a type to the variable 'x2' (line 1175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1175, 8), 'x2', result_mul_163693)
    
    # Assigning a Subscript to a Name (line 1176):
    
    # Assigning a Subscript to a Name (line 1176):
    
    # Obtaining the type of the subscript
    int_163694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1176, 15), 'int')
    # Getting the type of 'c' (line 1176)
    c_163695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 1176)
    getitem___163696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1176, 13), c_163695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1176)
    subscript_call_result_163697 = invoke(stypy.reporting.localization.Localization(__file__, 1176, 13), getitem___163696, int_163694)
    
    # Assigning a type to the variable 'c0' (line 1176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 8), 'c0', subscript_call_result_163697)
    
    # Assigning a Subscript to a Name (line 1177):
    
    # Assigning a Subscript to a Name (line 1177):
    
    # Obtaining the type of the subscript
    int_163698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 15), 'int')
    # Getting the type of 'c' (line 1177)
    c_163699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 1177)
    getitem___163700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1177, 13), c_163699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1177)
    subscript_call_result_163701 = invoke(stypy.reporting.localization.Localization(__file__, 1177, 13), getitem___163700, int_163698)
    
    # Assigning a type to the variable 'c1' (line 1177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 8), 'c1', subscript_call_result_163701)
    
    
    # Call to range(...): (line 1178)
    # Processing the call arguments (line 1178)
    int_163703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 23), 'int')
    
    # Call to len(...): (line 1178)
    # Processing the call arguments (line 1178)
    # Getting the type of 'c' (line 1178)
    c_163705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 30), 'c', False)
    # Processing the call keyword arguments (line 1178)
    kwargs_163706 = {}
    # Getting the type of 'len' (line 1178)
    len_163704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 26), 'len', False)
    # Calling len(args, kwargs) (line 1178)
    len_call_result_163707 = invoke(stypy.reporting.localization.Localization(__file__, 1178, 26), len_163704, *[c_163705], **kwargs_163706)
    
    int_163708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 35), 'int')
    # Applying the binary operator '+' (line 1178)
    result_add_163709 = python_operator(stypy.reporting.localization.Localization(__file__, 1178, 26), '+', len_call_result_163707, int_163708)
    
    # Processing the call keyword arguments (line 1178)
    kwargs_163710 = {}
    # Getting the type of 'range' (line 1178)
    range_163702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 17), 'range', False)
    # Calling range(args, kwargs) (line 1178)
    range_call_result_163711 = invoke(stypy.reporting.localization.Localization(__file__, 1178, 17), range_163702, *[int_163703, result_add_163709], **kwargs_163710)
    
    # Testing the type of a for loop iterable (line 1178)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1178, 8), range_call_result_163711)
    # Getting the type of the for loop variable (line 1178)
    for_loop_var_163712 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1178, 8), range_call_result_163711)
    # Assigning a type to the variable 'i' (line 1178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1178, 8), 'i', for_loop_var_163712)
    # SSA begins for a for statement (line 1178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 1179):
    
    # Assigning a Name to a Name (line 1179):
    # Getting the type of 'c0' (line 1179)
    c0_163713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1179, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 1179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1179, 12), 'tmp', c0_163713)
    
    # Assigning a BinOp to a Name (line 1180):
    
    # Assigning a BinOp to a Name (line 1180):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 1180)
    i_163714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 20), 'i')
    # Applying the 'usub' unary operator (line 1180)
    result___neg___163715 = python_operator(stypy.reporting.localization.Localization(__file__, 1180, 19), 'usub', i_163714)
    
    # Getting the type of 'c' (line 1180)
    c_163716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 17), 'c')
    # Obtaining the member '__getitem__' of a type (line 1180)
    getitem___163717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1180, 17), c_163716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1180)
    subscript_call_result_163718 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 17), getitem___163717, result___neg___163715)
    
    # Getting the type of 'c1' (line 1180)
    c1_163719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 25), 'c1')
    # Applying the binary operator '-' (line 1180)
    result_sub_163720 = python_operator(stypy.reporting.localization.Localization(__file__, 1180, 17), '-', subscript_call_result_163718, c1_163719)
    
    # Assigning a type to the variable 'c0' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 12), 'c0', result_sub_163720)
    
    # Assigning a BinOp to a Name (line 1181):
    
    # Assigning a BinOp to a Name (line 1181):
    # Getting the type of 'tmp' (line 1181)
    tmp_163721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 17), 'tmp')
    # Getting the type of 'c1' (line 1181)
    c1_163722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 23), 'c1')
    # Getting the type of 'x2' (line 1181)
    x2_163723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 26), 'x2')
    # Applying the binary operator '*' (line 1181)
    result_mul_163724 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 23), '*', c1_163722, x2_163723)
    
    # Applying the binary operator '+' (line 1181)
    result_add_163725 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 17), '+', tmp_163721, result_mul_163724)
    
    # Assigning a type to the variable 'c1' (line 1181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 12), 'c1', result_add_163725)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1171)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1168)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 1182)
    c0_163726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 11), 'c0')
    # Getting the type of 'c1' (line 1182)
    c1_163727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 16), 'c1')
    # Getting the type of 'x' (line 1182)
    x_163728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 19), 'x')
    # Applying the binary operator '*' (line 1182)
    result_mul_163729 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 16), '*', c1_163727, x_163728)
    
    # Applying the binary operator '+' (line 1182)
    result_add_163730 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 11), '+', c0_163726, result_mul_163729)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 4), 'stypy_return_type', result_add_163730)
    
    # ################# End of 'chebval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebval' in the type store
    # Getting the type of 'stypy_return_type' (line 1098)
    stypy_return_type_163731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163731)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebval'
    return stypy_return_type_163731

# Assigning a type to the variable 'chebval' (line 1098)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1098, 0), 'chebval', chebval)

@norecursion
def chebval2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebval2d'
    module_type_store = module_type_store.open_function_context('chebval2d', 1185, 0, False)
    
    # Passed parameters checking function
    chebval2d.stypy_localization = localization
    chebval2d.stypy_type_of_self = None
    chebval2d.stypy_type_store = module_type_store
    chebval2d.stypy_function_name = 'chebval2d'
    chebval2d.stypy_param_names_list = ['x', 'y', 'c']
    chebval2d.stypy_varargs_param_name = None
    chebval2d.stypy_kwargs_param_name = None
    chebval2d.stypy_call_defaults = defaults
    chebval2d.stypy_call_varargs = varargs
    chebval2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebval2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebval2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebval2d(...)' code ##################

    str_163732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, (-1)), 'str', "\n    Evaluate a 2-D Chebyshev series at points (x, y).\n\n    This function returns the values:\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * T_i(x) * T_j(y)\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` is a 1-D array a one is implicitly appended to its shape to make\n    it 2-D. The shape of the result will be c.shape[2:] + x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and if it isn't an ndarray it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in ``c[i,j]``. If `c` has\n        dimension greater than 2 the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional Chebyshev series at points formed\n        from pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    chebval, chebgrid2d, chebval3d, chebgrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1232):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1232)
    # Processing the call arguments (line 1232)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1232)
    tuple_163735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1232)
    # Adding element type (line 1232)
    # Getting the type of 'x' (line 1232)
    x_163736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 25), tuple_163735, x_163736)
    # Adding element type (line 1232)
    # Getting the type of 'y' (line 1232)
    y_163737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 28), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 25), tuple_163735, y_163737)
    
    # Processing the call keyword arguments (line 1232)
    int_163738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 37), 'int')
    keyword_163739 = int_163738
    kwargs_163740 = {'copy': keyword_163739}
    # Getting the type of 'np' (line 1232)
    np_163733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1232)
    array_163734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1232, 15), np_163733, 'array')
    # Calling array(args, kwargs) (line 1232)
    array_call_result_163741 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 15), array_163734, *[tuple_163735], **kwargs_163740)
    
    # Assigning a type to the variable 'call_assignment_161908' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161908', array_call_result_163741)
    
    # Assigning a Call to a Name (line 1232):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_163744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 8), 'int')
    # Processing the call keyword arguments
    kwargs_163745 = {}
    # Getting the type of 'call_assignment_161908' (line 1232)
    call_assignment_161908_163742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161908', False)
    # Obtaining the member '__getitem__' of a type (line 1232)
    getitem___163743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1232, 8), call_assignment_161908_163742, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163746 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___163743, *[int_163744], **kwargs_163745)
    
    # Assigning a type to the variable 'call_assignment_161909' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161909', getitem___call_result_163746)
    
    # Assigning a Name to a Name (line 1232):
    # Getting the type of 'call_assignment_161909' (line 1232)
    call_assignment_161909_163747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161909')
    # Assigning a type to the variable 'x' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'x', call_assignment_161909_163747)
    
    # Assigning a Call to a Name (line 1232):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_163750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 8), 'int')
    # Processing the call keyword arguments
    kwargs_163751 = {}
    # Getting the type of 'call_assignment_161908' (line 1232)
    call_assignment_161908_163748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161908', False)
    # Obtaining the member '__getitem__' of a type (line 1232)
    getitem___163749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1232, 8), call_assignment_161908_163748, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163752 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___163749, *[int_163750], **kwargs_163751)
    
    # Assigning a type to the variable 'call_assignment_161910' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161910', getitem___call_result_163752)
    
    # Assigning a Name to a Name (line 1232):
    # Getting the type of 'call_assignment_161910' (line 1232)
    call_assignment_161910_163753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'call_assignment_161910')
    # Assigning a type to the variable 'y' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 11), 'y', call_assignment_161910_163753)
    # SSA branch for the except part of a try statement (line 1231)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1231)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1234)
    # Processing the call arguments (line 1234)
    str_163755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 25), 'str', 'x, y are incompatible')
    # Processing the call keyword arguments (line 1234)
    kwargs_163756 = {}
    # Getting the type of 'ValueError' (line 1234)
    ValueError_163754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1234)
    ValueError_call_result_163757 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 14), ValueError_163754, *[str_163755], **kwargs_163756)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1234, 8), ValueError_call_result_163757, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1236):
    
    # Assigning a Call to a Name (line 1236):
    
    # Call to chebval(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'x' (line 1236)
    x_163759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 16), 'x', False)
    # Getting the type of 'c' (line 1236)
    c_163760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 19), 'c', False)
    # Processing the call keyword arguments (line 1236)
    kwargs_163761 = {}
    # Getting the type of 'chebval' (line 1236)
    chebval_163758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1236)
    chebval_call_result_163762 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 8), chebval_163758, *[x_163759, c_163760], **kwargs_163761)
    
    # Assigning a type to the variable 'c' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'c', chebval_call_result_163762)
    
    # Assigning a Call to a Name (line 1237):
    
    # Assigning a Call to a Name (line 1237):
    
    # Call to chebval(...): (line 1237)
    # Processing the call arguments (line 1237)
    # Getting the type of 'y' (line 1237)
    y_163764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 16), 'y', False)
    # Getting the type of 'c' (line 1237)
    c_163765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 19), 'c', False)
    # Processing the call keyword arguments (line 1237)
    # Getting the type of 'False' (line 1237)
    False_163766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 29), 'False', False)
    keyword_163767 = False_163766
    kwargs_163768 = {'tensor': keyword_163767}
    # Getting the type of 'chebval' (line 1237)
    chebval_163763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1237)
    chebval_call_result_163769 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 8), chebval_163763, *[y_163764, c_163765], **kwargs_163768)
    
    # Assigning a type to the variable 'c' (line 1237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 4), 'c', chebval_call_result_163769)
    # Getting the type of 'c' (line 1238)
    c_163770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 4), 'stypy_return_type', c_163770)
    
    # ################# End of 'chebval2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebval2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1185)
    stypy_return_type_163771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163771)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebval2d'
    return stypy_return_type_163771

# Assigning a type to the variable 'chebval2d' (line 1185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 0), 'chebval2d', chebval2d)

@norecursion
def chebgrid2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebgrid2d'
    module_type_store = module_type_store.open_function_context('chebgrid2d', 1241, 0, False)
    
    # Passed parameters checking function
    chebgrid2d.stypy_localization = localization
    chebgrid2d.stypy_type_of_self = None
    chebgrid2d.stypy_type_store = module_type_store
    chebgrid2d.stypy_function_name = 'chebgrid2d'
    chebgrid2d.stypy_param_names_list = ['x', 'y', 'c']
    chebgrid2d.stypy_varargs_param_name = None
    chebgrid2d.stypy_kwargs_param_name = None
    chebgrid2d.stypy_call_defaults = defaults
    chebgrid2d.stypy_call_varargs = varargs
    chebgrid2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebgrid2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebgrid2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebgrid2d(...)' code ##################

    str_163772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1290, (-1)), 'str', "\n    Evaluate a 2-D Chebyshev series on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * T_i(a) * T_j(b),\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape + y.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional Chebyshev series at points in the\n        Cartesian product of `x` and `y`.\n\n    See Also\n    --------\n    chebval, chebval2d, chebval3d, chebgrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1291):
    
    # Assigning a Call to a Name (line 1291):
    
    # Call to chebval(...): (line 1291)
    # Processing the call arguments (line 1291)
    # Getting the type of 'x' (line 1291)
    x_163774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 16), 'x', False)
    # Getting the type of 'c' (line 1291)
    c_163775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 19), 'c', False)
    # Processing the call keyword arguments (line 1291)
    kwargs_163776 = {}
    # Getting the type of 'chebval' (line 1291)
    chebval_163773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1291)
    chebval_call_result_163777 = invoke(stypy.reporting.localization.Localization(__file__, 1291, 8), chebval_163773, *[x_163774, c_163775], **kwargs_163776)
    
    # Assigning a type to the variable 'c' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'c', chebval_call_result_163777)
    
    # Assigning a Call to a Name (line 1292):
    
    # Assigning a Call to a Name (line 1292):
    
    # Call to chebval(...): (line 1292)
    # Processing the call arguments (line 1292)
    # Getting the type of 'y' (line 1292)
    y_163779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 16), 'y', False)
    # Getting the type of 'c' (line 1292)
    c_163780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 19), 'c', False)
    # Processing the call keyword arguments (line 1292)
    kwargs_163781 = {}
    # Getting the type of 'chebval' (line 1292)
    chebval_163778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1292)
    chebval_call_result_163782 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 8), chebval_163778, *[y_163779, c_163780], **kwargs_163781)
    
    # Assigning a type to the variable 'c' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'c', chebval_call_result_163782)
    # Getting the type of 'c' (line 1293)
    c_163783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 4), 'stypy_return_type', c_163783)
    
    # ################# End of 'chebgrid2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebgrid2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1241)
    stypy_return_type_163784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163784)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebgrid2d'
    return stypy_return_type_163784

# Assigning a type to the variable 'chebgrid2d' (line 1241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1241, 0), 'chebgrid2d', chebgrid2d)

@norecursion
def chebval3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebval3d'
    module_type_store = module_type_store.open_function_context('chebval3d', 1296, 0, False)
    
    # Passed parameters checking function
    chebval3d.stypy_localization = localization
    chebval3d.stypy_type_of_self = None
    chebval3d.stypy_type_store = module_type_store
    chebval3d.stypy_function_name = 'chebval3d'
    chebval3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    chebval3d.stypy_varargs_param_name = None
    chebval3d.stypy_kwargs_param_name = None
    chebval3d.stypy_call_defaults = defaults
    chebval3d.stypy_call_varargs = varargs
    chebval3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebval3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebval3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebval3d(...)' code ##################

    str_163785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1343, (-1)), 'str', "\n    Evaluate a 3-D Chebyshev series at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * T_i(x) * T_j(y) * T_k(z)\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimensional polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    chebval, chebval2d, chebgrid2d, chebgrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1345):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1345)
    # Processing the call arguments (line 1345)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1345)
    tuple_163788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1345)
    # Adding element type (line 1345)
    # Getting the type of 'x' (line 1345)
    x_163789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 28), tuple_163788, x_163789)
    # Adding element type (line 1345)
    # Getting the type of 'y' (line 1345)
    y_163790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 28), tuple_163788, y_163790)
    # Adding element type (line 1345)
    # Getting the type of 'z' (line 1345)
    z_163791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 34), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 28), tuple_163788, z_163791)
    
    # Processing the call keyword arguments (line 1345)
    int_163792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 43), 'int')
    keyword_163793 = int_163792
    kwargs_163794 = {'copy': keyword_163793}
    # Getting the type of 'np' (line 1345)
    np_163786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 1345)
    array_163787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 18), np_163786, 'array')
    # Calling array(args, kwargs) (line 1345)
    array_call_result_163795 = invoke(stypy.reporting.localization.Localization(__file__, 1345, 18), array_163787, *[tuple_163788], **kwargs_163794)
    
    # Assigning a type to the variable 'call_assignment_161911' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161911', array_call_result_163795)
    
    # Assigning a Call to a Name (line 1345):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_163798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 8), 'int')
    # Processing the call keyword arguments
    kwargs_163799 = {}
    # Getting the type of 'call_assignment_161911' (line 1345)
    call_assignment_161911_163796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161911', False)
    # Obtaining the member '__getitem__' of a type (line 1345)
    getitem___163797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 8), call_assignment_161911_163796, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163800 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___163797, *[int_163798], **kwargs_163799)
    
    # Assigning a type to the variable 'call_assignment_161912' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161912', getitem___call_result_163800)
    
    # Assigning a Name to a Name (line 1345):
    # Getting the type of 'call_assignment_161912' (line 1345)
    call_assignment_161912_163801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161912')
    # Assigning a type to the variable 'x' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'x', call_assignment_161912_163801)
    
    # Assigning a Call to a Name (line 1345):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_163804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 8), 'int')
    # Processing the call keyword arguments
    kwargs_163805 = {}
    # Getting the type of 'call_assignment_161911' (line 1345)
    call_assignment_161911_163802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161911', False)
    # Obtaining the member '__getitem__' of a type (line 1345)
    getitem___163803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 8), call_assignment_161911_163802, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163806 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___163803, *[int_163804], **kwargs_163805)
    
    # Assigning a type to the variable 'call_assignment_161913' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161913', getitem___call_result_163806)
    
    # Assigning a Name to a Name (line 1345):
    # Getting the type of 'call_assignment_161913' (line 1345)
    call_assignment_161913_163807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161913')
    # Assigning a type to the variable 'y' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 11), 'y', call_assignment_161913_163807)
    
    # Assigning a Call to a Name (line 1345):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_163810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 8), 'int')
    # Processing the call keyword arguments
    kwargs_163811 = {}
    # Getting the type of 'call_assignment_161911' (line 1345)
    call_assignment_161911_163808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161911', False)
    # Obtaining the member '__getitem__' of a type (line 1345)
    getitem___163809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 8), call_assignment_161911_163808, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163812 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___163809, *[int_163810], **kwargs_163811)
    
    # Assigning a type to the variable 'call_assignment_161914' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161914', getitem___call_result_163812)
    
    # Assigning a Name to a Name (line 1345):
    # Getting the type of 'call_assignment_161914' (line 1345)
    call_assignment_161914_163813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'call_assignment_161914')
    # Assigning a type to the variable 'z' (line 1345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 14), 'z', call_assignment_161914_163813)
    # SSA branch for the except part of a try statement (line 1344)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1344)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1347)
    # Processing the call arguments (line 1347)
    str_163815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1347, 25), 'str', 'x, y, z are incompatible')
    # Processing the call keyword arguments (line 1347)
    kwargs_163816 = {}
    # Getting the type of 'ValueError' (line 1347)
    ValueError_163814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1347, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1347)
    ValueError_call_result_163817 = invoke(stypy.reporting.localization.Localization(__file__, 1347, 14), ValueError_163814, *[str_163815], **kwargs_163816)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1347, 8), ValueError_call_result_163817, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1344)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1349):
    
    # Assigning a Call to a Name (line 1349):
    
    # Call to chebval(...): (line 1349)
    # Processing the call arguments (line 1349)
    # Getting the type of 'x' (line 1349)
    x_163819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 16), 'x', False)
    # Getting the type of 'c' (line 1349)
    c_163820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 19), 'c', False)
    # Processing the call keyword arguments (line 1349)
    kwargs_163821 = {}
    # Getting the type of 'chebval' (line 1349)
    chebval_163818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1349)
    chebval_call_result_163822 = invoke(stypy.reporting.localization.Localization(__file__, 1349, 8), chebval_163818, *[x_163819, c_163820], **kwargs_163821)
    
    # Assigning a type to the variable 'c' (line 1349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1349, 4), 'c', chebval_call_result_163822)
    
    # Assigning a Call to a Name (line 1350):
    
    # Assigning a Call to a Name (line 1350):
    
    # Call to chebval(...): (line 1350)
    # Processing the call arguments (line 1350)
    # Getting the type of 'y' (line 1350)
    y_163824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 16), 'y', False)
    # Getting the type of 'c' (line 1350)
    c_163825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 19), 'c', False)
    # Processing the call keyword arguments (line 1350)
    # Getting the type of 'False' (line 1350)
    False_163826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 29), 'False', False)
    keyword_163827 = False_163826
    kwargs_163828 = {'tensor': keyword_163827}
    # Getting the type of 'chebval' (line 1350)
    chebval_163823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1350)
    chebval_call_result_163829 = invoke(stypy.reporting.localization.Localization(__file__, 1350, 8), chebval_163823, *[y_163824, c_163825], **kwargs_163828)
    
    # Assigning a type to the variable 'c' (line 1350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1350, 4), 'c', chebval_call_result_163829)
    
    # Assigning a Call to a Name (line 1351):
    
    # Assigning a Call to a Name (line 1351):
    
    # Call to chebval(...): (line 1351)
    # Processing the call arguments (line 1351)
    # Getting the type of 'z' (line 1351)
    z_163831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 16), 'z', False)
    # Getting the type of 'c' (line 1351)
    c_163832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 19), 'c', False)
    # Processing the call keyword arguments (line 1351)
    # Getting the type of 'False' (line 1351)
    False_163833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 29), 'False', False)
    keyword_163834 = False_163833
    kwargs_163835 = {'tensor': keyword_163834}
    # Getting the type of 'chebval' (line 1351)
    chebval_163830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1351)
    chebval_call_result_163836 = invoke(stypy.reporting.localization.Localization(__file__, 1351, 8), chebval_163830, *[z_163831, c_163832], **kwargs_163835)
    
    # Assigning a type to the variable 'c' (line 1351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1351, 4), 'c', chebval_call_result_163836)
    # Getting the type of 'c' (line 1352)
    c_163837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 4), 'stypy_return_type', c_163837)
    
    # ################# End of 'chebval3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebval3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1296)
    stypy_return_type_163838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163838)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebval3d'
    return stypy_return_type_163838

# Assigning a type to the variable 'chebval3d' (line 1296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1296, 0), 'chebval3d', chebval3d)

@norecursion
def chebgrid3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebgrid3d'
    module_type_store = module_type_store.open_function_context('chebgrid3d', 1355, 0, False)
    
    # Passed parameters checking function
    chebgrid3d.stypy_localization = localization
    chebgrid3d.stypy_type_of_self = None
    chebgrid3d.stypy_type_store = module_type_store
    chebgrid3d.stypy_function_name = 'chebgrid3d'
    chebgrid3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    chebgrid3d.stypy_varargs_param_name = None
    chebgrid3d.stypy_kwargs_param_name = None
    chebgrid3d.stypy_call_defaults = defaults
    chebgrid3d.stypy_call_varargs = varargs
    chebgrid3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebgrid3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebgrid3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebgrid3d(...)' code ##################

    str_163839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1407, (-1)), 'str', "\n    Evaluate a 3-D Chebyshev series on the Cartesian product of x, y, and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * T_i(a) * T_j(b) * T_k(c)\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    chebval, chebval2d, chebgrid2d, chebval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1408):
    
    # Assigning a Call to a Name (line 1408):
    
    # Call to chebval(...): (line 1408)
    # Processing the call arguments (line 1408)
    # Getting the type of 'x' (line 1408)
    x_163841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 16), 'x', False)
    # Getting the type of 'c' (line 1408)
    c_163842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 19), 'c', False)
    # Processing the call keyword arguments (line 1408)
    kwargs_163843 = {}
    # Getting the type of 'chebval' (line 1408)
    chebval_163840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1408)
    chebval_call_result_163844 = invoke(stypy.reporting.localization.Localization(__file__, 1408, 8), chebval_163840, *[x_163841, c_163842], **kwargs_163843)
    
    # Assigning a type to the variable 'c' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 4), 'c', chebval_call_result_163844)
    
    # Assigning a Call to a Name (line 1409):
    
    # Assigning a Call to a Name (line 1409):
    
    # Call to chebval(...): (line 1409)
    # Processing the call arguments (line 1409)
    # Getting the type of 'y' (line 1409)
    y_163846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 16), 'y', False)
    # Getting the type of 'c' (line 1409)
    c_163847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 19), 'c', False)
    # Processing the call keyword arguments (line 1409)
    kwargs_163848 = {}
    # Getting the type of 'chebval' (line 1409)
    chebval_163845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1409)
    chebval_call_result_163849 = invoke(stypy.reporting.localization.Localization(__file__, 1409, 8), chebval_163845, *[y_163846, c_163847], **kwargs_163848)
    
    # Assigning a type to the variable 'c' (line 1409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1409, 4), 'c', chebval_call_result_163849)
    
    # Assigning a Call to a Name (line 1410):
    
    # Assigning a Call to a Name (line 1410):
    
    # Call to chebval(...): (line 1410)
    # Processing the call arguments (line 1410)
    # Getting the type of 'z' (line 1410)
    z_163851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 16), 'z', False)
    # Getting the type of 'c' (line 1410)
    c_163852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 19), 'c', False)
    # Processing the call keyword arguments (line 1410)
    kwargs_163853 = {}
    # Getting the type of 'chebval' (line 1410)
    chebval_163850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 8), 'chebval', False)
    # Calling chebval(args, kwargs) (line 1410)
    chebval_call_result_163854 = invoke(stypy.reporting.localization.Localization(__file__, 1410, 8), chebval_163850, *[z_163851, c_163852], **kwargs_163853)
    
    # Assigning a type to the variable 'c' (line 1410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1410, 4), 'c', chebval_call_result_163854)
    # Getting the type of 'c' (line 1411)
    c_163855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1411, 4), 'stypy_return_type', c_163855)
    
    # ################# End of 'chebgrid3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebgrid3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1355)
    stypy_return_type_163856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163856)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebgrid3d'
    return stypy_return_type_163856

# Assigning a type to the variable 'chebgrid3d' (line 1355)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 0), 'chebgrid3d', chebgrid3d)

@norecursion
def chebvander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebvander'
    module_type_store = module_type_store.open_function_context('chebvander', 1414, 0, False)
    
    # Passed parameters checking function
    chebvander.stypy_localization = localization
    chebvander.stypy_type_of_self = None
    chebvander.stypy_type_store = module_type_store
    chebvander.stypy_function_name = 'chebvander'
    chebvander.stypy_param_names_list = ['x', 'deg']
    chebvander.stypy_varargs_param_name = None
    chebvander.stypy_kwargs_param_name = None
    chebvander.stypy_call_defaults = defaults
    chebvander.stypy_call_varargs = varargs
    chebvander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebvander', ['x', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebvander', localization, ['x', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebvander(...)' code ##################

    str_163857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1448, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degree.\n\n    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points\n    `x`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = T_i(x),\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the degree of the Chebyshev polynomial.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and\n    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of Chebyshev series of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray\n        The pseudo Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where The last index is the degree of the\n        corresponding Chebyshev polynomial.  The dtype will be the same as\n        the converted `x`.\n\n    ')
    
    # Assigning a Call to a Name (line 1449):
    
    # Assigning a Call to a Name (line 1449):
    
    # Call to int(...): (line 1449)
    # Processing the call arguments (line 1449)
    # Getting the type of 'deg' (line 1449)
    deg_163859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 15), 'deg', False)
    # Processing the call keyword arguments (line 1449)
    kwargs_163860 = {}
    # Getting the type of 'int' (line 1449)
    int_163858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 11), 'int', False)
    # Calling int(args, kwargs) (line 1449)
    int_call_result_163861 = invoke(stypy.reporting.localization.Localization(__file__, 1449, 11), int_163858, *[deg_163859], **kwargs_163860)
    
    # Assigning a type to the variable 'ideg' (line 1449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1449, 4), 'ideg', int_call_result_163861)
    
    
    # Getting the type of 'ideg' (line 1450)
    ideg_163862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 7), 'ideg')
    # Getting the type of 'deg' (line 1450)
    deg_163863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 15), 'deg')
    # Applying the binary operator '!=' (line 1450)
    result_ne_163864 = python_operator(stypy.reporting.localization.Localization(__file__, 1450, 7), '!=', ideg_163862, deg_163863)
    
    # Testing the type of an if condition (line 1450)
    if_condition_163865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1450, 4), result_ne_163864)
    # Assigning a type to the variable 'if_condition_163865' (line 1450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1450, 4), 'if_condition_163865', if_condition_163865)
    # SSA begins for if statement (line 1450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1451)
    # Processing the call arguments (line 1451)
    str_163867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 25), 'str', 'deg must be integer')
    # Processing the call keyword arguments (line 1451)
    kwargs_163868 = {}
    # Getting the type of 'ValueError' (line 1451)
    ValueError_163866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1451)
    ValueError_call_result_163869 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 14), ValueError_163866, *[str_163867], **kwargs_163868)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1451, 8), ValueError_call_result_163869, 'raise parameter', BaseException)
    # SSA join for if statement (line 1450)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ideg' (line 1452)
    ideg_163870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1452, 7), 'ideg')
    int_163871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1452, 14), 'int')
    # Applying the binary operator '<' (line 1452)
    result_lt_163872 = python_operator(stypy.reporting.localization.Localization(__file__, 1452, 7), '<', ideg_163870, int_163871)
    
    # Testing the type of an if condition (line 1452)
    if_condition_163873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1452, 4), result_lt_163872)
    # Assigning a type to the variable 'if_condition_163873' (line 1452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1452, 4), 'if_condition_163873', if_condition_163873)
    # SSA begins for if statement (line 1452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1453)
    # Processing the call arguments (line 1453)
    str_163875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1453, 25), 'str', 'deg must be non-negative')
    # Processing the call keyword arguments (line 1453)
    kwargs_163876 = {}
    # Getting the type of 'ValueError' (line 1453)
    ValueError_163874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1453, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1453)
    ValueError_call_result_163877 = invoke(stypy.reporting.localization.Localization(__file__, 1453, 14), ValueError_163874, *[str_163875], **kwargs_163876)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1453, 8), ValueError_call_result_163877, 'raise parameter', BaseException)
    # SSA join for if statement (line 1452)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1455):
    
    # Assigning a BinOp to a Name (line 1455):
    
    # Call to array(...): (line 1455)
    # Processing the call arguments (line 1455)
    # Getting the type of 'x' (line 1455)
    x_163880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 17), 'x', False)
    # Processing the call keyword arguments (line 1455)
    int_163881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1455, 25), 'int')
    keyword_163882 = int_163881
    int_163883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1455, 34), 'int')
    keyword_163884 = int_163883
    kwargs_163885 = {'copy': keyword_163882, 'ndmin': keyword_163884}
    # Getting the type of 'np' (line 1455)
    np_163878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1455)
    array_163879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1455, 8), np_163878, 'array')
    # Calling array(args, kwargs) (line 1455)
    array_call_result_163886 = invoke(stypy.reporting.localization.Localization(__file__, 1455, 8), array_163879, *[x_163880], **kwargs_163885)
    
    float_163887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1455, 39), 'float')
    # Applying the binary operator '+' (line 1455)
    result_add_163888 = python_operator(stypy.reporting.localization.Localization(__file__, 1455, 8), '+', array_call_result_163886, float_163887)
    
    # Assigning a type to the variable 'x' (line 1455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1455, 4), 'x', result_add_163888)
    
    # Assigning a BinOp to a Name (line 1456):
    
    # Assigning a BinOp to a Name (line 1456):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1456)
    tuple_163889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1456, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1456)
    # Adding element type (line 1456)
    # Getting the type of 'ideg' (line 1456)
    ideg_163890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1456, 12), 'ideg')
    int_163891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1456, 19), 'int')
    # Applying the binary operator '+' (line 1456)
    result_add_163892 = python_operator(stypy.reporting.localization.Localization(__file__, 1456, 12), '+', ideg_163890, int_163891)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1456, 12), tuple_163889, result_add_163892)
    
    # Getting the type of 'x' (line 1456)
    x_163893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1456, 25), 'x')
    # Obtaining the member 'shape' of a type (line 1456)
    shape_163894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1456, 25), x_163893, 'shape')
    # Applying the binary operator '+' (line 1456)
    result_add_163895 = python_operator(stypy.reporting.localization.Localization(__file__, 1456, 11), '+', tuple_163889, shape_163894)
    
    # Assigning a type to the variable 'dims' (line 1456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1456, 4), 'dims', result_add_163895)
    
    # Assigning a Attribute to a Name (line 1457):
    
    # Assigning a Attribute to a Name (line 1457):
    # Getting the type of 'x' (line 1457)
    x_163896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1457, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 1457)
    dtype_163897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1457, 11), x_163896, 'dtype')
    # Assigning a type to the variable 'dtyp' (line 1457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1457, 4), 'dtyp', dtype_163897)
    
    # Assigning a Call to a Name (line 1458):
    
    # Assigning a Call to a Name (line 1458):
    
    # Call to empty(...): (line 1458)
    # Processing the call arguments (line 1458)
    # Getting the type of 'dims' (line 1458)
    dims_163900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 17), 'dims', False)
    # Processing the call keyword arguments (line 1458)
    # Getting the type of 'dtyp' (line 1458)
    dtyp_163901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 29), 'dtyp', False)
    keyword_163902 = dtyp_163901
    kwargs_163903 = {'dtype': keyword_163902}
    # Getting the type of 'np' (line 1458)
    np_163898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1458)
    empty_163899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1458, 8), np_163898, 'empty')
    # Calling empty(args, kwargs) (line 1458)
    empty_call_result_163904 = invoke(stypy.reporting.localization.Localization(__file__, 1458, 8), empty_163899, *[dims_163900], **kwargs_163903)
    
    # Assigning a type to the variable 'v' (line 1458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1458, 4), 'v', empty_call_result_163904)
    
    # Assigning a BinOp to a Subscript (line 1460):
    
    # Assigning a BinOp to a Subscript (line 1460):
    # Getting the type of 'x' (line 1460)
    x_163905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1460, 11), 'x')
    int_163906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1460, 13), 'int')
    # Applying the binary operator '*' (line 1460)
    result_mul_163907 = python_operator(stypy.reporting.localization.Localization(__file__, 1460, 11), '*', x_163905, int_163906)
    
    int_163908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1460, 17), 'int')
    # Applying the binary operator '+' (line 1460)
    result_add_163909 = python_operator(stypy.reporting.localization.Localization(__file__, 1460, 11), '+', result_mul_163907, int_163908)
    
    # Getting the type of 'v' (line 1460)
    v_163910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1460, 4), 'v')
    int_163911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1460, 6), 'int')
    # Storing an element on a container (line 1460)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1460, 4), v_163910, (int_163911, result_add_163909))
    
    
    # Getting the type of 'ideg' (line 1461)
    ideg_163912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 7), 'ideg')
    int_163913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1461, 14), 'int')
    # Applying the binary operator '>' (line 1461)
    result_gt_163914 = python_operator(stypy.reporting.localization.Localization(__file__, 1461, 7), '>', ideg_163912, int_163913)
    
    # Testing the type of an if condition (line 1461)
    if_condition_163915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1461, 4), result_gt_163914)
    # Assigning a type to the variable 'if_condition_163915' (line 1461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1461, 4), 'if_condition_163915', if_condition_163915)
    # SSA begins for if statement (line 1461)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1462):
    
    # Assigning a BinOp to a Name (line 1462):
    int_163916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1462, 13), 'int')
    # Getting the type of 'x' (line 1462)
    x_163917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1462, 15), 'x')
    # Applying the binary operator '*' (line 1462)
    result_mul_163918 = python_operator(stypy.reporting.localization.Localization(__file__, 1462, 13), '*', int_163916, x_163917)
    
    # Assigning a type to the variable 'x2' (line 1462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1462, 8), 'x2', result_mul_163918)
    
    # Assigning a Name to a Subscript (line 1463):
    
    # Assigning a Name to a Subscript (line 1463):
    # Getting the type of 'x' (line 1463)
    x_163919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 15), 'x')
    # Getting the type of 'v' (line 1463)
    v_163920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 8), 'v')
    int_163921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1463, 10), 'int')
    # Storing an element on a container (line 1463)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1463, 8), v_163920, (int_163921, x_163919))
    
    
    # Call to range(...): (line 1464)
    # Processing the call arguments (line 1464)
    int_163923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1464, 23), 'int')
    # Getting the type of 'ideg' (line 1464)
    ideg_163924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1464, 26), 'ideg', False)
    int_163925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1464, 33), 'int')
    # Applying the binary operator '+' (line 1464)
    result_add_163926 = python_operator(stypy.reporting.localization.Localization(__file__, 1464, 26), '+', ideg_163924, int_163925)
    
    # Processing the call keyword arguments (line 1464)
    kwargs_163927 = {}
    # Getting the type of 'range' (line 1464)
    range_163922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1464, 17), 'range', False)
    # Calling range(args, kwargs) (line 1464)
    range_call_result_163928 = invoke(stypy.reporting.localization.Localization(__file__, 1464, 17), range_163922, *[int_163923, result_add_163926], **kwargs_163927)
    
    # Testing the type of a for loop iterable (line 1464)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1464, 8), range_call_result_163928)
    # Getting the type of the for loop variable (line 1464)
    for_loop_var_163929 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1464, 8), range_call_result_163928)
    # Assigning a type to the variable 'i' (line 1464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1464, 8), 'i', for_loop_var_163929)
    # SSA begins for a for statement (line 1464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 1465):
    
    # Assigning a BinOp to a Subscript (line 1465):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1465)
    i_163930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 21), 'i')
    int_163931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1465, 23), 'int')
    # Applying the binary operator '-' (line 1465)
    result_sub_163932 = python_operator(stypy.reporting.localization.Localization(__file__, 1465, 21), '-', i_163930, int_163931)
    
    # Getting the type of 'v' (line 1465)
    v_163933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 19), 'v')
    # Obtaining the member '__getitem__' of a type (line 1465)
    getitem___163934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1465, 19), v_163933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1465)
    subscript_call_result_163935 = invoke(stypy.reporting.localization.Localization(__file__, 1465, 19), getitem___163934, result_sub_163932)
    
    # Getting the type of 'x2' (line 1465)
    x2_163936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 26), 'x2')
    # Applying the binary operator '*' (line 1465)
    result_mul_163937 = python_operator(stypy.reporting.localization.Localization(__file__, 1465, 19), '*', subscript_call_result_163935, x2_163936)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1465)
    i_163938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 33), 'i')
    int_163939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1465, 35), 'int')
    # Applying the binary operator '-' (line 1465)
    result_sub_163940 = python_operator(stypy.reporting.localization.Localization(__file__, 1465, 33), '-', i_163938, int_163939)
    
    # Getting the type of 'v' (line 1465)
    v_163941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 31), 'v')
    # Obtaining the member '__getitem__' of a type (line 1465)
    getitem___163942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1465, 31), v_163941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1465)
    subscript_call_result_163943 = invoke(stypy.reporting.localization.Localization(__file__, 1465, 31), getitem___163942, result_sub_163940)
    
    # Applying the binary operator '-' (line 1465)
    result_sub_163944 = python_operator(stypy.reporting.localization.Localization(__file__, 1465, 19), '-', result_mul_163937, subscript_call_result_163943)
    
    # Getting the type of 'v' (line 1465)
    v_163945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 12), 'v')
    # Getting the type of 'i' (line 1465)
    i_163946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 14), 'i')
    # Storing an element on a container (line 1465)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1465, 12), v_163945, (i_163946, result_sub_163944))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1461)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rollaxis(...): (line 1466)
    # Processing the call arguments (line 1466)
    # Getting the type of 'v' (line 1466)
    v_163949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1466, 23), 'v', False)
    int_163950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1466, 26), 'int')
    # Getting the type of 'v' (line 1466)
    v_163951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1466, 29), 'v', False)
    # Obtaining the member 'ndim' of a type (line 1466)
    ndim_163952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1466, 29), v_163951, 'ndim')
    # Processing the call keyword arguments (line 1466)
    kwargs_163953 = {}
    # Getting the type of 'np' (line 1466)
    np_163947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1466, 11), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1466)
    rollaxis_163948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1466, 11), np_163947, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1466)
    rollaxis_call_result_163954 = invoke(stypy.reporting.localization.Localization(__file__, 1466, 11), rollaxis_163948, *[v_163949, int_163950, ndim_163952], **kwargs_163953)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1466, 4), 'stypy_return_type', rollaxis_call_result_163954)
    
    # ################# End of 'chebvander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebvander' in the type store
    # Getting the type of 'stypy_return_type' (line 1414)
    stypy_return_type_163955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163955)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebvander'
    return stypy_return_type_163955

# Assigning a type to the variable 'chebvander' (line 1414)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 0), 'chebvander', chebvander)

@norecursion
def chebvander2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebvander2d'
    module_type_store = module_type_store.open_function_context('chebvander2d', 1469, 0, False)
    
    # Passed parameters checking function
    chebvander2d.stypy_localization = localization
    chebvander2d.stypy_type_of_self = None
    chebvander2d.stypy_type_store = module_type_store
    chebvander2d.stypy_function_name = 'chebvander2d'
    chebvander2d.stypy_param_names_list = ['x', 'y', 'deg']
    chebvander2d.stypy_varargs_param_name = None
    chebvander2d.stypy_kwargs_param_name = None
    chebvander2d.stypy_call_defaults = defaults
    chebvander2d.stypy_call_varargs = varargs
    chebvander2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebvander2d', ['x', 'y', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebvander2d', localization, ['x', 'y', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebvander2d(...)' code ##################

    str_163956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1518, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., deg[1]*i + j] = T_i(x) * T_j(y),\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the degrees of\n    the Chebyshev polynomials.\n\n    If ``V = chebvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``chebval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D Chebyshev\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to\n        1-D arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    chebvander, chebvander3d. chebval2d, chebval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1519):
    
    # Assigning a ListComp to a Name (line 1519):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1519)
    deg_163961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 28), 'deg')
    comprehension_163962 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1519, 12), deg_163961)
    # Assigning a type to the variable 'd' (line 1519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1519, 12), 'd', comprehension_163962)
    
    # Call to int(...): (line 1519)
    # Processing the call arguments (line 1519)
    # Getting the type of 'd' (line 1519)
    d_163958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 16), 'd', False)
    # Processing the call keyword arguments (line 1519)
    kwargs_163959 = {}
    # Getting the type of 'int' (line 1519)
    int_163957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 12), 'int', False)
    # Calling int(args, kwargs) (line 1519)
    int_call_result_163960 = invoke(stypy.reporting.localization.Localization(__file__, 1519, 12), int_163957, *[d_163958], **kwargs_163959)
    
    list_163963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1519, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1519, 12), list_163963, int_call_result_163960)
    # Assigning a type to the variable 'ideg' (line 1519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1519, 4), 'ideg', list_163963)
    
    # Assigning a ListComp to a Name (line 1520):
    
    # Assigning a ListComp to a Name (line 1520):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1520)
    # Processing the call arguments (line 1520)
    # Getting the type of 'ideg' (line 1520)
    ideg_163972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1520)
    deg_163973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 59), 'deg', False)
    # Processing the call keyword arguments (line 1520)
    kwargs_163974 = {}
    # Getting the type of 'zip' (line 1520)
    zip_163971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1520)
    zip_call_result_163975 = invoke(stypy.reporting.localization.Localization(__file__, 1520, 49), zip_163971, *[ideg_163972, deg_163973], **kwargs_163974)
    
    comprehension_163976 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1520, 16), zip_call_result_163975)
    # Assigning a type to the variable 'id' (line 1520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1520, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1520, 16), comprehension_163976))
    # Assigning a type to the variable 'd' (line 1520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1520, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1520, 16), comprehension_163976))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1520)
    id_163964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 16), 'id')
    # Getting the type of 'd' (line 1520)
    d_163965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 22), 'd')
    # Applying the binary operator '==' (line 1520)
    result_eq_163966 = python_operator(stypy.reporting.localization.Localization(__file__, 1520, 16), '==', id_163964, d_163965)
    
    
    # Getting the type of 'id' (line 1520)
    id_163967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 28), 'id')
    int_163968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1520, 34), 'int')
    # Applying the binary operator '>=' (line 1520)
    result_ge_163969 = python_operator(stypy.reporting.localization.Localization(__file__, 1520, 28), '>=', id_163967, int_163968)
    
    # Applying the binary operator 'and' (line 1520)
    result_and_keyword_163970 = python_operator(stypy.reporting.localization.Localization(__file__, 1520, 16), 'and', result_eq_163966, result_ge_163969)
    
    list_163977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1520, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1520, 16), list_163977, result_and_keyword_163970)
    # Assigning a type to the variable 'is_valid' (line 1520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1520, 4), 'is_valid', list_163977)
    
    
    # Getting the type of 'is_valid' (line 1521)
    is_valid_163978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1521)
    list_163979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1521, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1521)
    # Adding element type (line 1521)
    int_163980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1521, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1521, 19), list_163979, int_163980)
    # Adding element type (line 1521)
    int_163981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1521, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1521, 19), list_163979, int_163981)
    
    # Applying the binary operator '!=' (line 1521)
    result_ne_163982 = python_operator(stypy.reporting.localization.Localization(__file__, 1521, 7), '!=', is_valid_163978, list_163979)
    
    # Testing the type of an if condition (line 1521)
    if_condition_163983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1521, 4), result_ne_163982)
    # Assigning a type to the variable 'if_condition_163983' (line 1521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1521, 4), 'if_condition_163983', if_condition_163983)
    # SSA begins for if statement (line 1521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1522)
    # Processing the call arguments (line 1522)
    str_163985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1522, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1522)
    kwargs_163986 = {}
    # Getting the type of 'ValueError' (line 1522)
    ValueError_163984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1522)
    ValueError_call_result_163987 = invoke(stypy.reporting.localization.Localization(__file__, 1522, 14), ValueError_163984, *[str_163985], **kwargs_163986)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1522, 8), ValueError_call_result_163987, 'raise parameter', BaseException)
    # SSA join for if statement (line 1521)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1523):
    
    # Assigning a Subscript to a Name (line 1523):
    
    # Obtaining the type of the subscript
    int_163988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1523, 4), 'int')
    # Getting the type of 'ideg' (line 1523)
    ideg_163989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1523)
    getitem___163990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1523, 4), ideg_163989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1523)
    subscript_call_result_163991 = invoke(stypy.reporting.localization.Localization(__file__, 1523, 4), getitem___163990, int_163988)
    
    # Assigning a type to the variable 'tuple_var_assignment_161915' (line 1523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'tuple_var_assignment_161915', subscript_call_result_163991)
    
    # Assigning a Subscript to a Name (line 1523):
    
    # Obtaining the type of the subscript
    int_163992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1523, 4), 'int')
    # Getting the type of 'ideg' (line 1523)
    ideg_163993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1523)
    getitem___163994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1523, 4), ideg_163993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1523)
    subscript_call_result_163995 = invoke(stypy.reporting.localization.Localization(__file__, 1523, 4), getitem___163994, int_163992)
    
    # Assigning a type to the variable 'tuple_var_assignment_161916' (line 1523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'tuple_var_assignment_161916', subscript_call_result_163995)
    
    # Assigning a Name to a Name (line 1523):
    # Getting the type of 'tuple_var_assignment_161915' (line 1523)
    tuple_var_assignment_161915_163996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'tuple_var_assignment_161915')
    # Assigning a type to the variable 'degx' (line 1523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'degx', tuple_var_assignment_161915_163996)
    
    # Assigning a Name to a Name (line 1523):
    # Getting the type of 'tuple_var_assignment_161916' (line 1523)
    tuple_var_assignment_161916_163997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'tuple_var_assignment_161916')
    # Assigning a type to the variable 'degy' (line 1523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1523, 10), 'degy', tuple_var_assignment_161916_163997)
    
    # Assigning a BinOp to a Tuple (line 1524):
    
    # Assigning a Subscript to a Name (line 1524):
    
    # Obtaining the type of the subscript
    int_163998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 4), 'int')
    
    # Call to array(...): (line 1524)
    # Processing the call arguments (line 1524)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1524)
    tuple_164001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1524)
    # Adding element type (line 1524)
    # Getting the type of 'x' (line 1524)
    x_164002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1524, 21), tuple_164001, x_164002)
    # Adding element type (line 1524)
    # Getting the type of 'y' (line 1524)
    y_164003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1524, 21), tuple_164001, y_164003)
    
    # Processing the call keyword arguments (line 1524)
    int_164004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 33), 'int')
    keyword_164005 = int_164004
    kwargs_164006 = {'copy': keyword_164005}
    # Getting the type of 'np' (line 1524)
    np_163999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1524)
    array_164000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 11), np_163999, 'array')
    # Calling array(args, kwargs) (line 1524)
    array_call_result_164007 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 11), array_164000, *[tuple_164001], **kwargs_164006)
    
    float_164008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 38), 'float')
    # Applying the binary operator '+' (line 1524)
    result_add_164009 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 11), '+', array_call_result_164007, float_164008)
    
    # Obtaining the member '__getitem__' of a type (line 1524)
    getitem___164010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 4), result_add_164009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1524)
    subscript_call_result_164011 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 4), getitem___164010, int_163998)
    
    # Assigning a type to the variable 'tuple_var_assignment_161917' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_161917', subscript_call_result_164011)
    
    # Assigning a Subscript to a Name (line 1524):
    
    # Obtaining the type of the subscript
    int_164012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 4), 'int')
    
    # Call to array(...): (line 1524)
    # Processing the call arguments (line 1524)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1524)
    tuple_164015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1524)
    # Adding element type (line 1524)
    # Getting the type of 'x' (line 1524)
    x_164016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1524, 21), tuple_164015, x_164016)
    # Adding element type (line 1524)
    # Getting the type of 'y' (line 1524)
    y_164017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1524, 21), tuple_164015, y_164017)
    
    # Processing the call keyword arguments (line 1524)
    int_164018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 33), 'int')
    keyword_164019 = int_164018
    kwargs_164020 = {'copy': keyword_164019}
    # Getting the type of 'np' (line 1524)
    np_164013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1524)
    array_164014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 11), np_164013, 'array')
    # Calling array(args, kwargs) (line 1524)
    array_call_result_164021 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 11), array_164014, *[tuple_164015], **kwargs_164020)
    
    float_164022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 38), 'float')
    # Applying the binary operator '+' (line 1524)
    result_add_164023 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 11), '+', array_call_result_164021, float_164022)
    
    # Obtaining the member '__getitem__' of a type (line 1524)
    getitem___164024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 4), result_add_164023, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1524)
    subscript_call_result_164025 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 4), getitem___164024, int_164012)
    
    # Assigning a type to the variable 'tuple_var_assignment_161918' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_161918', subscript_call_result_164025)
    
    # Assigning a Name to a Name (line 1524):
    # Getting the type of 'tuple_var_assignment_161917' (line 1524)
    tuple_var_assignment_161917_164026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_161917')
    # Assigning a type to the variable 'x' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'x', tuple_var_assignment_161917_164026)
    
    # Assigning a Name to a Name (line 1524):
    # Getting the type of 'tuple_var_assignment_161918' (line 1524)
    tuple_var_assignment_161918_164027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_161918')
    # Assigning a type to the variable 'y' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 7), 'y', tuple_var_assignment_161918_164027)
    
    # Assigning a Call to a Name (line 1526):
    
    # Assigning a Call to a Name (line 1526):
    
    # Call to chebvander(...): (line 1526)
    # Processing the call arguments (line 1526)
    # Getting the type of 'x' (line 1526)
    x_164029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 20), 'x', False)
    # Getting the type of 'degx' (line 1526)
    degx_164030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 23), 'degx', False)
    # Processing the call keyword arguments (line 1526)
    kwargs_164031 = {}
    # Getting the type of 'chebvander' (line 1526)
    chebvander_164028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 9), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1526)
    chebvander_call_result_164032 = invoke(stypy.reporting.localization.Localization(__file__, 1526, 9), chebvander_164028, *[x_164029, degx_164030], **kwargs_164031)
    
    # Assigning a type to the variable 'vx' (line 1526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1526, 4), 'vx', chebvander_call_result_164032)
    
    # Assigning a Call to a Name (line 1527):
    
    # Assigning a Call to a Name (line 1527):
    
    # Call to chebvander(...): (line 1527)
    # Processing the call arguments (line 1527)
    # Getting the type of 'y' (line 1527)
    y_164034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 20), 'y', False)
    # Getting the type of 'degy' (line 1527)
    degy_164035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 23), 'degy', False)
    # Processing the call keyword arguments (line 1527)
    kwargs_164036 = {}
    # Getting the type of 'chebvander' (line 1527)
    chebvander_164033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 9), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1527)
    chebvander_call_result_164037 = invoke(stypy.reporting.localization.Localization(__file__, 1527, 9), chebvander_164033, *[y_164034, degy_164035], **kwargs_164036)
    
    # Assigning a type to the variable 'vy' (line 1527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1527, 4), 'vy', chebvander_call_result_164037)
    
    # Assigning a BinOp to a Name (line 1528):
    
    # Assigning a BinOp to a Name (line 1528):
    
    # Obtaining the type of the subscript
    Ellipsis_164038 = Ellipsis
    # Getting the type of 'None' (line 1528)
    None_164039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 16), 'None')
    # Getting the type of 'vx' (line 1528)
    vx_164040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1528)
    getitem___164041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 8), vx_164040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1528)
    subscript_call_result_164042 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 8), getitem___164041, (Ellipsis_164038, None_164039))
    
    
    # Obtaining the type of the subscript
    Ellipsis_164043 = Ellipsis
    # Getting the type of 'None' (line 1528)
    None_164044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 30), 'None')
    slice_164045 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1528, 22), None, None, None)
    # Getting the type of 'vy' (line 1528)
    vy_164046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 22), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1528)
    getitem___164047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 22), vy_164046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1528)
    subscript_call_result_164048 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 22), getitem___164047, (Ellipsis_164043, None_164044, slice_164045))
    
    # Applying the binary operator '*' (line 1528)
    result_mul_164049 = python_operator(stypy.reporting.localization.Localization(__file__, 1528, 8), '*', subscript_call_result_164042, subscript_call_result_164048)
    
    # Assigning a type to the variable 'v' (line 1528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1528, 4), 'v', result_mul_164049)
    
    # Call to reshape(...): (line 1529)
    # Processing the call arguments (line 1529)
    
    # Obtaining the type of the subscript
    int_164052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1529, 30), 'int')
    slice_164053 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1529, 21), None, int_164052, None)
    # Getting the type of 'v' (line 1529)
    v_164054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1529)
    shape_164055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1529, 21), v_164054, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1529)
    getitem___164056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1529, 21), shape_164055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1529)
    subscript_call_result_164057 = invoke(stypy.reporting.localization.Localization(__file__, 1529, 21), getitem___164056, slice_164053)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1529)
    tuple_164058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1529, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1529)
    # Adding element type (line 1529)
    int_164059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1529, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1529, 37), tuple_164058, int_164059)
    
    # Applying the binary operator '+' (line 1529)
    result_add_164060 = python_operator(stypy.reporting.localization.Localization(__file__, 1529, 21), '+', subscript_call_result_164057, tuple_164058)
    
    # Processing the call keyword arguments (line 1529)
    kwargs_164061 = {}
    # Getting the type of 'v' (line 1529)
    v_164050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1529)
    reshape_164051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1529, 11), v_164050, 'reshape')
    # Calling reshape(args, kwargs) (line 1529)
    reshape_call_result_164062 = invoke(stypy.reporting.localization.Localization(__file__, 1529, 11), reshape_164051, *[result_add_164060], **kwargs_164061)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1529, 4), 'stypy_return_type', reshape_call_result_164062)
    
    # ################# End of 'chebvander2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebvander2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1469)
    stypy_return_type_164063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1469, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164063)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebvander2d'
    return stypy_return_type_164063

# Assigning a type to the variable 'chebvander2d' (line 1469)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1469, 0), 'chebvander2d', chebvander2d)

@norecursion
def chebvander3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebvander3d'
    module_type_store = module_type_store.open_function_context('chebvander3d', 1532, 0, False)
    
    # Passed parameters checking function
    chebvander3d.stypy_localization = localization
    chebvander3d.stypy_type_of_self = None
    chebvander3d.stypy_type_store = module_type_store
    chebvander3d.stypy_function_name = 'chebvander3d'
    chebvander3d.stypy_param_names_list = ['x', 'y', 'z', 'deg']
    chebvander3d.stypy_varargs_param_name = None
    chebvander3d.stypy_kwargs_param_name = None
    chebvander3d.stypy_call_defaults = defaults
    chebvander3d.stypy_call_varargs = varargs
    chebvander3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebvander3d', ['x', 'y', 'z', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebvander3d', localization, ['x', 'y', 'z', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebvander3d(...)' code ##################

    str_164064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1582, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x)*T_j(y)*T_k(z),\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the degrees of the Chebyshev polynomials.\n\n    If ``V = chebvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and ``np.dot(V, c.flat)`` and ``chebval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D Chebyshev\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    chebvander, chebvander3d. chebval2d, chebval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1583):
    
    # Assigning a ListComp to a Name (line 1583):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1583)
    deg_164069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 28), 'deg')
    comprehension_164070 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1583, 12), deg_164069)
    # Assigning a type to the variable 'd' (line 1583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1583, 12), 'd', comprehension_164070)
    
    # Call to int(...): (line 1583)
    # Processing the call arguments (line 1583)
    # Getting the type of 'd' (line 1583)
    d_164066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 16), 'd', False)
    # Processing the call keyword arguments (line 1583)
    kwargs_164067 = {}
    # Getting the type of 'int' (line 1583)
    int_164065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 12), 'int', False)
    # Calling int(args, kwargs) (line 1583)
    int_call_result_164068 = invoke(stypy.reporting.localization.Localization(__file__, 1583, 12), int_164065, *[d_164066], **kwargs_164067)
    
    list_164071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1583, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1583, 12), list_164071, int_call_result_164068)
    # Assigning a type to the variable 'ideg' (line 1583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1583, 4), 'ideg', list_164071)
    
    # Assigning a ListComp to a Name (line 1584):
    
    # Assigning a ListComp to a Name (line 1584):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1584)
    # Processing the call arguments (line 1584)
    # Getting the type of 'ideg' (line 1584)
    ideg_164080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1584)
    deg_164081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 59), 'deg', False)
    # Processing the call keyword arguments (line 1584)
    kwargs_164082 = {}
    # Getting the type of 'zip' (line 1584)
    zip_164079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1584)
    zip_call_result_164083 = invoke(stypy.reporting.localization.Localization(__file__, 1584, 49), zip_164079, *[ideg_164080, deg_164081], **kwargs_164082)
    
    comprehension_164084 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1584, 16), zip_call_result_164083)
    # Assigning a type to the variable 'id' (line 1584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1584, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1584, 16), comprehension_164084))
    # Assigning a type to the variable 'd' (line 1584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1584, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1584, 16), comprehension_164084))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1584)
    id_164072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 16), 'id')
    # Getting the type of 'd' (line 1584)
    d_164073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 22), 'd')
    # Applying the binary operator '==' (line 1584)
    result_eq_164074 = python_operator(stypy.reporting.localization.Localization(__file__, 1584, 16), '==', id_164072, d_164073)
    
    
    # Getting the type of 'id' (line 1584)
    id_164075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 28), 'id')
    int_164076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1584, 34), 'int')
    # Applying the binary operator '>=' (line 1584)
    result_ge_164077 = python_operator(stypy.reporting.localization.Localization(__file__, 1584, 28), '>=', id_164075, int_164076)
    
    # Applying the binary operator 'and' (line 1584)
    result_and_keyword_164078 = python_operator(stypy.reporting.localization.Localization(__file__, 1584, 16), 'and', result_eq_164074, result_ge_164077)
    
    list_164085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1584, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1584, 16), list_164085, result_and_keyword_164078)
    # Assigning a type to the variable 'is_valid' (line 1584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1584, 4), 'is_valid', list_164085)
    
    
    # Getting the type of 'is_valid' (line 1585)
    is_valid_164086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1585)
    list_164087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1585)
    # Adding element type (line 1585)
    int_164088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1585, 19), list_164087, int_164088)
    # Adding element type (line 1585)
    int_164089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1585, 19), list_164087, int_164089)
    # Adding element type (line 1585)
    int_164090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1585, 19), list_164087, int_164090)
    
    # Applying the binary operator '!=' (line 1585)
    result_ne_164091 = python_operator(stypy.reporting.localization.Localization(__file__, 1585, 7), '!=', is_valid_164086, list_164087)
    
    # Testing the type of an if condition (line 1585)
    if_condition_164092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1585, 4), result_ne_164091)
    # Assigning a type to the variable 'if_condition_164092' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'if_condition_164092', if_condition_164092)
    # SSA begins for if statement (line 1585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1586)
    # Processing the call arguments (line 1586)
    str_164094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1586, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1586)
    kwargs_164095 = {}
    # Getting the type of 'ValueError' (line 1586)
    ValueError_164093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1586)
    ValueError_call_result_164096 = invoke(stypy.reporting.localization.Localization(__file__, 1586, 14), ValueError_164093, *[str_164094], **kwargs_164095)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1586, 8), ValueError_call_result_164096, 'raise parameter', BaseException)
    # SSA join for if statement (line 1585)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1587):
    
    # Assigning a Subscript to a Name (line 1587):
    
    # Obtaining the type of the subscript
    int_164097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1587, 4), 'int')
    # Getting the type of 'ideg' (line 1587)
    ideg_164098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1587)
    getitem___164099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1587, 4), ideg_164098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1587)
    subscript_call_result_164100 = invoke(stypy.reporting.localization.Localization(__file__, 1587, 4), getitem___164099, int_164097)
    
    # Assigning a type to the variable 'tuple_var_assignment_161919' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'tuple_var_assignment_161919', subscript_call_result_164100)
    
    # Assigning a Subscript to a Name (line 1587):
    
    # Obtaining the type of the subscript
    int_164101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1587, 4), 'int')
    # Getting the type of 'ideg' (line 1587)
    ideg_164102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1587)
    getitem___164103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1587, 4), ideg_164102, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1587)
    subscript_call_result_164104 = invoke(stypy.reporting.localization.Localization(__file__, 1587, 4), getitem___164103, int_164101)
    
    # Assigning a type to the variable 'tuple_var_assignment_161920' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'tuple_var_assignment_161920', subscript_call_result_164104)
    
    # Assigning a Subscript to a Name (line 1587):
    
    # Obtaining the type of the subscript
    int_164105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1587, 4), 'int')
    # Getting the type of 'ideg' (line 1587)
    ideg_164106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1587)
    getitem___164107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1587, 4), ideg_164106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1587)
    subscript_call_result_164108 = invoke(stypy.reporting.localization.Localization(__file__, 1587, 4), getitem___164107, int_164105)
    
    # Assigning a type to the variable 'tuple_var_assignment_161921' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'tuple_var_assignment_161921', subscript_call_result_164108)
    
    # Assigning a Name to a Name (line 1587):
    # Getting the type of 'tuple_var_assignment_161919' (line 1587)
    tuple_var_assignment_161919_164109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'tuple_var_assignment_161919')
    # Assigning a type to the variable 'degx' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'degx', tuple_var_assignment_161919_164109)
    
    # Assigning a Name to a Name (line 1587):
    # Getting the type of 'tuple_var_assignment_161920' (line 1587)
    tuple_var_assignment_161920_164110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'tuple_var_assignment_161920')
    # Assigning a type to the variable 'degy' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 10), 'degy', tuple_var_assignment_161920_164110)
    
    # Assigning a Name to a Name (line 1587):
    # Getting the type of 'tuple_var_assignment_161921' (line 1587)
    tuple_var_assignment_161921_164111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 4), 'tuple_var_assignment_161921')
    # Assigning a type to the variable 'degz' (line 1587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1587, 16), 'degz', tuple_var_assignment_161921_164111)
    
    # Assigning a BinOp to a Tuple (line 1588):
    
    # Assigning a Subscript to a Name (line 1588):
    
    # Obtaining the type of the subscript
    int_164112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 4), 'int')
    
    # Call to array(...): (line 1588)
    # Processing the call arguments (line 1588)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1588)
    tuple_164115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1588)
    # Adding element type (line 1588)
    # Getting the type of 'x' (line 1588)
    x_164116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164115, x_164116)
    # Adding element type (line 1588)
    # Getting the type of 'y' (line 1588)
    y_164117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164115, y_164117)
    # Adding element type (line 1588)
    # Getting the type of 'z' (line 1588)
    z_164118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164115, z_164118)
    
    # Processing the call keyword arguments (line 1588)
    int_164119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 39), 'int')
    keyword_164120 = int_164119
    kwargs_164121 = {'copy': keyword_164120}
    # Getting the type of 'np' (line 1588)
    np_164113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1588)
    array_164114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 14), np_164113, 'array')
    # Calling array(args, kwargs) (line 1588)
    array_call_result_164122 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 14), array_164114, *[tuple_164115], **kwargs_164121)
    
    float_164123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 44), 'float')
    # Applying the binary operator '+' (line 1588)
    result_add_164124 = python_operator(stypy.reporting.localization.Localization(__file__, 1588, 14), '+', array_call_result_164122, float_164123)
    
    # Obtaining the member '__getitem__' of a type (line 1588)
    getitem___164125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 4), result_add_164124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1588)
    subscript_call_result_164126 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 4), getitem___164125, int_164112)
    
    # Assigning a type to the variable 'tuple_var_assignment_161922' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'tuple_var_assignment_161922', subscript_call_result_164126)
    
    # Assigning a Subscript to a Name (line 1588):
    
    # Obtaining the type of the subscript
    int_164127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 4), 'int')
    
    # Call to array(...): (line 1588)
    # Processing the call arguments (line 1588)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1588)
    tuple_164130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1588)
    # Adding element type (line 1588)
    # Getting the type of 'x' (line 1588)
    x_164131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164130, x_164131)
    # Adding element type (line 1588)
    # Getting the type of 'y' (line 1588)
    y_164132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164130, y_164132)
    # Adding element type (line 1588)
    # Getting the type of 'z' (line 1588)
    z_164133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164130, z_164133)
    
    # Processing the call keyword arguments (line 1588)
    int_164134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 39), 'int')
    keyword_164135 = int_164134
    kwargs_164136 = {'copy': keyword_164135}
    # Getting the type of 'np' (line 1588)
    np_164128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1588)
    array_164129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 14), np_164128, 'array')
    # Calling array(args, kwargs) (line 1588)
    array_call_result_164137 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 14), array_164129, *[tuple_164130], **kwargs_164136)
    
    float_164138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 44), 'float')
    # Applying the binary operator '+' (line 1588)
    result_add_164139 = python_operator(stypy.reporting.localization.Localization(__file__, 1588, 14), '+', array_call_result_164137, float_164138)
    
    # Obtaining the member '__getitem__' of a type (line 1588)
    getitem___164140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 4), result_add_164139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1588)
    subscript_call_result_164141 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 4), getitem___164140, int_164127)
    
    # Assigning a type to the variable 'tuple_var_assignment_161923' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'tuple_var_assignment_161923', subscript_call_result_164141)
    
    # Assigning a Subscript to a Name (line 1588):
    
    # Obtaining the type of the subscript
    int_164142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 4), 'int')
    
    # Call to array(...): (line 1588)
    # Processing the call arguments (line 1588)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1588)
    tuple_164145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1588)
    # Adding element type (line 1588)
    # Getting the type of 'x' (line 1588)
    x_164146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164145, x_164146)
    # Adding element type (line 1588)
    # Getting the type of 'y' (line 1588)
    y_164147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164145, y_164147)
    # Adding element type (line 1588)
    # Getting the type of 'z' (line 1588)
    z_164148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 24), tuple_164145, z_164148)
    
    # Processing the call keyword arguments (line 1588)
    int_164149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 39), 'int')
    keyword_164150 = int_164149
    kwargs_164151 = {'copy': keyword_164150}
    # Getting the type of 'np' (line 1588)
    np_164143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1588)
    array_164144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 14), np_164143, 'array')
    # Calling array(args, kwargs) (line 1588)
    array_call_result_164152 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 14), array_164144, *[tuple_164145], **kwargs_164151)
    
    float_164153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 44), 'float')
    # Applying the binary operator '+' (line 1588)
    result_add_164154 = python_operator(stypy.reporting.localization.Localization(__file__, 1588, 14), '+', array_call_result_164152, float_164153)
    
    # Obtaining the member '__getitem__' of a type (line 1588)
    getitem___164155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1588, 4), result_add_164154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1588)
    subscript_call_result_164156 = invoke(stypy.reporting.localization.Localization(__file__, 1588, 4), getitem___164155, int_164142)
    
    # Assigning a type to the variable 'tuple_var_assignment_161924' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'tuple_var_assignment_161924', subscript_call_result_164156)
    
    # Assigning a Name to a Name (line 1588):
    # Getting the type of 'tuple_var_assignment_161922' (line 1588)
    tuple_var_assignment_161922_164157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'tuple_var_assignment_161922')
    # Assigning a type to the variable 'x' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'x', tuple_var_assignment_161922_164157)
    
    # Assigning a Name to a Name (line 1588):
    # Getting the type of 'tuple_var_assignment_161923' (line 1588)
    tuple_var_assignment_161923_164158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'tuple_var_assignment_161923')
    # Assigning a type to the variable 'y' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 7), 'y', tuple_var_assignment_161923_164158)
    
    # Assigning a Name to a Name (line 1588):
    # Getting the type of 'tuple_var_assignment_161924' (line 1588)
    tuple_var_assignment_161924_164159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'tuple_var_assignment_161924')
    # Assigning a type to the variable 'z' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 10), 'z', tuple_var_assignment_161924_164159)
    
    # Assigning a Call to a Name (line 1590):
    
    # Assigning a Call to a Name (line 1590):
    
    # Call to chebvander(...): (line 1590)
    # Processing the call arguments (line 1590)
    # Getting the type of 'x' (line 1590)
    x_164161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 20), 'x', False)
    # Getting the type of 'degx' (line 1590)
    degx_164162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 23), 'degx', False)
    # Processing the call keyword arguments (line 1590)
    kwargs_164163 = {}
    # Getting the type of 'chebvander' (line 1590)
    chebvander_164160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 9), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1590)
    chebvander_call_result_164164 = invoke(stypy.reporting.localization.Localization(__file__, 1590, 9), chebvander_164160, *[x_164161, degx_164162], **kwargs_164163)
    
    # Assigning a type to the variable 'vx' (line 1590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1590, 4), 'vx', chebvander_call_result_164164)
    
    # Assigning a Call to a Name (line 1591):
    
    # Assigning a Call to a Name (line 1591):
    
    # Call to chebvander(...): (line 1591)
    # Processing the call arguments (line 1591)
    # Getting the type of 'y' (line 1591)
    y_164166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 20), 'y', False)
    # Getting the type of 'degy' (line 1591)
    degy_164167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 23), 'degy', False)
    # Processing the call keyword arguments (line 1591)
    kwargs_164168 = {}
    # Getting the type of 'chebvander' (line 1591)
    chebvander_164165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 9), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1591)
    chebvander_call_result_164169 = invoke(stypy.reporting.localization.Localization(__file__, 1591, 9), chebvander_164165, *[y_164166, degy_164167], **kwargs_164168)
    
    # Assigning a type to the variable 'vy' (line 1591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1591, 4), 'vy', chebvander_call_result_164169)
    
    # Assigning a Call to a Name (line 1592):
    
    # Assigning a Call to a Name (line 1592):
    
    # Call to chebvander(...): (line 1592)
    # Processing the call arguments (line 1592)
    # Getting the type of 'z' (line 1592)
    z_164171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 20), 'z', False)
    # Getting the type of 'degz' (line 1592)
    degz_164172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 23), 'degz', False)
    # Processing the call keyword arguments (line 1592)
    kwargs_164173 = {}
    # Getting the type of 'chebvander' (line 1592)
    chebvander_164170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 9), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1592)
    chebvander_call_result_164174 = invoke(stypy.reporting.localization.Localization(__file__, 1592, 9), chebvander_164170, *[z_164171, degz_164172], **kwargs_164173)
    
    # Assigning a type to the variable 'vz' (line 1592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1592, 4), 'vz', chebvander_call_result_164174)
    
    # Assigning a BinOp to a Name (line 1593):
    
    # Assigning a BinOp to a Name (line 1593):
    
    # Obtaining the type of the subscript
    Ellipsis_164175 = Ellipsis
    # Getting the type of 'None' (line 1593)
    None_164176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 16), 'None')
    # Getting the type of 'None' (line 1593)
    None_164177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 22), 'None')
    # Getting the type of 'vx' (line 1593)
    vx_164178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1593)
    getitem___164179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1593, 8), vx_164178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1593)
    subscript_call_result_164180 = invoke(stypy.reporting.localization.Localization(__file__, 1593, 8), getitem___164179, (Ellipsis_164175, None_164176, None_164177))
    
    
    # Obtaining the type of the subscript
    Ellipsis_164181 = Ellipsis
    # Getting the type of 'None' (line 1593)
    None_164182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 36), 'None')
    slice_164183 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1593, 28), None, None, None)
    # Getting the type of 'None' (line 1593)
    None_164184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 44), 'None')
    # Getting the type of 'vy' (line 1593)
    vy_164185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 28), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1593)
    getitem___164186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1593, 28), vy_164185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1593)
    subscript_call_result_164187 = invoke(stypy.reporting.localization.Localization(__file__, 1593, 28), getitem___164186, (Ellipsis_164181, None_164182, slice_164183, None_164184))
    
    # Applying the binary operator '*' (line 1593)
    result_mul_164188 = python_operator(stypy.reporting.localization.Localization(__file__, 1593, 8), '*', subscript_call_result_164180, subscript_call_result_164187)
    
    
    # Obtaining the type of the subscript
    Ellipsis_164189 = Ellipsis
    # Getting the type of 'None' (line 1593)
    None_164190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 58), 'None')
    # Getting the type of 'None' (line 1593)
    None_164191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 64), 'None')
    slice_164192 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1593, 50), None, None, None)
    # Getting the type of 'vz' (line 1593)
    vz_164193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 50), 'vz')
    # Obtaining the member '__getitem__' of a type (line 1593)
    getitem___164194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1593, 50), vz_164193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1593)
    subscript_call_result_164195 = invoke(stypy.reporting.localization.Localization(__file__, 1593, 50), getitem___164194, (Ellipsis_164189, None_164190, None_164191, slice_164192))
    
    # Applying the binary operator '*' (line 1593)
    result_mul_164196 = python_operator(stypy.reporting.localization.Localization(__file__, 1593, 49), '*', result_mul_164188, subscript_call_result_164195)
    
    # Assigning a type to the variable 'v' (line 1593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1593, 4), 'v', result_mul_164196)
    
    # Call to reshape(...): (line 1594)
    # Processing the call arguments (line 1594)
    
    # Obtaining the type of the subscript
    int_164199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 30), 'int')
    slice_164200 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1594, 21), None, int_164199, None)
    # Getting the type of 'v' (line 1594)
    v_164201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1594)
    shape_164202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 21), v_164201, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1594)
    getitem___164203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 21), shape_164202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1594)
    subscript_call_result_164204 = invoke(stypy.reporting.localization.Localization(__file__, 1594, 21), getitem___164203, slice_164200)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1594)
    tuple_164205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1594)
    # Adding element type (line 1594)
    int_164206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1594, 37), tuple_164205, int_164206)
    
    # Applying the binary operator '+' (line 1594)
    result_add_164207 = python_operator(stypy.reporting.localization.Localization(__file__, 1594, 21), '+', subscript_call_result_164204, tuple_164205)
    
    # Processing the call keyword arguments (line 1594)
    kwargs_164208 = {}
    # Getting the type of 'v' (line 1594)
    v_164197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1594)
    reshape_164198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 11), v_164197, 'reshape')
    # Calling reshape(args, kwargs) (line 1594)
    reshape_call_result_164209 = invoke(stypy.reporting.localization.Localization(__file__, 1594, 11), reshape_164198, *[result_add_164207], **kwargs_164208)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'stypy_return_type', reshape_call_result_164209)
    
    # ################# End of 'chebvander3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebvander3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1532)
    stypy_return_type_164210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164210)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebvander3d'
    return stypy_return_type_164210

# Assigning a type to the variable 'chebvander3d' (line 1532)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1532, 0), 'chebvander3d', chebvander3d)

@norecursion
def chebfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1597)
    None_164211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 29), 'None')
    # Getting the type of 'False' (line 1597)
    False_164212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 40), 'False')
    # Getting the type of 'None' (line 1597)
    None_164213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 49), 'None')
    defaults = [None_164211, False_164212, None_164213]
    # Create a new context for function 'chebfit'
    module_type_store = module_type_store.open_function_context('chebfit', 1597, 0, False)
    
    # Passed parameters checking function
    chebfit.stypy_localization = localization
    chebfit.stypy_type_of_self = None
    chebfit.stypy_type_store = module_type_store
    chebfit.stypy_function_name = 'chebfit'
    chebfit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w']
    chebfit.stypy_varargs_param_name = None
    chebfit.stypy_kwargs_param_name = None
    chebfit.stypy_call_defaults = defaults
    chebfit.stypy_call_varargs = varargs
    chebfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebfit', ['x', 'y', 'deg', 'rcond', 'full', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebfit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebfit(...)' code ##################

    str_164214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1715, (-1)), 'str', '\n    Least squares fit of Chebyshev series to data.\n\n    Return the coefficients of a Legendre series of degree `deg` that is the\n    least squares fit to the data values `y` given at points `x`. If `y` is\n    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple\n    fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),\n\n    where `n` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For Numpy versions >= 1.11 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the contribution of each point\n        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n        weights are chosen so that the errors of the products ``w[i]*y[i]``\n        all have the same variance.  The default value is None.\n\n        .. versionadded:: 1.5.0\n\n    Returns\n    -------\n    coef : ndarray, shape (M,) or (M, K)\n        Chebyshev coefficients ordered from low to high. If `y` was 2-D,\n        the coefficients for the data in column k  of `y` are in column\n        `k`.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if `full` = True\n\n        resid -- sum of squared residuals of the least squares fit\n        rank -- the numerical rank of the scaled Vandermonde matrix\n        sv -- singular values of the scaled Vandermonde matrix\n        rcond -- value of `rcond`.\n\n        For more details, see `linalg.lstsq`.\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if `full` = False.  The\n        warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', RankWarning)\n\n    See Also\n    --------\n    polyfit, legfit, lagfit, hermfit, hermefit\n    chebval : Evaluates a Chebyshev series.\n    chebvander : Vandermonde matrix of Chebyshev series.\n    chebweight : Chebyshev weight function.\n    linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the Chebyshev series `p` that\n    minimizes the sum of the weighted squared errors\n\n    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where :math:`w_j` are the weights. This problem is solved by setting up\n    as the (typically) overdetermined matrix equation\n\n    .. math:: V(x) * c = w * y,\n\n    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the\n    coefficients to be solved for, `w` are the weights, and `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of `V`.\n\n    If some of the singular values of `V` are so small that they are\n    neglected, then a `RankWarning` will be issued. This means that the\n    coefficient values may be poorly determined. Using a lower order fit\n    will usually get rid of the warning.  The `rcond` parameter can also be\n    set to a value smaller than its default, but the resulting fit may be\n    spurious and have large contributions from roundoff error.\n\n    Fits using Chebyshev series are usually better conditioned than fits\n    using power series, but much can depend on the distribution of the\n    sample points and the smoothness of the data. If the quality of the fit\n    is inadequate splines may be a good alternative.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           http://en.wikipedia.org/wiki/Curve_fitting\n\n    Examples\n    --------\n\n    ')
    
    # Assigning a BinOp to a Name (line 1716):
    
    # Assigning a BinOp to a Name (line 1716):
    
    # Call to asarray(...): (line 1716)
    # Processing the call arguments (line 1716)
    # Getting the type of 'x' (line 1716)
    x_164217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1716, 19), 'x', False)
    # Processing the call keyword arguments (line 1716)
    kwargs_164218 = {}
    # Getting the type of 'np' (line 1716)
    np_164215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1716, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1716)
    asarray_164216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1716, 8), np_164215, 'asarray')
    # Calling asarray(args, kwargs) (line 1716)
    asarray_call_result_164219 = invoke(stypy.reporting.localization.Localization(__file__, 1716, 8), asarray_164216, *[x_164217], **kwargs_164218)
    
    float_164220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1716, 24), 'float')
    # Applying the binary operator '+' (line 1716)
    result_add_164221 = python_operator(stypy.reporting.localization.Localization(__file__, 1716, 8), '+', asarray_call_result_164219, float_164220)
    
    # Assigning a type to the variable 'x' (line 1716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1716, 4), 'x', result_add_164221)
    
    # Assigning a BinOp to a Name (line 1717):
    
    # Assigning a BinOp to a Name (line 1717):
    
    # Call to asarray(...): (line 1717)
    # Processing the call arguments (line 1717)
    # Getting the type of 'y' (line 1717)
    y_164224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1717, 19), 'y', False)
    # Processing the call keyword arguments (line 1717)
    kwargs_164225 = {}
    # Getting the type of 'np' (line 1717)
    np_164222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1717, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1717)
    asarray_164223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1717, 8), np_164222, 'asarray')
    # Calling asarray(args, kwargs) (line 1717)
    asarray_call_result_164226 = invoke(stypy.reporting.localization.Localization(__file__, 1717, 8), asarray_164223, *[y_164224], **kwargs_164225)
    
    float_164227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1717, 24), 'float')
    # Applying the binary operator '+' (line 1717)
    result_add_164228 = python_operator(stypy.reporting.localization.Localization(__file__, 1717, 8), '+', asarray_call_result_164226, float_164227)
    
    # Assigning a type to the variable 'y' (line 1717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1717, 4), 'y', result_add_164228)
    
    # Assigning a Call to a Name (line 1718):
    
    # Assigning a Call to a Name (line 1718):
    
    # Call to asarray(...): (line 1718)
    # Processing the call arguments (line 1718)
    # Getting the type of 'deg' (line 1718)
    deg_164231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 21), 'deg', False)
    # Processing the call keyword arguments (line 1718)
    kwargs_164232 = {}
    # Getting the type of 'np' (line 1718)
    np_164229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1718)
    asarray_164230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1718, 10), np_164229, 'asarray')
    # Calling asarray(args, kwargs) (line 1718)
    asarray_call_result_164233 = invoke(stypy.reporting.localization.Localization(__file__, 1718, 10), asarray_164230, *[deg_164231], **kwargs_164232)
    
    # Assigning a type to the variable 'deg' (line 1718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1718, 4), 'deg', asarray_call_result_164233)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deg' (line 1721)
    deg_164234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1721)
    ndim_164235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 7), deg_164234, 'ndim')
    int_164236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1721, 18), 'int')
    # Applying the binary operator '>' (line 1721)
    result_gt_164237 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 7), '>', ndim_164235, int_164236)
    
    
    # Getting the type of 'deg' (line 1721)
    deg_164238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 23), 'deg')
    # Obtaining the member 'dtype' of a type (line 1721)
    dtype_164239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 23), deg_164238, 'dtype')
    # Obtaining the member 'kind' of a type (line 1721)
    kind_164240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 23), dtype_164239, 'kind')
    str_164241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1721, 45), 'str', 'iu')
    # Applying the binary operator 'notin' (line 1721)
    result_contains_164242 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 23), 'notin', kind_164240, str_164241)
    
    # Applying the binary operator 'or' (line 1721)
    result_or_keyword_164243 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 7), 'or', result_gt_164237, result_contains_164242)
    
    # Getting the type of 'deg' (line 1721)
    deg_164244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 53), 'deg')
    # Obtaining the member 'size' of a type (line 1721)
    size_164245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 53), deg_164244, 'size')
    int_164246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1721, 65), 'int')
    # Applying the binary operator '==' (line 1721)
    result_eq_164247 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 53), '==', size_164245, int_164246)
    
    # Applying the binary operator 'or' (line 1721)
    result_or_keyword_164248 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 7), 'or', result_or_keyword_164243, result_eq_164247)
    
    # Testing the type of an if condition (line 1721)
    if_condition_164249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1721, 4), result_or_keyword_164248)
    # Assigning a type to the variable 'if_condition_164249' (line 1721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1721, 4), 'if_condition_164249', if_condition_164249)
    # SSA begins for if statement (line 1721)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1722)
    # Processing the call arguments (line 1722)
    str_164251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1722, 24), 'str', 'deg must be an int or non-empty 1-D array of int')
    # Processing the call keyword arguments (line 1722)
    kwargs_164252 = {}
    # Getting the type of 'TypeError' (line 1722)
    TypeError_164250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1722, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1722)
    TypeError_call_result_164253 = invoke(stypy.reporting.localization.Localization(__file__, 1722, 14), TypeError_164250, *[str_164251], **kwargs_164252)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1722, 8), TypeError_call_result_164253, 'raise parameter', BaseException)
    # SSA join for if statement (line 1721)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to min(...): (line 1723)
    # Processing the call keyword arguments (line 1723)
    kwargs_164256 = {}
    # Getting the type of 'deg' (line 1723)
    deg_164254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1723, 7), 'deg', False)
    # Obtaining the member 'min' of a type (line 1723)
    min_164255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1723, 7), deg_164254, 'min')
    # Calling min(args, kwargs) (line 1723)
    min_call_result_164257 = invoke(stypy.reporting.localization.Localization(__file__, 1723, 7), min_164255, *[], **kwargs_164256)
    
    int_164258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1723, 19), 'int')
    # Applying the binary operator '<' (line 1723)
    result_lt_164259 = python_operator(stypy.reporting.localization.Localization(__file__, 1723, 7), '<', min_call_result_164257, int_164258)
    
    # Testing the type of an if condition (line 1723)
    if_condition_164260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1723, 4), result_lt_164259)
    # Assigning a type to the variable 'if_condition_164260' (line 1723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1723, 4), 'if_condition_164260', if_condition_164260)
    # SSA begins for if statement (line 1723)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1724)
    # Processing the call arguments (line 1724)
    str_164262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1724, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 1724)
    kwargs_164263 = {}
    # Getting the type of 'ValueError' (line 1724)
    ValueError_164261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1724, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1724)
    ValueError_call_result_164264 = invoke(stypy.reporting.localization.Localization(__file__, 1724, 14), ValueError_164261, *[str_164262], **kwargs_164263)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1724, 8), ValueError_call_result_164264, 'raise parameter', BaseException)
    # SSA join for if statement (line 1723)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1725)
    x_164265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1725, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1725)
    ndim_164266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1725, 7), x_164265, 'ndim')
    int_164267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1725, 17), 'int')
    # Applying the binary operator '!=' (line 1725)
    result_ne_164268 = python_operator(stypy.reporting.localization.Localization(__file__, 1725, 7), '!=', ndim_164266, int_164267)
    
    # Testing the type of an if condition (line 1725)
    if_condition_164269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1725, 4), result_ne_164268)
    # Assigning a type to the variable 'if_condition_164269' (line 1725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1725, 4), 'if_condition_164269', if_condition_164269)
    # SSA begins for if statement (line 1725)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1726)
    # Processing the call arguments (line 1726)
    str_164271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1726, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 1726)
    kwargs_164272 = {}
    # Getting the type of 'TypeError' (line 1726)
    TypeError_164270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1726, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1726)
    TypeError_call_result_164273 = invoke(stypy.reporting.localization.Localization(__file__, 1726, 14), TypeError_164270, *[str_164271], **kwargs_164272)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1726, 8), TypeError_call_result_164273, 'raise parameter', BaseException)
    # SSA join for if statement (line 1725)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1727)
    x_164274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1727, 7), 'x')
    # Obtaining the member 'size' of a type (line 1727)
    size_164275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1727, 7), x_164274, 'size')
    int_164276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1727, 17), 'int')
    # Applying the binary operator '==' (line 1727)
    result_eq_164277 = python_operator(stypy.reporting.localization.Localization(__file__, 1727, 7), '==', size_164275, int_164276)
    
    # Testing the type of an if condition (line 1727)
    if_condition_164278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1727, 4), result_eq_164277)
    # Assigning a type to the variable 'if_condition_164278' (line 1727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1727, 4), 'if_condition_164278', if_condition_164278)
    # SSA begins for if statement (line 1727)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1728)
    # Processing the call arguments (line 1728)
    str_164280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1728, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 1728)
    kwargs_164281 = {}
    # Getting the type of 'TypeError' (line 1728)
    TypeError_164279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1728, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1728)
    TypeError_call_result_164282 = invoke(stypy.reporting.localization.Localization(__file__, 1728, 14), TypeError_164279, *[str_164280], **kwargs_164281)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1728, 8), TypeError_call_result_164282, 'raise parameter', BaseException)
    # SSA join for if statement (line 1727)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 1729)
    y_164283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1729, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1729)
    ndim_164284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1729, 7), y_164283, 'ndim')
    int_164285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1729, 16), 'int')
    # Applying the binary operator '<' (line 1729)
    result_lt_164286 = python_operator(stypy.reporting.localization.Localization(__file__, 1729, 7), '<', ndim_164284, int_164285)
    
    
    # Getting the type of 'y' (line 1729)
    y_164287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1729, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 1729)
    ndim_164288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1729, 21), y_164287, 'ndim')
    int_164289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1729, 30), 'int')
    # Applying the binary operator '>' (line 1729)
    result_gt_164290 = python_operator(stypy.reporting.localization.Localization(__file__, 1729, 21), '>', ndim_164288, int_164289)
    
    # Applying the binary operator 'or' (line 1729)
    result_or_keyword_164291 = python_operator(stypy.reporting.localization.Localization(__file__, 1729, 7), 'or', result_lt_164286, result_gt_164290)
    
    # Testing the type of an if condition (line 1729)
    if_condition_164292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1729, 4), result_or_keyword_164291)
    # Assigning a type to the variable 'if_condition_164292' (line 1729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1729, 4), 'if_condition_164292', if_condition_164292)
    # SSA begins for if statement (line 1729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1730)
    # Processing the call arguments (line 1730)
    str_164294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1730, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 1730)
    kwargs_164295 = {}
    # Getting the type of 'TypeError' (line 1730)
    TypeError_164293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1730, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1730)
    TypeError_call_result_164296 = invoke(stypy.reporting.localization.Localization(__file__, 1730, 14), TypeError_164293, *[str_164294], **kwargs_164295)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1730, 8), TypeError_call_result_164296, 'raise parameter', BaseException)
    # SSA join for if statement (line 1729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1731)
    # Processing the call arguments (line 1731)
    # Getting the type of 'x' (line 1731)
    x_164298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1731, 11), 'x', False)
    # Processing the call keyword arguments (line 1731)
    kwargs_164299 = {}
    # Getting the type of 'len' (line 1731)
    len_164297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1731, 7), 'len', False)
    # Calling len(args, kwargs) (line 1731)
    len_call_result_164300 = invoke(stypy.reporting.localization.Localization(__file__, 1731, 7), len_164297, *[x_164298], **kwargs_164299)
    
    
    # Call to len(...): (line 1731)
    # Processing the call arguments (line 1731)
    # Getting the type of 'y' (line 1731)
    y_164302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1731, 21), 'y', False)
    # Processing the call keyword arguments (line 1731)
    kwargs_164303 = {}
    # Getting the type of 'len' (line 1731)
    len_164301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1731, 17), 'len', False)
    # Calling len(args, kwargs) (line 1731)
    len_call_result_164304 = invoke(stypy.reporting.localization.Localization(__file__, 1731, 17), len_164301, *[y_164302], **kwargs_164303)
    
    # Applying the binary operator '!=' (line 1731)
    result_ne_164305 = python_operator(stypy.reporting.localization.Localization(__file__, 1731, 7), '!=', len_call_result_164300, len_call_result_164304)
    
    # Testing the type of an if condition (line 1731)
    if_condition_164306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1731, 4), result_ne_164305)
    # Assigning a type to the variable 'if_condition_164306' (line 1731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1731, 4), 'if_condition_164306', if_condition_164306)
    # SSA begins for if statement (line 1731)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1732)
    # Processing the call arguments (line 1732)
    str_164308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1732, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 1732)
    kwargs_164309 = {}
    # Getting the type of 'TypeError' (line 1732)
    TypeError_164307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1732, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1732)
    TypeError_call_result_164310 = invoke(stypy.reporting.localization.Localization(__file__, 1732, 14), TypeError_164307, *[str_164308], **kwargs_164309)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1732, 8), TypeError_call_result_164310, 'raise parameter', BaseException)
    # SSA join for if statement (line 1731)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'deg' (line 1734)
    deg_164311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1734, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1734)
    ndim_164312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1734, 7), deg_164311, 'ndim')
    int_164313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1734, 19), 'int')
    # Applying the binary operator '==' (line 1734)
    result_eq_164314 = python_operator(stypy.reporting.localization.Localization(__file__, 1734, 7), '==', ndim_164312, int_164313)
    
    # Testing the type of an if condition (line 1734)
    if_condition_164315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1734, 4), result_eq_164314)
    # Assigning a type to the variable 'if_condition_164315' (line 1734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1734, 4), 'if_condition_164315', if_condition_164315)
    # SSA begins for if statement (line 1734)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1735):
    
    # Assigning a Name to a Name (line 1735):
    # Getting the type of 'deg' (line 1735)
    deg_164316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1735, 15), 'deg')
    # Assigning a type to the variable 'lmax' (line 1735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1735, 8), 'lmax', deg_164316)
    
    # Assigning a BinOp to a Name (line 1736):
    
    # Assigning a BinOp to a Name (line 1736):
    # Getting the type of 'lmax' (line 1736)
    lmax_164317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 16), 'lmax')
    int_164318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1736, 23), 'int')
    # Applying the binary operator '+' (line 1736)
    result_add_164319 = python_operator(stypy.reporting.localization.Localization(__file__, 1736, 16), '+', lmax_164317, int_164318)
    
    # Assigning a type to the variable 'order' (line 1736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1736, 8), 'order', result_add_164319)
    
    # Assigning a Call to a Name (line 1737):
    
    # Assigning a Call to a Name (line 1737):
    
    # Call to chebvander(...): (line 1737)
    # Processing the call arguments (line 1737)
    # Getting the type of 'x' (line 1737)
    x_164321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 25), 'x', False)
    # Getting the type of 'lmax' (line 1737)
    lmax_164322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 28), 'lmax', False)
    # Processing the call keyword arguments (line 1737)
    kwargs_164323 = {}
    # Getting the type of 'chebvander' (line 1737)
    chebvander_164320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 14), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1737)
    chebvander_call_result_164324 = invoke(stypy.reporting.localization.Localization(__file__, 1737, 14), chebvander_164320, *[x_164321, lmax_164322], **kwargs_164323)
    
    # Assigning a type to the variable 'van' (line 1737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1737, 8), 'van', chebvander_call_result_164324)
    # SSA branch for the else part of an if statement (line 1734)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1739):
    
    # Assigning a Call to a Name (line 1739):
    
    # Call to sort(...): (line 1739)
    # Processing the call arguments (line 1739)
    # Getting the type of 'deg' (line 1739)
    deg_164327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 22), 'deg', False)
    # Processing the call keyword arguments (line 1739)
    kwargs_164328 = {}
    # Getting the type of 'np' (line 1739)
    np_164325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 14), 'np', False)
    # Obtaining the member 'sort' of a type (line 1739)
    sort_164326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1739, 14), np_164325, 'sort')
    # Calling sort(args, kwargs) (line 1739)
    sort_call_result_164329 = invoke(stypy.reporting.localization.Localization(__file__, 1739, 14), sort_164326, *[deg_164327], **kwargs_164328)
    
    # Assigning a type to the variable 'deg' (line 1739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1739, 8), 'deg', sort_call_result_164329)
    
    # Assigning a Subscript to a Name (line 1740):
    
    # Assigning a Subscript to a Name (line 1740):
    
    # Obtaining the type of the subscript
    int_164330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1740, 19), 'int')
    # Getting the type of 'deg' (line 1740)
    deg_164331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 15), 'deg')
    # Obtaining the member '__getitem__' of a type (line 1740)
    getitem___164332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1740, 15), deg_164331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1740)
    subscript_call_result_164333 = invoke(stypy.reporting.localization.Localization(__file__, 1740, 15), getitem___164332, int_164330)
    
    # Assigning a type to the variable 'lmax' (line 1740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1740, 8), 'lmax', subscript_call_result_164333)
    
    # Assigning a Call to a Name (line 1741):
    
    # Assigning a Call to a Name (line 1741):
    
    # Call to len(...): (line 1741)
    # Processing the call arguments (line 1741)
    # Getting the type of 'deg' (line 1741)
    deg_164335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1741, 20), 'deg', False)
    # Processing the call keyword arguments (line 1741)
    kwargs_164336 = {}
    # Getting the type of 'len' (line 1741)
    len_164334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1741, 16), 'len', False)
    # Calling len(args, kwargs) (line 1741)
    len_call_result_164337 = invoke(stypy.reporting.localization.Localization(__file__, 1741, 16), len_164334, *[deg_164335], **kwargs_164336)
    
    # Assigning a type to the variable 'order' (line 1741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1741, 8), 'order', len_call_result_164337)
    
    # Assigning a Subscript to a Name (line 1742):
    
    # Assigning a Subscript to a Name (line 1742):
    
    # Obtaining the type of the subscript
    slice_164338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1742, 14), None, None, None)
    # Getting the type of 'deg' (line 1742)
    deg_164339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1742, 37), 'deg')
    
    # Call to chebvander(...): (line 1742)
    # Processing the call arguments (line 1742)
    # Getting the type of 'x' (line 1742)
    x_164341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1742, 25), 'x', False)
    # Getting the type of 'lmax' (line 1742)
    lmax_164342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1742, 28), 'lmax', False)
    # Processing the call keyword arguments (line 1742)
    kwargs_164343 = {}
    # Getting the type of 'chebvander' (line 1742)
    chebvander_164340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1742, 14), 'chebvander', False)
    # Calling chebvander(args, kwargs) (line 1742)
    chebvander_call_result_164344 = invoke(stypy.reporting.localization.Localization(__file__, 1742, 14), chebvander_164340, *[x_164341, lmax_164342], **kwargs_164343)
    
    # Obtaining the member '__getitem__' of a type (line 1742)
    getitem___164345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1742, 14), chebvander_call_result_164344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1742)
    subscript_call_result_164346 = invoke(stypy.reporting.localization.Localization(__file__, 1742, 14), getitem___164345, (slice_164338, deg_164339))
    
    # Assigning a type to the variable 'van' (line 1742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1742, 8), 'van', subscript_call_result_164346)
    # SSA join for if statement (line 1734)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1745):
    
    # Assigning a Attribute to a Name (line 1745):
    # Getting the type of 'van' (line 1745)
    van_164347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 10), 'van')
    # Obtaining the member 'T' of a type (line 1745)
    T_164348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1745, 10), van_164347, 'T')
    # Assigning a type to the variable 'lhs' (line 1745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1745, 4), 'lhs', T_164348)
    
    # Assigning a Attribute to a Name (line 1746):
    
    # Assigning a Attribute to a Name (line 1746):
    # Getting the type of 'y' (line 1746)
    y_164349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1746, 10), 'y')
    # Obtaining the member 'T' of a type (line 1746)
    T_164350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1746, 10), y_164349, 'T')
    # Assigning a type to the variable 'rhs' (line 1746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1746, 4), 'rhs', T_164350)
    
    # Type idiom detected: calculating its left and rigth part (line 1747)
    # Getting the type of 'w' (line 1747)
    w_164351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1747, 4), 'w')
    # Getting the type of 'None' (line 1747)
    None_164352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1747, 16), 'None')
    
    (may_be_164353, more_types_in_union_164354) = may_not_be_none(w_164351, None_164352)

    if may_be_164353:

        if more_types_in_union_164354:
            # Runtime conditional SSA (line 1747)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1748):
        
        # Assigning a BinOp to a Name (line 1748):
        
        # Call to asarray(...): (line 1748)
        # Processing the call arguments (line 1748)
        # Getting the type of 'w' (line 1748)
        w_164357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1748, 23), 'w', False)
        # Processing the call keyword arguments (line 1748)
        kwargs_164358 = {}
        # Getting the type of 'np' (line 1748)
        np_164355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1748, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1748)
        asarray_164356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1748, 12), np_164355, 'asarray')
        # Calling asarray(args, kwargs) (line 1748)
        asarray_call_result_164359 = invoke(stypy.reporting.localization.Localization(__file__, 1748, 12), asarray_164356, *[w_164357], **kwargs_164358)
        
        float_164360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1748, 28), 'float')
        # Applying the binary operator '+' (line 1748)
        result_add_164361 = python_operator(stypy.reporting.localization.Localization(__file__, 1748, 12), '+', asarray_call_result_164359, float_164360)
        
        # Assigning a type to the variable 'w' (line 1748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1748, 8), 'w', result_add_164361)
        
        
        # Getting the type of 'w' (line 1749)
        w_164362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 1749)
        ndim_164363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1749, 11), w_164362, 'ndim')
        int_164364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1749, 21), 'int')
        # Applying the binary operator '!=' (line 1749)
        result_ne_164365 = python_operator(stypy.reporting.localization.Localization(__file__, 1749, 11), '!=', ndim_164363, int_164364)
        
        # Testing the type of an if condition (line 1749)
        if_condition_164366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1749, 8), result_ne_164365)
        # Assigning a type to the variable 'if_condition_164366' (line 1749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1749, 8), 'if_condition_164366', if_condition_164366)
        # SSA begins for if statement (line 1749)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1750)
        # Processing the call arguments (line 1750)
        str_164368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1750, 28), 'str', 'expected 1D vector for w')
        # Processing the call keyword arguments (line 1750)
        kwargs_164369 = {}
        # Getting the type of 'TypeError' (line 1750)
        TypeError_164367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1750)
        TypeError_call_result_164370 = invoke(stypy.reporting.localization.Localization(__file__, 1750, 18), TypeError_164367, *[str_164368], **kwargs_164369)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1750, 12), TypeError_call_result_164370, 'raise parameter', BaseException)
        # SSA join for if statement (line 1749)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 1751)
        # Processing the call arguments (line 1751)
        # Getting the type of 'x' (line 1751)
        x_164372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 15), 'x', False)
        # Processing the call keyword arguments (line 1751)
        kwargs_164373 = {}
        # Getting the type of 'len' (line 1751)
        len_164371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 11), 'len', False)
        # Calling len(args, kwargs) (line 1751)
        len_call_result_164374 = invoke(stypy.reporting.localization.Localization(__file__, 1751, 11), len_164371, *[x_164372], **kwargs_164373)
        
        
        # Call to len(...): (line 1751)
        # Processing the call arguments (line 1751)
        # Getting the type of 'w' (line 1751)
        w_164376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 25), 'w', False)
        # Processing the call keyword arguments (line 1751)
        kwargs_164377 = {}
        # Getting the type of 'len' (line 1751)
        len_164375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 21), 'len', False)
        # Calling len(args, kwargs) (line 1751)
        len_call_result_164378 = invoke(stypy.reporting.localization.Localization(__file__, 1751, 21), len_164375, *[w_164376], **kwargs_164377)
        
        # Applying the binary operator '!=' (line 1751)
        result_ne_164379 = python_operator(stypy.reporting.localization.Localization(__file__, 1751, 11), '!=', len_call_result_164374, len_call_result_164378)
        
        # Testing the type of an if condition (line 1751)
        if_condition_164380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1751, 8), result_ne_164379)
        # Assigning a type to the variable 'if_condition_164380' (line 1751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1751, 8), 'if_condition_164380', if_condition_164380)
        # SSA begins for if statement (line 1751)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1752)
        # Processing the call arguments (line 1752)
        str_164382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1752, 28), 'str', 'expected x and w to have same length')
        # Processing the call keyword arguments (line 1752)
        kwargs_164383 = {}
        # Getting the type of 'TypeError' (line 1752)
        TypeError_164381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1752)
        TypeError_call_result_164384 = invoke(stypy.reporting.localization.Localization(__file__, 1752, 18), TypeError_164381, *[str_164382], **kwargs_164383)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1752, 12), TypeError_call_result_164384, 'raise parameter', BaseException)
        # SSA join for if statement (line 1751)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1755):
        
        # Assigning a BinOp to a Name (line 1755):
        # Getting the type of 'lhs' (line 1755)
        lhs_164385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 14), 'lhs')
        # Getting the type of 'w' (line 1755)
        w_164386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 20), 'w')
        # Applying the binary operator '*' (line 1755)
        result_mul_164387 = python_operator(stypy.reporting.localization.Localization(__file__, 1755, 14), '*', lhs_164385, w_164386)
        
        # Assigning a type to the variable 'lhs' (line 1755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1755, 8), 'lhs', result_mul_164387)
        
        # Assigning a BinOp to a Name (line 1756):
        
        # Assigning a BinOp to a Name (line 1756):
        # Getting the type of 'rhs' (line 1756)
        rhs_164388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1756, 14), 'rhs')
        # Getting the type of 'w' (line 1756)
        w_164389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1756, 20), 'w')
        # Applying the binary operator '*' (line 1756)
        result_mul_164390 = python_operator(stypy.reporting.localization.Localization(__file__, 1756, 14), '*', rhs_164388, w_164389)
        
        # Assigning a type to the variable 'rhs' (line 1756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1756, 8), 'rhs', result_mul_164390)

        if more_types_in_union_164354:
            # SSA join for if statement (line 1747)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1759)
    # Getting the type of 'rcond' (line 1759)
    rcond_164391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 7), 'rcond')
    # Getting the type of 'None' (line 1759)
    None_164392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 16), 'None')
    
    (may_be_164393, more_types_in_union_164394) = may_be_none(rcond_164391, None_164392)

    if may_be_164393:

        if more_types_in_union_164394:
            # Runtime conditional SSA (line 1759)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1760):
        
        # Assigning a BinOp to a Name (line 1760):
        
        # Call to len(...): (line 1760)
        # Processing the call arguments (line 1760)
        # Getting the type of 'x' (line 1760)
        x_164396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 20), 'x', False)
        # Processing the call keyword arguments (line 1760)
        kwargs_164397 = {}
        # Getting the type of 'len' (line 1760)
        len_164395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 16), 'len', False)
        # Calling len(args, kwargs) (line 1760)
        len_call_result_164398 = invoke(stypy.reporting.localization.Localization(__file__, 1760, 16), len_164395, *[x_164396], **kwargs_164397)
        
        
        # Call to finfo(...): (line 1760)
        # Processing the call arguments (line 1760)
        # Getting the type of 'x' (line 1760)
        x_164401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 32), 'x', False)
        # Obtaining the member 'dtype' of a type (line 1760)
        dtype_164402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1760, 32), x_164401, 'dtype')
        # Processing the call keyword arguments (line 1760)
        kwargs_164403 = {}
        # Getting the type of 'np' (line 1760)
        np_164399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 23), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1760)
        finfo_164400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1760, 23), np_164399, 'finfo')
        # Calling finfo(args, kwargs) (line 1760)
        finfo_call_result_164404 = invoke(stypy.reporting.localization.Localization(__file__, 1760, 23), finfo_164400, *[dtype_164402], **kwargs_164403)
        
        # Obtaining the member 'eps' of a type (line 1760)
        eps_164405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1760, 23), finfo_call_result_164404, 'eps')
        # Applying the binary operator '*' (line 1760)
        result_mul_164406 = python_operator(stypy.reporting.localization.Localization(__file__, 1760, 16), '*', len_call_result_164398, eps_164405)
        
        # Assigning a type to the variable 'rcond' (line 1760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1760, 8), 'rcond', result_mul_164406)

        if more_types_in_union_164394:
            # SSA join for if statement (line 1759)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issubclass(...): (line 1763)
    # Processing the call arguments (line 1763)
    # Getting the type of 'lhs' (line 1763)
    lhs_164408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 18), 'lhs', False)
    # Obtaining the member 'dtype' of a type (line 1763)
    dtype_164409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1763, 18), lhs_164408, 'dtype')
    # Obtaining the member 'type' of a type (line 1763)
    type_164410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1763, 18), dtype_164409, 'type')
    # Getting the type of 'np' (line 1763)
    np_164411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 34), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1763)
    complexfloating_164412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1763, 34), np_164411, 'complexfloating')
    # Processing the call keyword arguments (line 1763)
    kwargs_164413 = {}
    # Getting the type of 'issubclass' (line 1763)
    issubclass_164407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1763)
    issubclass_call_result_164414 = invoke(stypy.reporting.localization.Localization(__file__, 1763, 7), issubclass_164407, *[type_164410, complexfloating_164412], **kwargs_164413)
    
    # Testing the type of an if condition (line 1763)
    if_condition_164415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1763, 4), issubclass_call_result_164414)
    # Assigning a type to the variable 'if_condition_164415' (line 1763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1763, 4), 'if_condition_164415', if_condition_164415)
    # SSA begins for if statement (line 1763)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1764):
    
    # Assigning a Call to a Name (line 1764):
    
    # Call to sqrt(...): (line 1764)
    # Processing the call arguments (line 1764)
    
    # Call to sum(...): (line 1764)
    # Processing the call arguments (line 1764)
    int_164432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1764, 70), 'int')
    # Processing the call keyword arguments (line 1764)
    kwargs_164433 = {}
    
    # Call to square(...): (line 1764)
    # Processing the call arguments (line 1764)
    # Getting the type of 'lhs' (line 1764)
    lhs_164420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 33), 'lhs', False)
    # Obtaining the member 'real' of a type (line 1764)
    real_164421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 33), lhs_164420, 'real')
    # Processing the call keyword arguments (line 1764)
    kwargs_164422 = {}
    # Getting the type of 'np' (line 1764)
    np_164418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 1764)
    square_164419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 23), np_164418, 'square')
    # Calling square(args, kwargs) (line 1764)
    square_call_result_164423 = invoke(stypy.reporting.localization.Localization(__file__, 1764, 23), square_164419, *[real_164421], **kwargs_164422)
    
    
    # Call to square(...): (line 1764)
    # Processing the call arguments (line 1764)
    # Getting the type of 'lhs' (line 1764)
    lhs_164426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 55), 'lhs', False)
    # Obtaining the member 'imag' of a type (line 1764)
    imag_164427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 55), lhs_164426, 'imag')
    # Processing the call keyword arguments (line 1764)
    kwargs_164428 = {}
    # Getting the type of 'np' (line 1764)
    np_164424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 45), 'np', False)
    # Obtaining the member 'square' of a type (line 1764)
    square_164425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 45), np_164424, 'square')
    # Calling square(args, kwargs) (line 1764)
    square_call_result_164429 = invoke(stypy.reporting.localization.Localization(__file__, 1764, 45), square_164425, *[imag_164427], **kwargs_164428)
    
    # Applying the binary operator '+' (line 1764)
    result_add_164430 = python_operator(stypy.reporting.localization.Localization(__file__, 1764, 23), '+', square_call_result_164423, square_call_result_164429)
    
    # Obtaining the member 'sum' of a type (line 1764)
    sum_164431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 23), result_add_164430, 'sum')
    # Calling sum(args, kwargs) (line 1764)
    sum_call_result_164434 = invoke(stypy.reporting.localization.Localization(__file__, 1764, 23), sum_164431, *[int_164432], **kwargs_164433)
    
    # Processing the call keyword arguments (line 1764)
    kwargs_164435 = {}
    # Getting the type of 'np' (line 1764)
    np_164416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1764)
    sqrt_164417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 14), np_164416, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1764)
    sqrt_call_result_164436 = invoke(stypy.reporting.localization.Localization(__file__, 1764, 14), sqrt_164417, *[sum_call_result_164434], **kwargs_164435)
    
    # Assigning a type to the variable 'scl' (line 1764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1764, 8), 'scl', sqrt_call_result_164436)
    # SSA branch for the else part of an if statement (line 1763)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1766):
    
    # Assigning a Call to a Name (line 1766):
    
    # Call to sqrt(...): (line 1766)
    # Processing the call arguments (line 1766)
    
    # Call to sum(...): (line 1766)
    # Processing the call arguments (line 1766)
    int_164445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1766, 41), 'int')
    # Processing the call keyword arguments (line 1766)
    kwargs_164446 = {}
    
    # Call to square(...): (line 1766)
    # Processing the call arguments (line 1766)
    # Getting the type of 'lhs' (line 1766)
    lhs_164441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1766, 32), 'lhs', False)
    # Processing the call keyword arguments (line 1766)
    kwargs_164442 = {}
    # Getting the type of 'np' (line 1766)
    np_164439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1766, 22), 'np', False)
    # Obtaining the member 'square' of a type (line 1766)
    square_164440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1766, 22), np_164439, 'square')
    # Calling square(args, kwargs) (line 1766)
    square_call_result_164443 = invoke(stypy.reporting.localization.Localization(__file__, 1766, 22), square_164440, *[lhs_164441], **kwargs_164442)
    
    # Obtaining the member 'sum' of a type (line 1766)
    sum_164444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1766, 22), square_call_result_164443, 'sum')
    # Calling sum(args, kwargs) (line 1766)
    sum_call_result_164447 = invoke(stypy.reporting.localization.Localization(__file__, 1766, 22), sum_164444, *[int_164445], **kwargs_164446)
    
    # Processing the call keyword arguments (line 1766)
    kwargs_164448 = {}
    # Getting the type of 'np' (line 1766)
    np_164437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1766, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1766)
    sqrt_164438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1766, 14), np_164437, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1766)
    sqrt_call_result_164449 = invoke(stypy.reporting.localization.Localization(__file__, 1766, 14), sqrt_164438, *[sum_call_result_164447], **kwargs_164448)
    
    # Assigning a type to the variable 'scl' (line 1766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1766, 8), 'scl', sqrt_call_result_164449)
    # SSA join for if statement (line 1763)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 1767):
    
    # Assigning a Num to a Subscript (line 1767):
    int_164450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1767, 20), 'int')
    # Getting the type of 'scl' (line 1767)
    scl_164451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 4), 'scl')
    
    # Getting the type of 'scl' (line 1767)
    scl_164452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 8), 'scl')
    int_164453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1767, 15), 'int')
    # Applying the binary operator '==' (line 1767)
    result_eq_164454 = python_operator(stypy.reporting.localization.Localization(__file__, 1767, 8), '==', scl_164452, int_164453)
    
    # Storing an element on a container (line 1767)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1767, 4), scl_164451, (result_eq_164454, int_164450))
    
    # Assigning a Call to a Tuple (line 1770):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 1770)
    # Processing the call arguments (line 1770)
    # Getting the type of 'lhs' (line 1770)
    lhs_164457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 34), 'lhs', False)
    # Obtaining the member 'T' of a type (line 1770)
    T_164458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 34), lhs_164457, 'T')
    # Getting the type of 'scl' (line 1770)
    scl_164459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 40), 'scl', False)
    # Applying the binary operator 'div' (line 1770)
    result_div_164460 = python_operator(stypy.reporting.localization.Localization(__file__, 1770, 34), 'div', T_164458, scl_164459)
    
    # Getting the type of 'rhs' (line 1770)
    rhs_164461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 45), 'rhs', False)
    # Obtaining the member 'T' of a type (line 1770)
    T_164462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 45), rhs_164461, 'T')
    # Getting the type of 'rcond' (line 1770)
    rcond_164463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 52), 'rcond', False)
    # Processing the call keyword arguments (line 1770)
    kwargs_164464 = {}
    # Getting the type of 'la' (line 1770)
    la_164455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 25), 'la', False)
    # Obtaining the member 'lstsq' of a type (line 1770)
    lstsq_164456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 25), la_164455, 'lstsq')
    # Calling lstsq(args, kwargs) (line 1770)
    lstsq_call_result_164465 = invoke(stypy.reporting.localization.Localization(__file__, 1770, 25), lstsq_164456, *[result_div_164460, T_164462, rcond_164463], **kwargs_164464)
    
    # Assigning a type to the variable 'call_assignment_161925' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161925', lstsq_call_result_164465)
    
    # Assigning a Call to a Name (line 1770):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_164468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1770, 4), 'int')
    # Processing the call keyword arguments
    kwargs_164469 = {}
    # Getting the type of 'call_assignment_161925' (line 1770)
    call_assignment_161925_164466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161925', False)
    # Obtaining the member '__getitem__' of a type (line 1770)
    getitem___164467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 4), call_assignment_161925_164466, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_164470 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___164467, *[int_164468], **kwargs_164469)
    
    # Assigning a type to the variable 'call_assignment_161926' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161926', getitem___call_result_164470)
    
    # Assigning a Name to a Name (line 1770):
    # Getting the type of 'call_assignment_161926' (line 1770)
    call_assignment_161926_164471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161926')
    # Assigning a type to the variable 'c' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'c', call_assignment_161926_164471)
    
    # Assigning a Call to a Name (line 1770):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_164474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1770, 4), 'int')
    # Processing the call keyword arguments
    kwargs_164475 = {}
    # Getting the type of 'call_assignment_161925' (line 1770)
    call_assignment_161925_164472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161925', False)
    # Obtaining the member '__getitem__' of a type (line 1770)
    getitem___164473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 4), call_assignment_161925_164472, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_164476 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___164473, *[int_164474], **kwargs_164475)
    
    # Assigning a type to the variable 'call_assignment_161927' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161927', getitem___call_result_164476)
    
    # Assigning a Name to a Name (line 1770):
    # Getting the type of 'call_assignment_161927' (line 1770)
    call_assignment_161927_164477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161927')
    # Assigning a type to the variable 'resids' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 7), 'resids', call_assignment_161927_164477)
    
    # Assigning a Call to a Name (line 1770):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_164480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1770, 4), 'int')
    # Processing the call keyword arguments
    kwargs_164481 = {}
    # Getting the type of 'call_assignment_161925' (line 1770)
    call_assignment_161925_164478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161925', False)
    # Obtaining the member '__getitem__' of a type (line 1770)
    getitem___164479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 4), call_assignment_161925_164478, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_164482 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___164479, *[int_164480], **kwargs_164481)
    
    # Assigning a type to the variable 'call_assignment_161928' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161928', getitem___call_result_164482)
    
    # Assigning a Name to a Name (line 1770):
    # Getting the type of 'call_assignment_161928' (line 1770)
    call_assignment_161928_164483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161928')
    # Assigning a type to the variable 'rank' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 15), 'rank', call_assignment_161928_164483)
    
    # Assigning a Call to a Name (line 1770):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_164486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1770, 4), 'int')
    # Processing the call keyword arguments
    kwargs_164487 = {}
    # Getting the type of 'call_assignment_161925' (line 1770)
    call_assignment_161925_164484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161925', False)
    # Obtaining the member '__getitem__' of a type (line 1770)
    getitem___164485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 4), call_assignment_161925_164484, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_164488 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___164485, *[int_164486], **kwargs_164487)
    
    # Assigning a type to the variable 'call_assignment_161929' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161929', getitem___call_result_164488)
    
    # Assigning a Name to a Name (line 1770):
    # Getting the type of 'call_assignment_161929' (line 1770)
    call_assignment_161929_164489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'call_assignment_161929')
    # Assigning a type to the variable 's' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 21), 's', call_assignment_161929_164489)
    
    # Assigning a Attribute to a Name (line 1771):
    
    # Assigning a Attribute to a Name (line 1771):
    # Getting the type of 'c' (line 1771)
    c_164490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1771, 9), 'c')
    # Obtaining the member 'T' of a type (line 1771)
    T_164491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1771, 9), c_164490, 'T')
    # Getting the type of 'scl' (line 1771)
    scl_164492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1771, 13), 'scl')
    # Applying the binary operator 'div' (line 1771)
    result_div_164493 = python_operator(stypy.reporting.localization.Localization(__file__, 1771, 9), 'div', T_164491, scl_164492)
    
    # Obtaining the member 'T' of a type (line 1771)
    T_164494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1771, 9), result_div_164493, 'T')
    # Assigning a type to the variable 'c' (line 1771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1771, 4), 'c', T_164494)
    
    
    # Getting the type of 'deg' (line 1774)
    deg_164495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1774, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1774)
    ndim_164496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1774, 7), deg_164495, 'ndim')
    int_164497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1774, 18), 'int')
    # Applying the binary operator '>' (line 1774)
    result_gt_164498 = python_operator(stypy.reporting.localization.Localization(__file__, 1774, 7), '>', ndim_164496, int_164497)
    
    # Testing the type of an if condition (line 1774)
    if_condition_164499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1774, 4), result_gt_164498)
    # Assigning a type to the variable 'if_condition_164499' (line 1774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1774, 4), 'if_condition_164499', if_condition_164499)
    # SSA begins for if statement (line 1774)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'c' (line 1775)
    c_164500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1775, 11), 'c')
    # Obtaining the member 'ndim' of a type (line 1775)
    ndim_164501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1775, 11), c_164500, 'ndim')
    int_164502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1775, 21), 'int')
    # Applying the binary operator '==' (line 1775)
    result_eq_164503 = python_operator(stypy.reporting.localization.Localization(__file__, 1775, 11), '==', ndim_164501, int_164502)
    
    # Testing the type of an if condition (line 1775)
    if_condition_164504 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1775, 8), result_eq_164503)
    # Assigning a type to the variable 'if_condition_164504' (line 1775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1775, 8), 'if_condition_164504', if_condition_164504)
    # SSA begins for if statement (line 1775)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1776):
    
    # Assigning a Call to a Name (line 1776):
    
    # Call to zeros(...): (line 1776)
    # Processing the call arguments (line 1776)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1776)
    tuple_164507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1776, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1776)
    # Adding element type (line 1776)
    # Getting the type of 'lmax' (line 1776)
    lmax_164508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 27), 'lmax', False)
    int_164509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1776, 34), 'int')
    # Applying the binary operator '+' (line 1776)
    result_add_164510 = python_operator(stypy.reporting.localization.Localization(__file__, 1776, 27), '+', lmax_164508, int_164509)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1776, 27), tuple_164507, result_add_164510)
    # Adding element type (line 1776)
    
    # Obtaining the type of the subscript
    int_164511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1776, 45), 'int')
    # Getting the type of 'c' (line 1776)
    c_164512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 37), 'c', False)
    # Obtaining the member 'shape' of a type (line 1776)
    shape_164513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 37), c_164512, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1776)
    getitem___164514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 37), shape_164513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1776)
    subscript_call_result_164515 = invoke(stypy.reporting.localization.Localization(__file__, 1776, 37), getitem___164514, int_164511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1776, 27), tuple_164507, subscript_call_result_164515)
    
    # Processing the call keyword arguments (line 1776)
    # Getting the type of 'c' (line 1776)
    c_164516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 56), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1776)
    dtype_164517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 56), c_164516, 'dtype')
    keyword_164518 = dtype_164517
    kwargs_164519 = {'dtype': keyword_164518}
    # Getting the type of 'np' (line 1776)
    np_164505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1776)
    zeros_164506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 17), np_164505, 'zeros')
    # Calling zeros(args, kwargs) (line 1776)
    zeros_call_result_164520 = invoke(stypy.reporting.localization.Localization(__file__, 1776, 17), zeros_164506, *[tuple_164507], **kwargs_164519)
    
    # Assigning a type to the variable 'cc' (line 1776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1776, 12), 'cc', zeros_call_result_164520)
    # SSA branch for the else part of an if statement (line 1775)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1778):
    
    # Assigning a Call to a Name (line 1778):
    
    # Call to zeros(...): (line 1778)
    # Processing the call arguments (line 1778)
    # Getting the type of 'lmax' (line 1778)
    lmax_164523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 26), 'lmax', False)
    int_164524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1778, 33), 'int')
    # Applying the binary operator '+' (line 1778)
    result_add_164525 = python_operator(stypy.reporting.localization.Localization(__file__, 1778, 26), '+', lmax_164523, int_164524)
    
    # Processing the call keyword arguments (line 1778)
    # Getting the type of 'c' (line 1778)
    c_164526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 42), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1778)
    dtype_164527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1778, 42), c_164526, 'dtype')
    keyword_164528 = dtype_164527
    kwargs_164529 = {'dtype': keyword_164528}
    # Getting the type of 'np' (line 1778)
    np_164521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1778)
    zeros_164522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1778, 17), np_164521, 'zeros')
    # Calling zeros(args, kwargs) (line 1778)
    zeros_call_result_164530 = invoke(stypy.reporting.localization.Localization(__file__, 1778, 17), zeros_164522, *[result_add_164525], **kwargs_164529)
    
    # Assigning a type to the variable 'cc' (line 1778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1778, 12), 'cc', zeros_call_result_164530)
    # SSA join for if statement (line 1775)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1779):
    
    # Assigning a Name to a Subscript (line 1779):
    # Getting the type of 'c' (line 1779)
    c_164531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1779, 18), 'c')
    # Getting the type of 'cc' (line 1779)
    cc_164532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1779, 8), 'cc')
    # Getting the type of 'deg' (line 1779)
    deg_164533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1779, 11), 'deg')
    # Storing an element on a container (line 1779)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1779, 8), cc_164532, (deg_164533, c_164531))
    
    # Assigning a Name to a Name (line 1780):
    
    # Assigning a Name to a Name (line 1780):
    # Getting the type of 'cc' (line 1780)
    cc_164534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1780, 12), 'cc')
    # Assigning a type to the variable 'c' (line 1780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1780, 8), 'c', cc_164534)
    # SSA join for if statement (line 1774)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1783)
    rank_164535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1783, 7), 'rank')
    # Getting the type of 'order' (line 1783)
    order_164536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1783, 15), 'order')
    # Applying the binary operator '!=' (line 1783)
    result_ne_164537 = python_operator(stypy.reporting.localization.Localization(__file__, 1783, 7), '!=', rank_164535, order_164536)
    
    
    # Getting the type of 'full' (line 1783)
    full_164538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1783, 29), 'full')
    # Applying the 'not' unary operator (line 1783)
    result_not__164539 = python_operator(stypy.reporting.localization.Localization(__file__, 1783, 25), 'not', full_164538)
    
    # Applying the binary operator 'and' (line 1783)
    result_and_keyword_164540 = python_operator(stypy.reporting.localization.Localization(__file__, 1783, 7), 'and', result_ne_164537, result_not__164539)
    
    # Testing the type of an if condition (line 1783)
    if_condition_164541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1783, 4), result_and_keyword_164540)
    # Assigning a type to the variable 'if_condition_164541' (line 1783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1783, 4), 'if_condition_164541', if_condition_164541)
    # SSA begins for if statement (line 1783)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1784):
    
    # Assigning a Str to a Name (line 1784):
    str_164542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1784, 14), 'str', 'The fit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 1784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1784, 8), 'msg', str_164542)
    
    # Call to warn(...): (line 1785)
    # Processing the call arguments (line 1785)
    # Getting the type of 'msg' (line 1785)
    msg_164545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1785, 22), 'msg', False)
    # Getting the type of 'pu' (line 1785)
    pu_164546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1785, 27), 'pu', False)
    # Obtaining the member 'RankWarning' of a type (line 1785)
    RankWarning_164547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1785, 27), pu_164546, 'RankWarning')
    # Processing the call keyword arguments (line 1785)
    kwargs_164548 = {}
    # Getting the type of 'warnings' (line 1785)
    warnings_164543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1785, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1785)
    warn_164544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1785, 8), warnings_164543, 'warn')
    # Calling warn(args, kwargs) (line 1785)
    warn_call_result_164549 = invoke(stypy.reporting.localization.Localization(__file__, 1785, 8), warn_164544, *[msg_164545, RankWarning_164547], **kwargs_164548)
    
    # SSA join for if statement (line 1783)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 1787)
    full_164550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1787, 7), 'full')
    # Testing the type of an if condition (line 1787)
    if_condition_164551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1787, 4), full_164550)
    # Assigning a type to the variable 'if_condition_164551' (line 1787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1787, 4), 'if_condition_164551', if_condition_164551)
    # SSA begins for if statement (line 1787)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1788)
    tuple_164552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1788, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1788)
    # Adding element type (line 1788)
    # Getting the type of 'c' (line 1788)
    c_164553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1788, 15), tuple_164552, c_164553)
    # Adding element type (line 1788)
    
    # Obtaining an instance of the builtin type 'list' (line 1788)
    list_164554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1788, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1788)
    # Adding element type (line 1788)
    # Getting the type of 'resids' (line 1788)
    resids_164555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 19), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1788, 18), list_164554, resids_164555)
    # Adding element type (line 1788)
    # Getting the type of 'rank' (line 1788)
    rank_164556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 27), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1788, 18), list_164554, rank_164556)
    # Adding element type (line 1788)
    # Getting the type of 's' (line 1788)
    s_164557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 33), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1788, 18), list_164554, s_164557)
    # Adding element type (line 1788)
    # Getting the type of 'rcond' (line 1788)
    rcond_164558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1788, 36), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1788, 18), list_164554, rcond_164558)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1788, 15), tuple_164552, list_164554)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1788, 8), 'stypy_return_type', tuple_164552)
    # SSA branch for the else part of an if statement (line 1787)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 1790)
    c_164559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1790, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1790, 8), 'stypy_return_type', c_164559)
    # SSA join for if statement (line 1787)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'chebfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebfit' in the type store
    # Getting the type of 'stypy_return_type' (line 1597)
    stypy_return_type_164560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164560)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebfit'
    return stypy_return_type_164560

# Assigning a type to the variable 'chebfit' (line 1597)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1597, 0), 'chebfit', chebfit)

@norecursion
def chebcompanion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebcompanion'
    module_type_store = module_type_store.open_function_context('chebcompanion', 1793, 0, False)
    
    # Passed parameters checking function
    chebcompanion.stypy_localization = localization
    chebcompanion.stypy_type_of_self = None
    chebcompanion.stypy_type_store = module_type_store
    chebcompanion.stypy_function_name = 'chebcompanion'
    chebcompanion.stypy_param_names_list = ['c']
    chebcompanion.stypy_varargs_param_name = None
    chebcompanion.stypy_kwargs_param_name = None
    chebcompanion.stypy_call_defaults = defaults
    chebcompanion.stypy_call_varargs = varargs
    chebcompanion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebcompanion', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebcompanion', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebcompanion(...)' code ##################

    str_164561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1818, (-1)), 'str', 'Return the scaled companion matrix of c.\n\n    The basis polynomials are scaled so that the companion matrix is\n    symmetric when `c` is a Chebyshev basis polynomial. This provides\n    better eigenvalue estimates than the unscaled case and for basis\n    polynomials the eigenvalues are guaranteed to be real if\n    `numpy.linalg.eigvalsh` is used to obtain them.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Chebyshev series coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Scaled companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a List (line 1820):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1820)
    # Processing the call arguments (line 1820)
    
    # Obtaining an instance of the builtin type 'list' (line 1820)
    list_164564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1820, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1820)
    # Adding element type (line 1820)
    # Getting the type of 'c' (line 1820)
    c_164565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1820, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1820, 23), list_164564, c_164565)
    
    # Processing the call keyword arguments (line 1820)
    kwargs_164566 = {}
    # Getting the type of 'pu' (line 1820)
    pu_164562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1820, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1820)
    as_series_164563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1820, 10), pu_164562, 'as_series')
    # Calling as_series(args, kwargs) (line 1820)
    as_series_call_result_164567 = invoke(stypy.reporting.localization.Localization(__file__, 1820, 10), as_series_164563, *[list_164564], **kwargs_164566)
    
    # Assigning a type to the variable 'call_assignment_161930' (line 1820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1820, 4), 'call_assignment_161930', as_series_call_result_164567)
    
    # Assigning a Call to a Name (line 1820):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_164570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1820, 4), 'int')
    # Processing the call keyword arguments
    kwargs_164571 = {}
    # Getting the type of 'call_assignment_161930' (line 1820)
    call_assignment_161930_164568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1820, 4), 'call_assignment_161930', False)
    # Obtaining the member '__getitem__' of a type (line 1820)
    getitem___164569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1820, 4), call_assignment_161930_164568, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_164572 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___164569, *[int_164570], **kwargs_164571)
    
    # Assigning a type to the variable 'call_assignment_161931' (line 1820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1820, 4), 'call_assignment_161931', getitem___call_result_164572)
    
    # Assigning a Name to a Name (line 1820):
    # Getting the type of 'call_assignment_161931' (line 1820)
    call_assignment_161931_164573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1820, 4), 'call_assignment_161931')
    # Assigning a type to the variable 'c' (line 1820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1820, 5), 'c', call_assignment_161931_164573)
    
    
    
    # Call to len(...): (line 1821)
    # Processing the call arguments (line 1821)
    # Getting the type of 'c' (line 1821)
    c_164575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1821, 11), 'c', False)
    # Processing the call keyword arguments (line 1821)
    kwargs_164576 = {}
    # Getting the type of 'len' (line 1821)
    len_164574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1821, 7), 'len', False)
    # Calling len(args, kwargs) (line 1821)
    len_call_result_164577 = invoke(stypy.reporting.localization.Localization(__file__, 1821, 7), len_164574, *[c_164575], **kwargs_164576)
    
    int_164578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1821, 16), 'int')
    # Applying the binary operator '<' (line 1821)
    result_lt_164579 = python_operator(stypy.reporting.localization.Localization(__file__, 1821, 7), '<', len_call_result_164577, int_164578)
    
    # Testing the type of an if condition (line 1821)
    if_condition_164580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1821, 4), result_lt_164579)
    # Assigning a type to the variable 'if_condition_164580' (line 1821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1821, 4), 'if_condition_164580', if_condition_164580)
    # SSA begins for if statement (line 1821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1822)
    # Processing the call arguments (line 1822)
    str_164582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1822, 25), 'str', 'Series must have maximum degree of at least 1.')
    # Processing the call keyword arguments (line 1822)
    kwargs_164583 = {}
    # Getting the type of 'ValueError' (line 1822)
    ValueError_164581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1822, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1822)
    ValueError_call_result_164584 = invoke(stypy.reporting.localization.Localization(__file__, 1822, 14), ValueError_164581, *[str_164582], **kwargs_164583)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1822, 8), ValueError_call_result_164584, 'raise parameter', BaseException)
    # SSA join for if statement (line 1821)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1823)
    # Processing the call arguments (line 1823)
    # Getting the type of 'c' (line 1823)
    c_164586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1823, 11), 'c', False)
    # Processing the call keyword arguments (line 1823)
    kwargs_164587 = {}
    # Getting the type of 'len' (line 1823)
    len_164585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1823, 7), 'len', False)
    # Calling len(args, kwargs) (line 1823)
    len_call_result_164588 = invoke(stypy.reporting.localization.Localization(__file__, 1823, 7), len_164585, *[c_164586], **kwargs_164587)
    
    int_164589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1823, 17), 'int')
    # Applying the binary operator '==' (line 1823)
    result_eq_164590 = python_operator(stypy.reporting.localization.Localization(__file__, 1823, 7), '==', len_call_result_164588, int_164589)
    
    # Testing the type of an if condition (line 1823)
    if_condition_164591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1823, 4), result_eq_164590)
    # Assigning a type to the variable 'if_condition_164591' (line 1823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1823, 4), 'if_condition_164591', if_condition_164591)
    # SSA begins for if statement (line 1823)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1824)
    # Processing the call arguments (line 1824)
    
    # Obtaining an instance of the builtin type 'list' (line 1824)
    list_164594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1824, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1824)
    # Adding element type (line 1824)
    
    # Obtaining an instance of the builtin type 'list' (line 1824)
    list_164595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1824, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1824)
    # Adding element type (line 1824)
    
    
    # Obtaining the type of the subscript
    int_164596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1824, 29), 'int')
    # Getting the type of 'c' (line 1824)
    c_164597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 27), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1824)
    getitem___164598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1824, 27), c_164597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1824)
    subscript_call_result_164599 = invoke(stypy.reporting.localization.Localization(__file__, 1824, 27), getitem___164598, int_164596)
    
    # Applying the 'usub' unary operator (line 1824)
    result___neg___164600 = python_operator(stypy.reporting.localization.Localization(__file__, 1824, 26), 'usub', subscript_call_result_164599)
    
    
    # Obtaining the type of the subscript
    int_164601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1824, 34), 'int')
    # Getting the type of 'c' (line 1824)
    c_164602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 32), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1824)
    getitem___164603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1824, 32), c_164602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1824)
    subscript_call_result_164604 = invoke(stypy.reporting.localization.Localization(__file__, 1824, 32), getitem___164603, int_164601)
    
    # Applying the binary operator 'div' (line 1824)
    result_div_164605 = python_operator(stypy.reporting.localization.Localization(__file__, 1824, 26), 'div', result___neg___164600, subscript_call_result_164604)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1824, 25), list_164595, result_div_164605)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1824, 24), list_164594, list_164595)
    
    # Processing the call keyword arguments (line 1824)
    kwargs_164606 = {}
    # Getting the type of 'np' (line 1824)
    np_164592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1824)
    array_164593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1824, 15), np_164592, 'array')
    # Calling array(args, kwargs) (line 1824)
    array_call_result_164607 = invoke(stypy.reporting.localization.Localization(__file__, 1824, 15), array_164593, *[list_164594], **kwargs_164606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1824, 8), 'stypy_return_type', array_call_result_164607)
    # SSA join for if statement (line 1823)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1826):
    
    # Assigning a BinOp to a Name (line 1826):
    
    # Call to len(...): (line 1826)
    # Processing the call arguments (line 1826)
    # Getting the type of 'c' (line 1826)
    c_164609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 12), 'c', False)
    # Processing the call keyword arguments (line 1826)
    kwargs_164610 = {}
    # Getting the type of 'len' (line 1826)
    len_164608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 8), 'len', False)
    # Calling len(args, kwargs) (line 1826)
    len_call_result_164611 = invoke(stypy.reporting.localization.Localization(__file__, 1826, 8), len_164608, *[c_164609], **kwargs_164610)
    
    int_164612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1826, 17), 'int')
    # Applying the binary operator '-' (line 1826)
    result_sub_164613 = python_operator(stypy.reporting.localization.Localization(__file__, 1826, 8), '-', len_call_result_164611, int_164612)
    
    # Assigning a type to the variable 'n' (line 1826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1826, 4), 'n', result_sub_164613)
    
    # Assigning a Call to a Name (line 1827):
    
    # Assigning a Call to a Name (line 1827):
    
    # Call to zeros(...): (line 1827)
    # Processing the call arguments (line 1827)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1827)
    tuple_164616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1827, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1827)
    # Adding element type (line 1827)
    # Getting the type of 'n' (line 1827)
    n_164617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1827, 20), tuple_164616, n_164617)
    # Adding element type (line 1827)
    # Getting the type of 'n' (line 1827)
    n_164618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1827, 20), tuple_164616, n_164618)
    
    # Processing the call keyword arguments (line 1827)
    # Getting the type of 'c' (line 1827)
    c_164619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 33), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1827)
    dtype_164620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1827, 33), c_164619, 'dtype')
    keyword_164621 = dtype_164620
    kwargs_164622 = {'dtype': keyword_164621}
    # Getting the type of 'np' (line 1827)
    np_164614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1827)
    zeros_164615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1827, 10), np_164614, 'zeros')
    # Calling zeros(args, kwargs) (line 1827)
    zeros_call_result_164623 = invoke(stypy.reporting.localization.Localization(__file__, 1827, 10), zeros_164615, *[tuple_164616], **kwargs_164622)
    
    # Assigning a type to the variable 'mat' (line 1827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1827, 4), 'mat', zeros_call_result_164623)
    
    # Assigning a Call to a Name (line 1828):
    
    # Assigning a Call to a Name (line 1828):
    
    # Call to array(...): (line 1828)
    # Processing the call arguments (line 1828)
    
    # Obtaining an instance of the builtin type 'list' (line 1828)
    list_164626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1828, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1828)
    # Adding element type (line 1828)
    float_164627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1828, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1828, 19), list_164626, float_164627)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1828)
    list_164628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1828, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1828)
    # Adding element type (line 1828)
    
    # Call to sqrt(...): (line 1828)
    # Processing the call arguments (line 1828)
    float_164631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1828, 35), 'float')
    # Processing the call keyword arguments (line 1828)
    kwargs_164632 = {}
    # Getting the type of 'np' (line 1828)
    np_164629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 27), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1828)
    sqrt_164630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1828, 27), np_164629, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1828)
    sqrt_call_result_164633 = invoke(stypy.reporting.localization.Localization(__file__, 1828, 27), sqrt_164630, *[float_164631], **kwargs_164632)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1828, 26), list_164628, sqrt_call_result_164633)
    
    # Getting the type of 'n' (line 1828)
    n_164634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 41), 'n', False)
    int_164635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1828, 43), 'int')
    # Applying the binary operator '-' (line 1828)
    result_sub_164636 = python_operator(stypy.reporting.localization.Localization(__file__, 1828, 41), '-', n_164634, int_164635)
    
    # Applying the binary operator '*' (line 1828)
    result_mul_164637 = python_operator(stypy.reporting.localization.Localization(__file__, 1828, 26), '*', list_164628, result_sub_164636)
    
    # Applying the binary operator '+' (line 1828)
    result_add_164638 = python_operator(stypy.reporting.localization.Localization(__file__, 1828, 19), '+', list_164626, result_mul_164637)
    
    # Processing the call keyword arguments (line 1828)
    kwargs_164639 = {}
    # Getting the type of 'np' (line 1828)
    np_164624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 1828)
    array_164625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1828, 10), np_164624, 'array')
    # Calling array(args, kwargs) (line 1828)
    array_call_result_164640 = invoke(stypy.reporting.localization.Localization(__file__, 1828, 10), array_164625, *[result_add_164638], **kwargs_164639)
    
    # Assigning a type to the variable 'scl' (line 1828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1828, 4), 'scl', array_call_result_164640)
    
    # Assigning a Subscript to a Name (line 1829):
    
    # Assigning a Subscript to a Name (line 1829):
    
    # Obtaining the type of the subscript
    int_164641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1829, 26), 'int')
    # Getting the type of 'n' (line 1829)
    n_164642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1829, 29), 'n')
    int_164643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1829, 31), 'int')
    # Applying the binary operator '+' (line 1829)
    result_add_164644 = python_operator(stypy.reporting.localization.Localization(__file__, 1829, 29), '+', n_164642, int_164643)
    
    slice_164645 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1829, 10), int_164641, None, result_add_164644)
    
    # Call to reshape(...): (line 1829)
    # Processing the call arguments (line 1829)
    int_164648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1829, 22), 'int')
    # Processing the call keyword arguments (line 1829)
    kwargs_164649 = {}
    # Getting the type of 'mat' (line 1829)
    mat_164646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1829, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1829)
    reshape_164647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1829, 10), mat_164646, 'reshape')
    # Calling reshape(args, kwargs) (line 1829)
    reshape_call_result_164650 = invoke(stypy.reporting.localization.Localization(__file__, 1829, 10), reshape_164647, *[int_164648], **kwargs_164649)
    
    # Obtaining the member '__getitem__' of a type (line 1829)
    getitem___164651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1829, 10), reshape_call_result_164650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1829)
    subscript_call_result_164652 = invoke(stypy.reporting.localization.Localization(__file__, 1829, 10), getitem___164651, slice_164645)
    
    # Assigning a type to the variable 'top' (line 1829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1829, 4), 'top', subscript_call_result_164652)
    
    # Assigning a Subscript to a Name (line 1830):
    
    # Assigning a Subscript to a Name (line 1830):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1830)
    n_164653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 26), 'n')
    # Getting the type of 'n' (line 1830)
    n_164654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 29), 'n')
    int_164655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1830, 31), 'int')
    # Applying the binary operator '+' (line 1830)
    result_add_164656 = python_operator(stypy.reporting.localization.Localization(__file__, 1830, 29), '+', n_164654, int_164655)
    
    slice_164657 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1830, 10), n_164653, None, result_add_164656)
    
    # Call to reshape(...): (line 1830)
    # Processing the call arguments (line 1830)
    int_164660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1830, 22), 'int')
    # Processing the call keyword arguments (line 1830)
    kwargs_164661 = {}
    # Getting the type of 'mat' (line 1830)
    mat_164658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1830)
    reshape_164659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1830, 10), mat_164658, 'reshape')
    # Calling reshape(args, kwargs) (line 1830)
    reshape_call_result_164662 = invoke(stypy.reporting.localization.Localization(__file__, 1830, 10), reshape_164659, *[int_164660], **kwargs_164661)
    
    # Obtaining the member '__getitem__' of a type (line 1830)
    getitem___164663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1830, 10), reshape_call_result_164662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1830)
    subscript_call_result_164664 = invoke(stypy.reporting.localization.Localization(__file__, 1830, 10), getitem___164663, slice_164657)
    
    # Assigning a type to the variable 'bot' (line 1830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1830, 4), 'bot', subscript_call_result_164664)
    
    # Assigning a Call to a Subscript (line 1831):
    
    # Assigning a Call to a Subscript (line 1831):
    
    # Call to sqrt(...): (line 1831)
    # Processing the call arguments (line 1831)
    float_164667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1831, 21), 'float')
    # Processing the call keyword arguments (line 1831)
    kwargs_164668 = {}
    # Getting the type of 'np' (line 1831)
    np_164665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1831, 13), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1831)
    sqrt_164666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1831, 13), np_164665, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1831)
    sqrt_call_result_164669 = invoke(stypy.reporting.localization.Localization(__file__, 1831, 13), sqrt_164666, *[float_164667], **kwargs_164668)
    
    # Getting the type of 'top' (line 1831)
    top_164670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1831, 4), 'top')
    int_164671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1831, 8), 'int')
    # Storing an element on a container (line 1831)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1831, 4), top_164670, (int_164671, sqrt_call_result_164669))
    
    # Assigning a BinOp to a Subscript (line 1832):
    
    # Assigning a BinOp to a Subscript (line 1832):
    int_164672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1832, 14), 'int')
    int_164673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1832, 16), 'int')
    # Applying the binary operator 'div' (line 1832)
    result_div_164674 = python_operator(stypy.reporting.localization.Localization(__file__, 1832, 14), 'div', int_164672, int_164673)
    
    # Getting the type of 'top' (line 1832)
    top_164675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1832, 4), 'top')
    int_164676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1832, 8), 'int')
    slice_164677 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1832, 4), int_164676, None, None)
    # Storing an element on a container (line 1832)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1832, 4), top_164675, (slice_164677, result_div_164674))
    
    # Assigning a Name to a Subscript (line 1833):
    
    # Assigning a Name to a Subscript (line 1833):
    # Getting the type of 'top' (line 1833)
    top_164678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 15), 'top')
    # Getting the type of 'bot' (line 1833)
    bot_164679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 4), 'bot')
    Ellipsis_164680 = Ellipsis
    # Storing an element on a container (line 1833)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1833, 4), bot_164679, (Ellipsis_164680, top_164678))
    
    # Getting the type of 'mat' (line 1834)
    mat_164681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 4), 'mat')
    
    # Obtaining the type of the subscript
    slice_164682 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1834, 4), None, None, None)
    int_164683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 11), 'int')
    # Getting the type of 'mat' (line 1834)
    mat_164684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 4), 'mat')
    # Obtaining the member '__getitem__' of a type (line 1834)
    getitem___164685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 4), mat_164684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1834)
    subscript_call_result_164686 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 4), getitem___164685, (slice_164682, int_164683))
    
    
    # Obtaining the type of the subscript
    int_164687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 22), 'int')
    slice_164688 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1834, 19), None, int_164687, None)
    # Getting the type of 'c' (line 1834)
    c_164689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 1834)
    getitem___164690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 19), c_164689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1834)
    subscript_call_result_164691 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 19), getitem___164690, slice_164688)
    
    
    # Obtaining the type of the subscript
    int_164692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 28), 'int')
    # Getting the type of 'c' (line 1834)
    c_164693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 26), 'c')
    # Obtaining the member '__getitem__' of a type (line 1834)
    getitem___164694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 26), c_164693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1834)
    subscript_call_result_164695 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 26), getitem___164694, int_164692)
    
    # Applying the binary operator 'div' (line 1834)
    result_div_164696 = python_operator(stypy.reporting.localization.Localization(__file__, 1834, 19), 'div', subscript_call_result_164691, subscript_call_result_164695)
    
    # Getting the type of 'scl' (line 1834)
    scl_164697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 34), 'scl')
    
    # Obtaining the type of the subscript
    int_164698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 42), 'int')
    # Getting the type of 'scl' (line 1834)
    scl_164699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 38), 'scl')
    # Obtaining the member '__getitem__' of a type (line 1834)
    getitem___164700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 38), scl_164699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1834)
    subscript_call_result_164701 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 38), getitem___164700, int_164698)
    
    # Applying the binary operator 'div' (line 1834)
    result_div_164702 = python_operator(stypy.reporting.localization.Localization(__file__, 1834, 34), 'div', scl_164697, subscript_call_result_164701)
    
    # Applying the binary operator '*' (line 1834)
    result_mul_164703 = python_operator(stypy.reporting.localization.Localization(__file__, 1834, 18), '*', result_div_164696, result_div_164702)
    
    float_164704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 47), 'float')
    # Applying the binary operator '*' (line 1834)
    result_mul_164705 = python_operator(stypy.reporting.localization.Localization(__file__, 1834, 46), '*', result_mul_164703, float_164704)
    
    # Applying the binary operator '-=' (line 1834)
    result_isub_164706 = python_operator(stypy.reporting.localization.Localization(__file__, 1834, 4), '-=', subscript_call_result_164686, result_mul_164705)
    # Getting the type of 'mat' (line 1834)
    mat_164707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 4), 'mat')
    slice_164708 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1834, 4), None, None, None)
    int_164709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 11), 'int')
    # Storing an element on a container (line 1834)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1834, 4), mat_164707, ((slice_164708, int_164709), result_isub_164706))
    
    # Getting the type of 'mat' (line 1835)
    mat_164710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 11), 'mat')
    # Assigning a type to the variable 'stypy_return_type' (line 1835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1835, 4), 'stypy_return_type', mat_164710)
    
    # ################# End of 'chebcompanion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebcompanion' in the type store
    # Getting the type of 'stypy_return_type' (line 1793)
    stypy_return_type_164711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1793, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164711)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebcompanion'
    return stypy_return_type_164711

# Assigning a type to the variable 'chebcompanion' (line 1793)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1793, 0), 'chebcompanion', chebcompanion)

@norecursion
def chebroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebroots'
    module_type_store = module_type_store.open_function_context('chebroots', 1838, 0, False)
    
    # Passed parameters checking function
    chebroots.stypy_localization = localization
    chebroots.stypy_type_of_self = None
    chebroots.stypy_type_store = module_type_store
    chebroots.stypy_function_name = 'chebroots'
    chebroots.stypy_param_names_list = ['c']
    chebroots.stypy_varargs_param_name = None
    chebroots.stypy_kwargs_param_name = None
    chebroots.stypy_call_defaults = defaults
    chebroots.stypy_call_varargs = varargs
    chebroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebroots', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebroots', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebroots(...)' code ##################

    str_164712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1880, (-1)), 'str', '\n    Compute the roots of a Chebyshev series.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * T_i(x).\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the series. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    polyroots, legroots, lagroots, hermroots, hermeroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the series for such\n    values. Roots with multiplicity greater than 1 will also show larger\n    errors as the value of the series near such points is relatively\n    insensitive to errors in the roots. Isolated roots near the origin can\n    be improved by a few iterations of Newton\'s method.\n\n    The Chebyshev series basis polynomials aren\'t powers of `x` so the\n    results of this function may seem unintuitive.\n\n    Examples\n    --------\n    >>> import numpy.polynomial.chebyshev as cheb\n    >>> cheb.chebroots((-1, 1,-1, 1)) # T3 - T2 + T1 - T0 has real roots\n    array([ -5.00000000e-01,   2.60860684e-17,   1.00000000e+00])\n\n    ')
    
    # Assigning a Call to a List (line 1882):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1882)
    # Processing the call arguments (line 1882)
    
    # Obtaining an instance of the builtin type 'list' (line 1882)
    list_164715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1882, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1882)
    # Adding element type (line 1882)
    # Getting the type of 'c' (line 1882)
    c_164716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1882, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1882, 23), list_164715, c_164716)
    
    # Processing the call keyword arguments (line 1882)
    kwargs_164717 = {}
    # Getting the type of 'pu' (line 1882)
    pu_164713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1882, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1882)
    as_series_164714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1882, 10), pu_164713, 'as_series')
    # Calling as_series(args, kwargs) (line 1882)
    as_series_call_result_164718 = invoke(stypy.reporting.localization.Localization(__file__, 1882, 10), as_series_164714, *[list_164715], **kwargs_164717)
    
    # Assigning a type to the variable 'call_assignment_161932' (line 1882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1882, 4), 'call_assignment_161932', as_series_call_result_164718)
    
    # Assigning a Call to a Name (line 1882):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_164721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1882, 4), 'int')
    # Processing the call keyword arguments
    kwargs_164722 = {}
    # Getting the type of 'call_assignment_161932' (line 1882)
    call_assignment_161932_164719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1882, 4), 'call_assignment_161932', False)
    # Obtaining the member '__getitem__' of a type (line 1882)
    getitem___164720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1882, 4), call_assignment_161932_164719, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_164723 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___164720, *[int_164721], **kwargs_164722)
    
    # Assigning a type to the variable 'call_assignment_161933' (line 1882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1882, 4), 'call_assignment_161933', getitem___call_result_164723)
    
    # Assigning a Name to a Name (line 1882):
    # Getting the type of 'call_assignment_161933' (line 1882)
    call_assignment_161933_164724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1882, 4), 'call_assignment_161933')
    # Assigning a type to the variable 'c' (line 1882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1882, 5), 'c', call_assignment_161933_164724)
    
    
    
    # Call to len(...): (line 1883)
    # Processing the call arguments (line 1883)
    # Getting the type of 'c' (line 1883)
    c_164726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1883, 11), 'c', False)
    # Processing the call keyword arguments (line 1883)
    kwargs_164727 = {}
    # Getting the type of 'len' (line 1883)
    len_164725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1883, 7), 'len', False)
    # Calling len(args, kwargs) (line 1883)
    len_call_result_164728 = invoke(stypy.reporting.localization.Localization(__file__, 1883, 7), len_164725, *[c_164726], **kwargs_164727)
    
    int_164729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1883, 16), 'int')
    # Applying the binary operator '<' (line 1883)
    result_lt_164730 = python_operator(stypy.reporting.localization.Localization(__file__, 1883, 7), '<', len_call_result_164728, int_164729)
    
    # Testing the type of an if condition (line 1883)
    if_condition_164731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1883, 4), result_lt_164730)
    # Assigning a type to the variable 'if_condition_164731' (line 1883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1883, 4), 'if_condition_164731', if_condition_164731)
    # SSA begins for if statement (line 1883)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1884)
    # Processing the call arguments (line 1884)
    
    # Obtaining an instance of the builtin type 'list' (line 1884)
    list_164734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1884, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1884)
    
    # Processing the call keyword arguments (line 1884)
    # Getting the type of 'c' (line 1884)
    c_164735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1884, 34), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1884)
    dtype_164736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1884, 34), c_164735, 'dtype')
    keyword_164737 = dtype_164736
    kwargs_164738 = {'dtype': keyword_164737}
    # Getting the type of 'np' (line 1884)
    np_164732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1884, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1884)
    array_164733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1884, 15), np_164732, 'array')
    # Calling array(args, kwargs) (line 1884)
    array_call_result_164739 = invoke(stypy.reporting.localization.Localization(__file__, 1884, 15), array_164733, *[list_164734], **kwargs_164738)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1884, 8), 'stypy_return_type', array_call_result_164739)
    # SSA join for if statement (line 1883)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1885)
    # Processing the call arguments (line 1885)
    # Getting the type of 'c' (line 1885)
    c_164741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1885, 11), 'c', False)
    # Processing the call keyword arguments (line 1885)
    kwargs_164742 = {}
    # Getting the type of 'len' (line 1885)
    len_164740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1885, 7), 'len', False)
    # Calling len(args, kwargs) (line 1885)
    len_call_result_164743 = invoke(stypy.reporting.localization.Localization(__file__, 1885, 7), len_164740, *[c_164741], **kwargs_164742)
    
    int_164744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1885, 17), 'int')
    # Applying the binary operator '==' (line 1885)
    result_eq_164745 = python_operator(stypy.reporting.localization.Localization(__file__, 1885, 7), '==', len_call_result_164743, int_164744)
    
    # Testing the type of an if condition (line 1885)
    if_condition_164746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1885, 4), result_eq_164745)
    # Assigning a type to the variable 'if_condition_164746' (line 1885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1885, 4), 'if_condition_164746', if_condition_164746)
    # SSA begins for if statement (line 1885)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1886)
    # Processing the call arguments (line 1886)
    
    # Obtaining an instance of the builtin type 'list' (line 1886)
    list_164749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1886, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1886)
    # Adding element type (line 1886)
    
    
    # Obtaining the type of the subscript
    int_164750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1886, 28), 'int')
    # Getting the type of 'c' (line 1886)
    c_164751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 26), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1886)
    getitem___164752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1886, 26), c_164751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1886)
    subscript_call_result_164753 = invoke(stypy.reporting.localization.Localization(__file__, 1886, 26), getitem___164752, int_164750)
    
    # Applying the 'usub' unary operator (line 1886)
    result___neg___164754 = python_operator(stypy.reporting.localization.Localization(__file__, 1886, 25), 'usub', subscript_call_result_164753)
    
    
    # Obtaining the type of the subscript
    int_164755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1886, 33), 'int')
    # Getting the type of 'c' (line 1886)
    c_164756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 31), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1886)
    getitem___164757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1886, 31), c_164756, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1886)
    subscript_call_result_164758 = invoke(stypy.reporting.localization.Localization(__file__, 1886, 31), getitem___164757, int_164755)
    
    # Applying the binary operator 'div' (line 1886)
    result_div_164759 = python_operator(stypy.reporting.localization.Localization(__file__, 1886, 25), 'div', result___neg___164754, subscript_call_result_164758)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1886, 24), list_164749, result_div_164759)
    
    # Processing the call keyword arguments (line 1886)
    kwargs_164760 = {}
    # Getting the type of 'np' (line 1886)
    np_164747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1886)
    array_164748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1886, 15), np_164747, 'array')
    # Calling array(args, kwargs) (line 1886)
    array_call_result_164761 = invoke(stypy.reporting.localization.Localization(__file__, 1886, 15), array_164748, *[list_164749], **kwargs_164760)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1886, 8), 'stypy_return_type', array_call_result_164761)
    # SSA join for if statement (line 1885)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1888):
    
    # Assigning a Call to a Name (line 1888):
    
    # Call to chebcompanion(...): (line 1888)
    # Processing the call arguments (line 1888)
    # Getting the type of 'c' (line 1888)
    c_164763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1888, 22), 'c', False)
    # Processing the call keyword arguments (line 1888)
    kwargs_164764 = {}
    # Getting the type of 'chebcompanion' (line 1888)
    chebcompanion_164762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1888, 8), 'chebcompanion', False)
    # Calling chebcompanion(args, kwargs) (line 1888)
    chebcompanion_call_result_164765 = invoke(stypy.reporting.localization.Localization(__file__, 1888, 8), chebcompanion_164762, *[c_164763], **kwargs_164764)
    
    # Assigning a type to the variable 'm' (line 1888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1888, 4), 'm', chebcompanion_call_result_164765)
    
    # Assigning a Call to a Name (line 1889):
    
    # Assigning a Call to a Name (line 1889):
    
    # Call to eigvals(...): (line 1889)
    # Processing the call arguments (line 1889)
    # Getting the type of 'm' (line 1889)
    m_164768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1889, 19), 'm', False)
    # Processing the call keyword arguments (line 1889)
    kwargs_164769 = {}
    # Getting the type of 'la' (line 1889)
    la_164766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1889, 8), 'la', False)
    # Obtaining the member 'eigvals' of a type (line 1889)
    eigvals_164767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1889, 8), la_164766, 'eigvals')
    # Calling eigvals(args, kwargs) (line 1889)
    eigvals_call_result_164770 = invoke(stypy.reporting.localization.Localization(__file__, 1889, 8), eigvals_164767, *[m_164768], **kwargs_164769)
    
    # Assigning a type to the variable 'r' (line 1889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1889, 4), 'r', eigvals_call_result_164770)
    
    # Call to sort(...): (line 1890)
    # Processing the call keyword arguments (line 1890)
    kwargs_164773 = {}
    # Getting the type of 'r' (line 1890)
    r_164771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1890, 4), 'r', False)
    # Obtaining the member 'sort' of a type (line 1890)
    sort_164772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1890, 4), r_164771, 'sort')
    # Calling sort(args, kwargs) (line 1890)
    sort_call_result_164774 = invoke(stypy.reporting.localization.Localization(__file__, 1890, 4), sort_164772, *[], **kwargs_164773)
    
    # Getting the type of 'r' (line 1891)
    r_164775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1891, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 1891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1891, 4), 'stypy_return_type', r_164775)
    
    # ################# End of 'chebroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebroots' in the type store
    # Getting the type of 'stypy_return_type' (line 1838)
    stypy_return_type_164776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1838, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164776)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebroots'
    return stypy_return_type_164776

# Assigning a type to the variable 'chebroots' (line 1838)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1838, 0), 'chebroots', chebroots)

@norecursion
def chebgauss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebgauss'
    module_type_store = module_type_store.open_function_context('chebgauss', 1894, 0, False)
    
    # Passed parameters checking function
    chebgauss.stypy_localization = localization
    chebgauss.stypy_type_of_self = None
    chebgauss.stypy_type_store = module_type_store
    chebgauss.stypy_function_name = 'chebgauss'
    chebgauss.stypy_param_names_list = ['deg']
    chebgauss.stypy_varargs_param_name = None
    chebgauss.stypy_kwargs_param_name = None
    chebgauss.stypy_call_defaults = defaults
    chebgauss.stypy_call_varargs = varargs
    chebgauss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebgauss', ['deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebgauss', localization, ['deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebgauss(...)' code ##################

    str_164777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1928, (-1)), 'str', '\n    Gauss-Chebyshev quadrature.\n\n    Computes the sample points and weights for Gauss-Chebyshev quadrature.\n    These sample points and weights will correctly integrate polynomials of\n    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with\n    the weight function :math:`f(x) = 1/\\sqrt{1 - x^2}`.\n\n    Parameters\n    ----------\n    deg : int\n        Number of sample points and weights. It must be >= 1.\n\n    Returns\n    -------\n    x : ndarray\n        1-D ndarray containing the sample points.\n    y : ndarray\n        1-D ndarray containing the weights.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    The results have only been tested up to degree 100, higher degrees may\n    be problematic. For Gauss-Chebyshev there are closed form solutions for\n    the sample points and weights. If n = `deg`, then\n\n    .. math:: x_i = \\cos(\\pi (2 i - 1) / (2 n))\n\n    .. math:: w_i = \\pi / n\n\n    ')
    
    # Assigning a Call to a Name (line 1929):
    
    # Assigning a Call to a Name (line 1929):
    
    # Call to int(...): (line 1929)
    # Processing the call arguments (line 1929)
    # Getting the type of 'deg' (line 1929)
    deg_164779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1929, 15), 'deg', False)
    # Processing the call keyword arguments (line 1929)
    kwargs_164780 = {}
    # Getting the type of 'int' (line 1929)
    int_164778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1929, 11), 'int', False)
    # Calling int(args, kwargs) (line 1929)
    int_call_result_164781 = invoke(stypy.reporting.localization.Localization(__file__, 1929, 11), int_164778, *[deg_164779], **kwargs_164780)
    
    # Assigning a type to the variable 'ideg' (line 1929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1929, 4), 'ideg', int_call_result_164781)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ideg' (line 1930)
    ideg_164782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1930, 7), 'ideg')
    # Getting the type of 'deg' (line 1930)
    deg_164783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1930, 15), 'deg')
    # Applying the binary operator '!=' (line 1930)
    result_ne_164784 = python_operator(stypy.reporting.localization.Localization(__file__, 1930, 7), '!=', ideg_164782, deg_164783)
    
    
    # Getting the type of 'ideg' (line 1930)
    ideg_164785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1930, 22), 'ideg')
    int_164786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1930, 29), 'int')
    # Applying the binary operator '<' (line 1930)
    result_lt_164787 = python_operator(stypy.reporting.localization.Localization(__file__, 1930, 22), '<', ideg_164785, int_164786)
    
    # Applying the binary operator 'or' (line 1930)
    result_or_keyword_164788 = python_operator(stypy.reporting.localization.Localization(__file__, 1930, 7), 'or', result_ne_164784, result_lt_164787)
    
    # Testing the type of an if condition (line 1930)
    if_condition_164789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1930, 4), result_or_keyword_164788)
    # Assigning a type to the variable 'if_condition_164789' (line 1930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1930, 4), 'if_condition_164789', if_condition_164789)
    # SSA begins for if statement (line 1930)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1931)
    # Processing the call arguments (line 1931)
    str_164791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1931, 25), 'str', 'deg must be a non-negative integer')
    # Processing the call keyword arguments (line 1931)
    kwargs_164792 = {}
    # Getting the type of 'ValueError' (line 1931)
    ValueError_164790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1931, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1931)
    ValueError_call_result_164793 = invoke(stypy.reporting.localization.Localization(__file__, 1931, 14), ValueError_164790, *[str_164791], **kwargs_164792)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1931, 8), ValueError_call_result_164793, 'raise parameter', BaseException)
    # SSA join for if statement (line 1930)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1933):
    
    # Assigning a Call to a Name (line 1933):
    
    # Call to cos(...): (line 1933)
    # Processing the call arguments (line 1933)
    # Getting the type of 'np' (line 1933)
    np_164796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1933, 15), 'np', False)
    # Obtaining the member 'pi' of a type (line 1933)
    pi_164797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1933, 15), np_164796, 'pi')
    
    # Call to arange(...): (line 1933)
    # Processing the call arguments (line 1933)
    int_164800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1933, 33), 'int')
    int_164801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1933, 36), 'int')
    # Getting the type of 'ideg' (line 1933)
    ideg_164802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1933, 38), 'ideg', False)
    # Applying the binary operator '*' (line 1933)
    result_mul_164803 = python_operator(stypy.reporting.localization.Localization(__file__, 1933, 36), '*', int_164801, ideg_164802)
    
    int_164804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1933, 44), 'int')
    # Processing the call keyword arguments (line 1933)
    kwargs_164805 = {}
    # Getting the type of 'np' (line 1933)
    np_164798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1933, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 1933)
    arange_164799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1933, 23), np_164798, 'arange')
    # Calling arange(args, kwargs) (line 1933)
    arange_call_result_164806 = invoke(stypy.reporting.localization.Localization(__file__, 1933, 23), arange_164799, *[int_164800, result_mul_164803, int_164804], **kwargs_164805)
    
    # Applying the binary operator '*' (line 1933)
    result_mul_164807 = python_operator(stypy.reporting.localization.Localization(__file__, 1933, 15), '*', pi_164797, arange_call_result_164806)
    
    float_164808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1933, 50), 'float')
    # Getting the type of 'ideg' (line 1933)
    ideg_164809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1933, 54), 'ideg', False)
    # Applying the binary operator '*' (line 1933)
    result_mul_164810 = python_operator(stypy.reporting.localization.Localization(__file__, 1933, 50), '*', float_164808, ideg_164809)
    
    # Applying the binary operator 'div' (line 1933)
    result_div_164811 = python_operator(stypy.reporting.localization.Localization(__file__, 1933, 47), 'div', result_mul_164807, result_mul_164810)
    
    # Processing the call keyword arguments (line 1933)
    kwargs_164812 = {}
    # Getting the type of 'np' (line 1933)
    np_164794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1933, 8), 'np', False)
    # Obtaining the member 'cos' of a type (line 1933)
    cos_164795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1933, 8), np_164794, 'cos')
    # Calling cos(args, kwargs) (line 1933)
    cos_call_result_164813 = invoke(stypy.reporting.localization.Localization(__file__, 1933, 8), cos_164795, *[result_div_164811], **kwargs_164812)
    
    # Assigning a type to the variable 'x' (line 1933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1933, 4), 'x', cos_call_result_164813)
    
    # Assigning a BinOp to a Name (line 1934):
    
    # Assigning a BinOp to a Name (line 1934):
    
    # Call to ones(...): (line 1934)
    # Processing the call arguments (line 1934)
    # Getting the type of 'ideg' (line 1934)
    ideg_164816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 16), 'ideg', False)
    # Processing the call keyword arguments (line 1934)
    kwargs_164817 = {}
    # Getting the type of 'np' (line 1934)
    np_164814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 1934)
    ones_164815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1934, 8), np_164814, 'ones')
    # Calling ones(args, kwargs) (line 1934)
    ones_call_result_164818 = invoke(stypy.reporting.localization.Localization(__file__, 1934, 8), ones_164815, *[ideg_164816], **kwargs_164817)
    
    # Getting the type of 'np' (line 1934)
    np_164819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 23), 'np')
    # Obtaining the member 'pi' of a type (line 1934)
    pi_164820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1934, 23), np_164819, 'pi')
    # Getting the type of 'ideg' (line 1934)
    ideg_164821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 29), 'ideg')
    # Applying the binary operator 'div' (line 1934)
    result_div_164822 = python_operator(stypy.reporting.localization.Localization(__file__, 1934, 23), 'div', pi_164820, ideg_164821)
    
    # Applying the binary operator '*' (line 1934)
    result_mul_164823 = python_operator(stypy.reporting.localization.Localization(__file__, 1934, 8), '*', ones_call_result_164818, result_div_164822)
    
    # Assigning a type to the variable 'w' (line 1934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1934, 4), 'w', result_mul_164823)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1936)
    tuple_164824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1936, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1936)
    # Adding element type (line 1936)
    # Getting the type of 'x' (line 1936)
    x_164825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1936, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1936, 11), tuple_164824, x_164825)
    # Adding element type (line 1936)
    # Getting the type of 'w' (line 1936)
    w_164826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1936, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1936, 11), tuple_164824, w_164826)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1936)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1936, 4), 'stypy_return_type', tuple_164824)
    
    # ################# End of 'chebgauss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebgauss' in the type store
    # Getting the type of 'stypy_return_type' (line 1894)
    stypy_return_type_164827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1894, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164827)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebgauss'
    return stypy_return_type_164827

# Assigning a type to the variable 'chebgauss' (line 1894)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1894, 0), 'chebgauss', chebgauss)

@norecursion
def chebweight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebweight'
    module_type_store = module_type_store.open_function_context('chebweight', 1939, 0, False)
    
    # Passed parameters checking function
    chebweight.stypy_localization = localization
    chebweight.stypy_type_of_self = None
    chebweight.stypy_type_store = module_type_store
    chebweight.stypy_function_name = 'chebweight'
    chebweight.stypy_param_names_list = ['x']
    chebweight.stypy_varargs_param_name = None
    chebweight.stypy_kwargs_param_name = None
    chebweight.stypy_call_defaults = defaults
    chebweight.stypy_call_varargs = varargs
    chebweight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebweight', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebweight', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebweight(...)' code ##################

    str_164828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1962, (-1)), 'str', '\n    The weight function of the Chebyshev polynomials.\n\n    The weight function is :math:`1/\\sqrt{1 - x^2}` and the interval of\n    integration is :math:`[-1, 1]`. The Chebyshev polynomials are\n    orthogonal, but not normalized, with respect to this weight function.\n\n    Parameters\n    ----------\n    x : array_like\n       Values at which the weight function will be computed.\n\n    Returns\n    -------\n    w : ndarray\n       The weight function at `x`.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    ')
    
    # Assigning a BinOp to a Name (line 1963):
    
    # Assigning a BinOp to a Name (line 1963):
    float_164829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1963, 8), 'float')
    
    # Call to sqrt(...): (line 1963)
    # Processing the call arguments (line 1963)
    float_164832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1963, 20), 'float')
    # Getting the type of 'x' (line 1963)
    x_164833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1963, 25), 'x', False)
    # Applying the binary operator '+' (line 1963)
    result_add_164834 = python_operator(stypy.reporting.localization.Localization(__file__, 1963, 20), '+', float_164832, x_164833)
    
    # Processing the call keyword arguments (line 1963)
    kwargs_164835 = {}
    # Getting the type of 'np' (line 1963)
    np_164830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1963, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1963)
    sqrt_164831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1963, 12), np_164830, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1963)
    sqrt_call_result_164836 = invoke(stypy.reporting.localization.Localization(__file__, 1963, 12), sqrt_164831, *[result_add_164834], **kwargs_164835)
    
    
    # Call to sqrt(...): (line 1963)
    # Processing the call arguments (line 1963)
    float_164839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1963, 38), 'float')
    # Getting the type of 'x' (line 1963)
    x_164840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1963, 43), 'x', False)
    # Applying the binary operator '-' (line 1963)
    result_sub_164841 = python_operator(stypy.reporting.localization.Localization(__file__, 1963, 38), '-', float_164839, x_164840)
    
    # Processing the call keyword arguments (line 1963)
    kwargs_164842 = {}
    # Getting the type of 'np' (line 1963)
    np_164837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1963, 30), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1963)
    sqrt_164838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1963, 30), np_164837, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1963)
    sqrt_call_result_164843 = invoke(stypy.reporting.localization.Localization(__file__, 1963, 30), sqrt_164838, *[result_sub_164841], **kwargs_164842)
    
    # Applying the binary operator '*' (line 1963)
    result_mul_164844 = python_operator(stypy.reporting.localization.Localization(__file__, 1963, 12), '*', sqrt_call_result_164836, sqrt_call_result_164843)
    
    # Applying the binary operator 'div' (line 1963)
    result_div_164845 = python_operator(stypy.reporting.localization.Localization(__file__, 1963, 8), 'div', float_164829, result_mul_164844)
    
    # Assigning a type to the variable 'w' (line 1963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1963, 4), 'w', result_div_164845)
    # Getting the type of 'w' (line 1964)
    w_164846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1964, 11), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 1964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1964, 4), 'stypy_return_type', w_164846)
    
    # ################# End of 'chebweight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebweight' in the type store
    # Getting the type of 'stypy_return_type' (line 1939)
    stypy_return_type_164847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1939, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164847)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebweight'
    return stypy_return_type_164847

# Assigning a type to the variable 'chebweight' (line 1939)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1939, 0), 'chebweight', chebweight)

@norecursion
def chebpts1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebpts1'
    module_type_store = module_type_store.open_function_context('chebpts1', 1967, 0, False)
    
    # Passed parameters checking function
    chebpts1.stypy_localization = localization
    chebpts1.stypy_type_of_self = None
    chebpts1.stypy_type_store = module_type_store
    chebpts1.stypy_function_name = 'chebpts1'
    chebpts1.stypy_param_names_list = ['npts']
    chebpts1.stypy_varargs_param_name = None
    chebpts1.stypy_kwargs_param_name = None
    chebpts1.stypy_call_defaults = defaults
    chebpts1.stypy_call_varargs = varargs
    chebpts1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebpts1', ['npts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebpts1', localization, ['npts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebpts1(...)' code ##################

    str_164848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1993, (-1)), 'str', '\n    Chebyshev points of the first kind.\n\n    The Chebyshev points of the first kind are the points ``cos(x)``,\n    where ``x = [pi*(k + .5)/npts for k in range(npts)]``.\n\n    Parameters\n    ----------\n    npts : int\n        Number of sample points desired.\n\n    Returns\n    -------\n    pts : ndarray\n        The Chebyshev points of the first kind.\n\n    See Also\n    --------\n    chebpts2\n\n    Notes\n    -----\n\n    .. versionadded:: 1.5.0\n\n    ')
    
    # Assigning a Call to a Name (line 1994):
    
    # Assigning a Call to a Name (line 1994):
    
    # Call to int(...): (line 1994)
    # Processing the call arguments (line 1994)
    # Getting the type of 'npts' (line 1994)
    npts_164850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1994, 16), 'npts', False)
    # Processing the call keyword arguments (line 1994)
    kwargs_164851 = {}
    # Getting the type of 'int' (line 1994)
    int_164849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1994, 12), 'int', False)
    # Calling int(args, kwargs) (line 1994)
    int_call_result_164852 = invoke(stypy.reporting.localization.Localization(__file__, 1994, 12), int_164849, *[npts_164850], **kwargs_164851)
    
    # Assigning a type to the variable '_npts' (line 1994)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1994, 4), '_npts', int_call_result_164852)
    
    
    # Getting the type of '_npts' (line 1995)
    _npts_164853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1995, 7), '_npts')
    # Getting the type of 'npts' (line 1995)
    npts_164854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1995, 16), 'npts')
    # Applying the binary operator '!=' (line 1995)
    result_ne_164855 = python_operator(stypy.reporting.localization.Localization(__file__, 1995, 7), '!=', _npts_164853, npts_164854)
    
    # Testing the type of an if condition (line 1995)
    if_condition_164856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1995, 4), result_ne_164855)
    # Assigning a type to the variable 'if_condition_164856' (line 1995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1995, 4), 'if_condition_164856', if_condition_164856)
    # SSA begins for if statement (line 1995)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1996)
    # Processing the call arguments (line 1996)
    str_164858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1996, 25), 'str', 'npts must be integer')
    # Processing the call keyword arguments (line 1996)
    kwargs_164859 = {}
    # Getting the type of 'ValueError' (line 1996)
    ValueError_164857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1996, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1996)
    ValueError_call_result_164860 = invoke(stypy.reporting.localization.Localization(__file__, 1996, 14), ValueError_164857, *[str_164858], **kwargs_164859)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1996, 8), ValueError_call_result_164860, 'raise parameter', BaseException)
    # SSA join for if statement (line 1995)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of '_npts' (line 1997)
    _npts_164861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1997, 7), '_npts')
    int_164862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1997, 15), 'int')
    # Applying the binary operator '<' (line 1997)
    result_lt_164863 = python_operator(stypy.reporting.localization.Localization(__file__, 1997, 7), '<', _npts_164861, int_164862)
    
    # Testing the type of an if condition (line 1997)
    if_condition_164864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1997, 4), result_lt_164863)
    # Assigning a type to the variable 'if_condition_164864' (line 1997)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1997, 4), 'if_condition_164864', if_condition_164864)
    # SSA begins for if statement (line 1997)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1998)
    # Processing the call arguments (line 1998)
    str_164866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1998, 25), 'str', 'npts must be >= 1')
    # Processing the call keyword arguments (line 1998)
    kwargs_164867 = {}
    # Getting the type of 'ValueError' (line 1998)
    ValueError_164865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1998, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1998)
    ValueError_call_result_164868 = invoke(stypy.reporting.localization.Localization(__file__, 1998, 14), ValueError_164865, *[str_164866], **kwargs_164867)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1998, 8), ValueError_call_result_164868, 'raise parameter', BaseException)
    # SSA join for if statement (line 1997)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 2000):
    
    # Assigning a BinOp to a Name (line 2000):
    
    # Call to linspace(...): (line 2000)
    # Processing the call arguments (line 2000)
    
    # Getting the type of 'np' (line 2000)
    np_164871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2000, 21), 'np', False)
    # Obtaining the member 'pi' of a type (line 2000)
    pi_164872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2000, 21), np_164871, 'pi')
    # Applying the 'usub' unary operator (line 2000)
    result___neg___164873 = python_operator(stypy.reporting.localization.Localization(__file__, 2000, 20), 'usub', pi_164872)
    
    int_164874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2000, 28), 'int')
    # Getting the type of '_npts' (line 2000)
    _npts_164875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2000, 31), '_npts', False)
    # Processing the call keyword arguments (line 2000)
    # Getting the type of 'False' (line 2000)
    False_164876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2000, 47), 'False', False)
    keyword_164877 = False_164876
    kwargs_164878 = {'endpoint': keyword_164877}
    # Getting the type of 'np' (line 2000)
    np_164869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2000, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 2000)
    linspace_164870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2000, 8), np_164869, 'linspace')
    # Calling linspace(args, kwargs) (line 2000)
    linspace_call_result_164879 = invoke(stypy.reporting.localization.Localization(__file__, 2000, 8), linspace_164870, *[result___neg___164873, int_164874, _npts_164875], **kwargs_164878)
    
    # Getting the type of 'np' (line 2000)
    np_164880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2000, 56), 'np')
    # Obtaining the member 'pi' of a type (line 2000)
    pi_164881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2000, 56), np_164880, 'pi')
    int_164882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2000, 63), 'int')
    # Getting the type of '_npts' (line 2000)
    _npts_164883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2000, 65), '_npts')
    # Applying the binary operator '*' (line 2000)
    result_mul_164884 = python_operator(stypy.reporting.localization.Localization(__file__, 2000, 63), '*', int_164882, _npts_164883)
    
    # Applying the binary operator 'div' (line 2000)
    result_div_164885 = python_operator(stypy.reporting.localization.Localization(__file__, 2000, 56), 'div', pi_164881, result_mul_164884)
    
    # Applying the binary operator '+' (line 2000)
    result_add_164886 = python_operator(stypy.reporting.localization.Localization(__file__, 2000, 8), '+', linspace_call_result_164879, result_div_164885)
    
    # Assigning a type to the variable 'x' (line 2000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2000, 4), 'x', result_add_164886)
    
    # Call to cos(...): (line 2001)
    # Processing the call arguments (line 2001)
    # Getting the type of 'x' (line 2001)
    x_164889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2001, 18), 'x', False)
    # Processing the call keyword arguments (line 2001)
    kwargs_164890 = {}
    # Getting the type of 'np' (line 2001)
    np_164887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2001, 11), 'np', False)
    # Obtaining the member 'cos' of a type (line 2001)
    cos_164888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2001, 11), np_164887, 'cos')
    # Calling cos(args, kwargs) (line 2001)
    cos_call_result_164891 = invoke(stypy.reporting.localization.Localization(__file__, 2001, 11), cos_164888, *[x_164889], **kwargs_164890)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2001, 4), 'stypy_return_type', cos_call_result_164891)
    
    # ################# End of 'chebpts1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebpts1' in the type store
    # Getting the type of 'stypy_return_type' (line 1967)
    stypy_return_type_164892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164892)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebpts1'
    return stypy_return_type_164892

# Assigning a type to the variable 'chebpts1' (line 1967)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1967, 0), 'chebpts1', chebpts1)

@norecursion
def chebpts2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chebpts2'
    module_type_store = module_type_store.open_function_context('chebpts2', 2004, 0, False)
    
    # Passed parameters checking function
    chebpts2.stypy_localization = localization
    chebpts2.stypy_type_of_self = None
    chebpts2.stypy_type_store = module_type_store
    chebpts2.stypy_function_name = 'chebpts2'
    chebpts2.stypy_param_names_list = ['npts']
    chebpts2.stypy_varargs_param_name = None
    chebpts2.stypy_kwargs_param_name = None
    chebpts2.stypy_call_defaults = defaults
    chebpts2.stypy_call_varargs = varargs
    chebpts2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebpts2', ['npts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebpts2', localization, ['npts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebpts2(...)' code ##################

    str_164893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2026, (-1)), 'str', '\n    Chebyshev points of the second kind.\n\n    The Chebyshev points of the second kind are the points ``cos(x)``,\n    where ``x = [pi*k/(npts - 1) for k in range(npts)]``.\n\n    Parameters\n    ----------\n    npts : int\n        Number of sample points desired.\n\n    Returns\n    -------\n    pts : ndarray\n        The Chebyshev points of the second kind.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.5.0\n\n    ')
    
    # Assigning a Call to a Name (line 2027):
    
    # Assigning a Call to a Name (line 2027):
    
    # Call to int(...): (line 2027)
    # Processing the call arguments (line 2027)
    # Getting the type of 'npts' (line 2027)
    npts_164895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2027, 16), 'npts', False)
    # Processing the call keyword arguments (line 2027)
    kwargs_164896 = {}
    # Getting the type of 'int' (line 2027)
    int_164894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2027, 12), 'int', False)
    # Calling int(args, kwargs) (line 2027)
    int_call_result_164897 = invoke(stypy.reporting.localization.Localization(__file__, 2027, 12), int_164894, *[npts_164895], **kwargs_164896)
    
    # Assigning a type to the variable '_npts' (line 2027)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2027, 4), '_npts', int_call_result_164897)
    
    
    # Getting the type of '_npts' (line 2028)
    _npts_164898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2028, 7), '_npts')
    # Getting the type of 'npts' (line 2028)
    npts_164899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2028, 16), 'npts')
    # Applying the binary operator '!=' (line 2028)
    result_ne_164900 = python_operator(stypy.reporting.localization.Localization(__file__, 2028, 7), '!=', _npts_164898, npts_164899)
    
    # Testing the type of an if condition (line 2028)
    if_condition_164901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2028, 4), result_ne_164900)
    # Assigning a type to the variable 'if_condition_164901' (line 2028)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2028, 4), 'if_condition_164901', if_condition_164901)
    # SSA begins for if statement (line 2028)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 2029)
    # Processing the call arguments (line 2029)
    str_164903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2029, 25), 'str', 'npts must be integer')
    # Processing the call keyword arguments (line 2029)
    kwargs_164904 = {}
    # Getting the type of 'ValueError' (line 2029)
    ValueError_164902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2029, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 2029)
    ValueError_call_result_164905 = invoke(stypy.reporting.localization.Localization(__file__, 2029, 14), ValueError_164902, *[str_164903], **kwargs_164904)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2029, 8), ValueError_call_result_164905, 'raise parameter', BaseException)
    # SSA join for if statement (line 2028)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of '_npts' (line 2030)
    _npts_164906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2030, 7), '_npts')
    int_164907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2030, 15), 'int')
    # Applying the binary operator '<' (line 2030)
    result_lt_164908 = python_operator(stypy.reporting.localization.Localization(__file__, 2030, 7), '<', _npts_164906, int_164907)
    
    # Testing the type of an if condition (line 2030)
    if_condition_164909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2030, 4), result_lt_164908)
    # Assigning a type to the variable 'if_condition_164909' (line 2030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2030, 4), 'if_condition_164909', if_condition_164909)
    # SSA begins for if statement (line 2030)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 2031)
    # Processing the call arguments (line 2031)
    str_164911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2031, 25), 'str', 'npts must be >= 2')
    # Processing the call keyword arguments (line 2031)
    kwargs_164912 = {}
    # Getting the type of 'ValueError' (line 2031)
    ValueError_164910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2031, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 2031)
    ValueError_call_result_164913 = invoke(stypy.reporting.localization.Localization(__file__, 2031, 14), ValueError_164910, *[str_164911], **kwargs_164912)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2031, 8), ValueError_call_result_164913, 'raise parameter', BaseException)
    # SSA join for if statement (line 2030)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 2033):
    
    # Assigning a Call to a Name (line 2033):
    
    # Call to linspace(...): (line 2033)
    # Processing the call arguments (line 2033)
    
    # Getting the type of 'np' (line 2033)
    np_164916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2033, 21), 'np', False)
    # Obtaining the member 'pi' of a type (line 2033)
    pi_164917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2033, 21), np_164916, 'pi')
    # Applying the 'usub' unary operator (line 2033)
    result___neg___164918 = python_operator(stypy.reporting.localization.Localization(__file__, 2033, 20), 'usub', pi_164917)
    
    int_164919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2033, 28), 'int')
    # Getting the type of '_npts' (line 2033)
    _npts_164920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2033, 31), '_npts', False)
    # Processing the call keyword arguments (line 2033)
    kwargs_164921 = {}
    # Getting the type of 'np' (line 2033)
    np_164914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2033, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 2033)
    linspace_164915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2033, 8), np_164914, 'linspace')
    # Calling linspace(args, kwargs) (line 2033)
    linspace_call_result_164922 = invoke(stypy.reporting.localization.Localization(__file__, 2033, 8), linspace_164915, *[result___neg___164918, int_164919, _npts_164920], **kwargs_164921)
    
    # Assigning a type to the variable 'x' (line 2033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2033, 4), 'x', linspace_call_result_164922)
    
    # Call to cos(...): (line 2034)
    # Processing the call arguments (line 2034)
    # Getting the type of 'x' (line 2034)
    x_164925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2034, 18), 'x', False)
    # Processing the call keyword arguments (line 2034)
    kwargs_164926 = {}
    # Getting the type of 'np' (line 2034)
    np_164923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2034, 11), 'np', False)
    # Obtaining the member 'cos' of a type (line 2034)
    cos_164924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2034, 11), np_164923, 'cos')
    # Calling cos(args, kwargs) (line 2034)
    cos_call_result_164927 = invoke(stypy.reporting.localization.Localization(__file__, 2034, 11), cos_164924, *[x_164925], **kwargs_164926)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2034, 4), 'stypy_return_type', cos_call_result_164927)
    
    # ################# End of 'chebpts2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebpts2' in the type store
    # Getting the type of 'stypy_return_type' (line 2004)
    stypy_return_type_164928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2004, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_164928)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebpts2'
    return stypy_return_type_164928

# Assigning a type to the variable 'chebpts2' (line 2004)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2004, 0), 'chebpts2', chebpts2)
# Declaration of the 'Chebyshev' class
# Getting the type of 'ABCPolyBase' (line 2041)
ABCPolyBase_164929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2041, 16), 'ABCPolyBase')

class Chebyshev(ABCPolyBase_164929, ):
    str_164930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2062, (-1)), 'str', "A Chebyshev series class.\n\n    The Chebyshev class provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the\n    methods listed below.\n\n    Parameters\n    ----------\n    coef : array_like\n        Chebyshev coefficients in order of increasing degree, i.e.,\n        ``(1, 2, 3)`` gives ``1*T_0(x) + 2*T_1(x) + 3*T_2(x)``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is [-1, 1].\n    window : (2,) array_like, optional\n        Window, see `domain` for its use. The default value is [-1, 1].\n\n        .. versionadded:: 1.6.0\n\n    ")
    
    # Assigning a Call to a Name (line 2064):
    
    # Assigning a Call to a Name (line 2065):
    
    # Assigning a Call to a Name (line 2066):
    
    # Assigning a Call to a Name (line 2067):
    
    # Assigning a Call to a Name (line 2068):
    
    # Assigning a Call to a Name (line 2069):
    
    # Assigning a Call to a Name (line 2070):
    
    # Assigning a Call to a Name (line 2071):
    
    # Assigning a Call to a Name (line 2072):
    
    # Assigning a Call to a Name (line 2073):
    
    # Assigning a Call to a Name (line 2074):
    
    # Assigning a Call to a Name (line 2075):
    
    # Assigning a Str to a Name (line 2078):
    
    # Assigning a Call to a Name (line 2079):
    
    # Assigning a Call to a Name (line 2080):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 2041, 0, False)
        # Assigning a type to the variable 'self' (line 2042)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2042, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chebyshev.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Chebyshev' (line 2041)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2041, 0), 'Chebyshev', Chebyshev)

# Assigning a Call to a Name (line 2064):

# Call to staticmethod(...): (line 2064)
# Processing the call arguments (line 2064)
# Getting the type of 'chebadd' (line 2064)
chebadd_164932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2064, 24), 'chebadd', False)
# Processing the call keyword arguments (line 2064)
kwargs_164933 = {}
# Getting the type of 'staticmethod' (line 2064)
staticmethod_164931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2064, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2064)
staticmethod_call_result_164934 = invoke(stypy.reporting.localization.Localization(__file__, 2064, 11), staticmethod_164931, *[chebadd_164932], **kwargs_164933)

# Getting the type of 'Chebyshev'
Chebyshev_164935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164935, '_add', staticmethod_call_result_164934)

# Assigning a Call to a Name (line 2065):

# Call to staticmethod(...): (line 2065)
# Processing the call arguments (line 2065)
# Getting the type of 'chebsub' (line 2065)
chebsub_164937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2065, 24), 'chebsub', False)
# Processing the call keyword arguments (line 2065)
kwargs_164938 = {}
# Getting the type of 'staticmethod' (line 2065)
staticmethod_164936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2065, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2065)
staticmethod_call_result_164939 = invoke(stypy.reporting.localization.Localization(__file__, 2065, 11), staticmethod_164936, *[chebsub_164937], **kwargs_164938)

# Getting the type of 'Chebyshev'
Chebyshev_164940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_sub' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164940, '_sub', staticmethod_call_result_164939)

# Assigning a Call to a Name (line 2066):

# Call to staticmethod(...): (line 2066)
# Processing the call arguments (line 2066)
# Getting the type of 'chebmul' (line 2066)
chebmul_164942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2066, 24), 'chebmul', False)
# Processing the call keyword arguments (line 2066)
kwargs_164943 = {}
# Getting the type of 'staticmethod' (line 2066)
staticmethod_164941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2066, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2066)
staticmethod_call_result_164944 = invoke(stypy.reporting.localization.Localization(__file__, 2066, 11), staticmethod_164941, *[chebmul_164942], **kwargs_164943)

# Getting the type of 'Chebyshev'
Chebyshev_164945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_mul' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164945, '_mul', staticmethod_call_result_164944)

# Assigning a Call to a Name (line 2067):

# Call to staticmethod(...): (line 2067)
# Processing the call arguments (line 2067)
# Getting the type of 'chebdiv' (line 2067)
chebdiv_164947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2067, 24), 'chebdiv', False)
# Processing the call keyword arguments (line 2067)
kwargs_164948 = {}
# Getting the type of 'staticmethod' (line 2067)
staticmethod_164946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2067, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2067)
staticmethod_call_result_164949 = invoke(stypy.reporting.localization.Localization(__file__, 2067, 11), staticmethod_164946, *[chebdiv_164947], **kwargs_164948)

# Getting the type of 'Chebyshev'
Chebyshev_164950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_div' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164950, '_div', staticmethod_call_result_164949)

# Assigning a Call to a Name (line 2068):

# Call to staticmethod(...): (line 2068)
# Processing the call arguments (line 2068)
# Getting the type of 'chebpow' (line 2068)
chebpow_164952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2068, 24), 'chebpow', False)
# Processing the call keyword arguments (line 2068)
kwargs_164953 = {}
# Getting the type of 'staticmethod' (line 2068)
staticmethod_164951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2068, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2068)
staticmethod_call_result_164954 = invoke(stypy.reporting.localization.Localization(__file__, 2068, 11), staticmethod_164951, *[chebpow_164952], **kwargs_164953)

# Getting the type of 'Chebyshev'
Chebyshev_164955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_pow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164955, '_pow', staticmethod_call_result_164954)

# Assigning a Call to a Name (line 2069):

# Call to staticmethod(...): (line 2069)
# Processing the call arguments (line 2069)
# Getting the type of 'chebval' (line 2069)
chebval_164957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2069, 24), 'chebval', False)
# Processing the call keyword arguments (line 2069)
kwargs_164958 = {}
# Getting the type of 'staticmethod' (line 2069)
staticmethod_164956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2069, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2069)
staticmethod_call_result_164959 = invoke(stypy.reporting.localization.Localization(__file__, 2069, 11), staticmethod_164956, *[chebval_164957], **kwargs_164958)

# Getting the type of 'Chebyshev'
Chebyshev_164960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_val' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164960, '_val', staticmethod_call_result_164959)

# Assigning a Call to a Name (line 2070):

# Call to staticmethod(...): (line 2070)
# Processing the call arguments (line 2070)
# Getting the type of 'chebint' (line 2070)
chebint_164962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2070, 24), 'chebint', False)
# Processing the call keyword arguments (line 2070)
kwargs_164963 = {}
# Getting the type of 'staticmethod' (line 2070)
staticmethod_164961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2070, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2070)
staticmethod_call_result_164964 = invoke(stypy.reporting.localization.Localization(__file__, 2070, 11), staticmethod_164961, *[chebint_164962], **kwargs_164963)

# Getting the type of 'Chebyshev'
Chebyshev_164965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_int' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164965, '_int', staticmethod_call_result_164964)

# Assigning a Call to a Name (line 2071):

# Call to staticmethod(...): (line 2071)
# Processing the call arguments (line 2071)
# Getting the type of 'chebder' (line 2071)
chebder_164967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2071, 24), 'chebder', False)
# Processing the call keyword arguments (line 2071)
kwargs_164968 = {}
# Getting the type of 'staticmethod' (line 2071)
staticmethod_164966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2071, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2071)
staticmethod_call_result_164969 = invoke(stypy.reporting.localization.Localization(__file__, 2071, 11), staticmethod_164966, *[chebder_164967], **kwargs_164968)

# Getting the type of 'Chebyshev'
Chebyshev_164970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_der' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164970, '_der', staticmethod_call_result_164969)

# Assigning a Call to a Name (line 2072):

# Call to staticmethod(...): (line 2072)
# Processing the call arguments (line 2072)
# Getting the type of 'chebfit' (line 2072)
chebfit_164972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2072, 24), 'chebfit', False)
# Processing the call keyword arguments (line 2072)
kwargs_164973 = {}
# Getting the type of 'staticmethod' (line 2072)
staticmethod_164971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2072, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2072)
staticmethod_call_result_164974 = invoke(stypy.reporting.localization.Localization(__file__, 2072, 11), staticmethod_164971, *[chebfit_164972], **kwargs_164973)

# Getting the type of 'Chebyshev'
Chebyshev_164975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_fit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164975, '_fit', staticmethod_call_result_164974)

# Assigning a Call to a Name (line 2073):

# Call to staticmethod(...): (line 2073)
# Processing the call arguments (line 2073)
# Getting the type of 'chebline' (line 2073)
chebline_164977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2073, 25), 'chebline', False)
# Processing the call keyword arguments (line 2073)
kwargs_164978 = {}
# Getting the type of 'staticmethod' (line 2073)
staticmethod_164976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2073, 12), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2073)
staticmethod_call_result_164979 = invoke(stypy.reporting.localization.Localization(__file__, 2073, 12), staticmethod_164976, *[chebline_164977], **kwargs_164978)

# Getting the type of 'Chebyshev'
Chebyshev_164980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_line' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164980, '_line', staticmethod_call_result_164979)

# Assigning a Call to a Name (line 2074):

# Call to staticmethod(...): (line 2074)
# Processing the call arguments (line 2074)
# Getting the type of 'chebroots' (line 2074)
chebroots_164982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2074, 26), 'chebroots', False)
# Processing the call keyword arguments (line 2074)
kwargs_164983 = {}
# Getting the type of 'staticmethod' (line 2074)
staticmethod_164981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2074, 13), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2074)
staticmethod_call_result_164984 = invoke(stypy.reporting.localization.Localization(__file__, 2074, 13), staticmethod_164981, *[chebroots_164982], **kwargs_164983)

# Getting the type of 'Chebyshev'
Chebyshev_164985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_roots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164985, '_roots', staticmethod_call_result_164984)

# Assigning a Call to a Name (line 2075):

# Call to staticmethod(...): (line 2075)
# Processing the call arguments (line 2075)
# Getting the type of 'chebfromroots' (line 2075)
chebfromroots_164987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 30), 'chebfromroots', False)
# Processing the call keyword arguments (line 2075)
kwargs_164988 = {}
# Getting the type of 'staticmethod' (line 2075)
staticmethod_164986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 17), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 2075)
staticmethod_call_result_164989 = invoke(stypy.reporting.localization.Localization(__file__, 2075, 17), staticmethod_164986, *[chebfromroots_164987], **kwargs_164988)

# Getting the type of 'Chebyshev'
Chebyshev_164990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member '_fromroots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164990, '_fromroots', staticmethod_call_result_164989)

# Assigning a Str to a Name (line 2078):
str_164991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2078, 15), 'str', 'cheb')
# Getting the type of 'Chebyshev'
Chebyshev_164992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member 'nickname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164992, 'nickname', str_164991)

# Assigning a Call to a Name (line 2079):

# Call to array(...): (line 2079)
# Processing the call arguments (line 2079)
# Getting the type of 'chebdomain' (line 2079)
chebdomain_164995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2079, 22), 'chebdomain', False)
# Processing the call keyword arguments (line 2079)
kwargs_164996 = {}
# Getting the type of 'np' (line 2079)
np_164993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2079, 13), 'np', False)
# Obtaining the member 'array' of a type (line 2079)
array_164994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2079, 13), np_164993, 'array')
# Calling array(args, kwargs) (line 2079)
array_call_result_164997 = invoke(stypy.reporting.localization.Localization(__file__, 2079, 13), array_164994, *[chebdomain_164995], **kwargs_164996)

# Getting the type of 'Chebyshev'
Chebyshev_164998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member 'domain' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_164998, 'domain', array_call_result_164997)

# Assigning a Call to a Name (line 2080):

# Call to array(...): (line 2080)
# Processing the call arguments (line 2080)
# Getting the type of 'chebdomain' (line 2080)
chebdomain_165001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2080, 22), 'chebdomain', False)
# Processing the call keyword arguments (line 2080)
kwargs_165002 = {}
# Getting the type of 'np' (line 2080)
np_164999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2080, 13), 'np', False)
# Obtaining the member 'array' of a type (line 2080)
array_165000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2080, 13), np_164999, 'array')
# Calling array(args, kwargs) (line 2080)
array_call_result_165003 = invoke(stypy.reporting.localization.Localization(__file__, 2080, 13), array_165000, *[chebdomain_165001], **kwargs_165002)

# Getting the type of 'Chebyshev'
Chebyshev_165004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Chebyshev')
# Setting the type of the member 'window' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Chebyshev_165004, 'window', array_call_result_165003)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
