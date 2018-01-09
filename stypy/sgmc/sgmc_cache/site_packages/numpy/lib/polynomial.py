
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Functions to operate on polynomials.
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: __all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
8:            'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
9:            'polyfit', 'RankWarning']
10: 
11: import re
12: import warnings
13: import numpy.core.numeric as NX
14: 
15: from numpy.core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
16:                         ones)
17: from numpy.lib.twodim_base import diag, vander
18: from numpy.lib.function_base import trim_zeros, sort_complex
19: from numpy.lib.type_check import iscomplex, real, imag, mintypecode
20: from numpy.linalg import eigvals, lstsq, inv
21: 
22: class RankWarning(UserWarning):
23:     '''
24:     Issued by `polyfit` when the Vandermonde matrix is rank deficient.
25: 
26:     For more information, a way to suppress the warning, and an example of
27:     `RankWarning` being issued, see `polyfit`.
28: 
29:     '''
30:     pass
31: 
32: def poly(seq_of_zeros):
33:     '''
34:     Find the coefficients of a polynomial with the given sequence of roots.
35: 
36:     Returns the coefficients of the polynomial whose leading coefficient
37:     is one for the given sequence of zeros (multiple roots must be included
38:     in the sequence as many times as their multiplicity; see Examples).
39:     A square matrix (or array, which will be treated as a matrix) can also
40:     be given, in which case the coefficients of the characteristic polynomial
41:     of the matrix are returned.
42: 
43:     Parameters
44:     ----------
45:     seq_of_zeros : array_like, shape (N,) or (N, N)
46:         A sequence of polynomial roots, or a square array or matrix object.
47: 
48:     Returns
49:     -------
50:     c : ndarray
51:         1D array of polynomial coefficients from highest to lowest degree:
52: 
53:         ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
54:         where c[0] always equals 1.
55: 
56:     Raises
57:     ------
58:     ValueError
59:         If input is the wrong shape (the input must be a 1-D or square
60:         2-D array).
61: 
62:     See Also
63:     --------
64:     polyval : Compute polynomial values.
65:     roots : Return the roots of a polynomial.
66:     polyfit : Least squares polynomial fit.
67:     poly1d : A one-dimensional polynomial class.
68: 
69:     Notes
70:     -----
71:     Specifying the roots of a polynomial still leaves one degree of
72:     freedom, typically represented by an undetermined leading
73:     coefficient. [1]_ In the case of this function, that coefficient -
74:     the first one in the returned array - is always taken as one. (If
75:     for some reason you have one other point, the only automatic way
76:     presently to leverage that information is to use ``polyfit``.)
77: 
78:     The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`
79:     matrix **A** is given by
80: 
81:         :math:`p_a(t) = \\mathrm{det}(t\\, \\mathbf{I} - \\mathbf{A})`,
82: 
83:     where **I** is the `n`-by-`n` identity matrix. [2]_
84: 
85:     References
86:     ----------
87:     .. [1] M. Sullivan and M. Sullivan, III, "Algebra and Trignometry,
88:        Enhanced With Graphing Utilities," Prentice-Hall, pg. 318, 1996.
89: 
90:     .. [2] G. Strang, "Linear Algebra and Its Applications, 2nd Edition,"
91:        Academic Press, pg. 182, 1980.
92: 
93:     Examples
94:     --------
95:     Given a sequence of a polynomial's zeros:
96: 
97:     >>> np.poly((0, 0, 0)) # Multiple root example
98:     array([1, 0, 0, 0])
99: 
100:     The line above represents z**3 + 0*z**2 + 0*z + 0.
101: 
102:     >>> np.poly((-1./2, 0, 1./2))
103:     array([ 1.  ,  0.  , -0.25,  0.  ])
104: 
105:     The line above represents z**3 - z/4
106: 
107:     >>> np.poly((np.random.random(1.)[0], 0, np.random.random(1.)[0]))
108:     array([ 1.        , -0.77086955,  0.08618131,  0.        ]) #random
109: 
110:     Given a square array object:
111: 
112:     >>> P = np.array([[0, 1./3], [-1./2, 0]])
113:     >>> np.poly(P)
114:     array([ 1.        ,  0.        ,  0.16666667])
115: 
116:     Or a square matrix object:
117: 
118:     >>> np.poly(np.matrix(P))
119:     array([ 1.        ,  0.        ,  0.16666667])
120: 
121:     Note how in all cases the leading coefficient is always 1.
122: 
123:     '''
124:     seq_of_zeros = atleast_1d(seq_of_zeros)
125:     sh = seq_of_zeros.shape
126: 
127:     if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
128:         seq_of_zeros = eigvals(seq_of_zeros)
129:     elif len(sh) == 1:
130:         dt = seq_of_zeros.dtype
131:         # Let object arrays slip through, e.g. for arbitrary precision
132:         if dt != object:
133:             seq_of_zeros = seq_of_zeros.astype(mintypecode(dt.char))
134:     else:
135:         raise ValueError("input must be 1d or non-empty square 2d array.")
136: 
137:     if len(seq_of_zeros) == 0:
138:         return 1.0
139:     dt = seq_of_zeros.dtype
140:     a = ones((1,), dtype=dt)
141:     for k in range(len(seq_of_zeros)):
142:         a = NX.convolve(a, array([1, -seq_of_zeros[k]], dtype=dt),
143:                         mode='full')
144: 
145:     if issubclass(a.dtype.type, NX.complexfloating):
146:         # if complex roots are all complex conjugates, the roots are real.
147:         roots = NX.asarray(seq_of_zeros, complex)
148:         pos_roots = sort_complex(NX.compress(roots.imag > 0, roots))
149:         neg_roots = NX.conjugate(sort_complex(
150:                                         NX.compress(roots.imag < 0, roots)))
151:         if (len(pos_roots) == len(neg_roots) and
152:                 NX.alltrue(neg_roots == pos_roots)):
153:             a = a.real.copy()
154: 
155:     return a
156: 
157: def roots(p):
158:     '''
159:     Return the roots of a polynomial with coefficients given in p.
160: 
161:     The values in the rank-1 array `p` are coefficients of a polynomial.
162:     If the length of `p` is n+1 then the polynomial is described by::
163: 
164:       p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
165: 
166:     Parameters
167:     ----------
168:     p : array_like
169:         Rank-1 array of polynomial coefficients.
170: 
171:     Returns
172:     -------
173:     out : ndarray
174:         An array containing the complex roots of the polynomial.
175: 
176:     Raises
177:     ------
178:     ValueError
179:         When `p` cannot be converted to a rank-1 array.
180: 
181:     See also
182:     --------
183:     poly : Find the coefficients of a polynomial with a given sequence
184:            of roots.
185:     polyval : Compute polynomial values.
186:     polyfit : Least squares polynomial fit.
187:     poly1d : A one-dimensional polynomial class.
188: 
189:     Notes
190:     -----
191:     The algorithm relies on computing the eigenvalues of the
192:     companion matrix [1]_.
193: 
194:     References
195:     ----------
196:     .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
197:         Cambridge University Press, 1999, pp. 146-7.
198: 
199:     Examples
200:     --------
201:     >>> coeff = [3.2, 2, 1]
202:     >>> np.roots(coeff)
203:     array([-0.3125+0.46351241j, -0.3125-0.46351241j])
204: 
205:     '''
206:     # If input is scalar, this makes it an array
207:     p = atleast_1d(p)
208:     if len(p.shape) != 1:
209:         raise ValueError("Input must be a rank-1 array.")
210: 
211:     # find non-zero array entries
212:     non_zero = NX.nonzero(NX.ravel(p))[0]
213: 
214:     # Return an empty array if polynomial is all zeros
215:     if len(non_zero) == 0:
216:         return NX.array([])
217: 
218:     # find the number of trailing zeros -- this is the number of roots at 0.
219:     trailing_zeros = len(p) - non_zero[-1] - 1
220: 
221:     # strip leading and trailing zeros
222:     p = p[int(non_zero[0]):int(non_zero[-1])+1]
223: 
224:     # casting: if incoming array isn't floating point, make it floating point.
225:     if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
226:         p = p.astype(float)
227: 
228:     N = len(p)
229:     if N > 1:
230:         # build companion matrix and find its eigenvalues (the roots)
231:         A = diag(NX.ones((N-2,), p.dtype), -1)
232:         A[0,:] = -p[1:] / p[0]
233:         roots = eigvals(A)
234:     else:
235:         roots = NX.array([])
236: 
237:     # tack any zeros onto the back of the array
238:     roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
239:     return roots
240: 
241: def polyint(p, m=1, k=None):
242:     '''
243:     Return an antiderivative (indefinite integral) of a polynomial.
244: 
245:     The returned order `m` antiderivative `P` of polynomial `p` satisfies
246:     :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
247:     integration constants `k`. The constants determine the low-order
248:     polynomial part
249: 
250:     .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}
251: 
252:     of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.
253: 
254:     Parameters
255:     ----------
256:     p : array_like or poly1d
257:         Polynomial to differentiate.
258:         A sequence is interpreted as polynomial coefficients, see `poly1d`.
259:     m : int, optional
260:         Order of the antiderivative. (Default: 1)
261:     k : list of `m` scalars or scalar, optional
262:         Integration constants. They are given in the order of integration:
263:         those corresponding to highest-order terms come first.
264: 
265:         If ``None`` (default), all constants are assumed to be zero.
266:         If `m = 1`, a single scalar can be given instead of a list.
267: 
268:     See Also
269:     --------
270:     polyder : derivative of a polynomial
271:     poly1d.integ : equivalent method
272: 
273:     Examples
274:     --------
275:     The defining property of the antiderivative:
276: 
277:     >>> p = np.poly1d([1,1,1])
278:     >>> P = np.polyint(p)
279:     >>> P
280:     poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ])
281:     >>> np.polyder(P) == p
282:     True
283: 
284:     The integration constants default to zero, but can be specified:
285: 
286:     >>> P = np.polyint(p, 3)
287:     >>> P(0)
288:     0.0
289:     >>> np.polyder(P)(0)
290:     0.0
291:     >>> np.polyder(P, 2)(0)
292:     0.0
293:     >>> P = np.polyint(p, 3, k=[6,5,3])
294:     >>> P
295:     poly1d([ 0.01666667,  0.04166667,  0.16666667,  3. ,  5. ,  3. ])
296: 
297:     Note that 3 = 6 / 2!, and that the constants are given in the order of
298:     integrations. Constant of the highest-order polynomial term comes first:
299: 
300:     >>> np.polyder(P, 2)(0)
301:     6.0
302:     >>> np.polyder(P, 1)(0)
303:     5.0
304:     >>> P(0)
305:     3.0
306: 
307:     '''
308:     m = int(m)
309:     if m < 0:
310:         raise ValueError("Order of integral must be positive (see polyder)")
311:     if k is None:
312:         k = NX.zeros(m, float)
313:     k = atleast_1d(k)
314:     if len(k) == 1 and m > 1:
315:         k = k[0]*NX.ones(m, float)
316:     if len(k) < m:
317:         raise ValueError(
318:               "k must be a scalar or a rank-1 array of length 1 or >m.")
319: 
320:     truepoly = isinstance(p, poly1d)
321:     p = NX.asarray(p)
322:     if m == 0:
323:         if truepoly:
324:             return poly1d(p)
325:         return p
326:     else:
327:         # Note: this must work also with object and integer arrays
328:         y = NX.concatenate((p.__truediv__(NX.arange(len(p), 0, -1)), [k[0]]))
329:         val = polyint(y, m - 1, k=k[1:])
330:         if truepoly:
331:             return poly1d(val)
332:         return val
333: 
334: def polyder(p, m=1):
335:     '''
336:     Return the derivative of the specified order of a polynomial.
337: 
338:     Parameters
339:     ----------
340:     p : poly1d or sequence
341:         Polynomial to differentiate.
342:         A sequence is interpreted as polynomial coefficients, see `poly1d`.
343:     m : int, optional
344:         Order of differentiation (default: 1)
345: 
346:     Returns
347:     -------
348:     der : poly1d
349:         A new polynomial representing the derivative.
350: 
351:     See Also
352:     --------
353:     polyint : Anti-derivative of a polynomial.
354:     poly1d : Class for one-dimensional polynomials.
355: 
356:     Examples
357:     --------
358:     The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:
359: 
360:     >>> p = np.poly1d([1,1,1,1])
361:     >>> p2 = np.polyder(p)
362:     >>> p2
363:     poly1d([3, 2, 1])
364: 
365:     which evaluates to:
366: 
367:     >>> p2(2.)
368:     17.0
369: 
370:     We can verify this, approximating the derivative with
371:     ``(f(x + h) - f(x))/h``:
372: 
373:     >>> (p(2. + 0.001) - p(2.)) / 0.001
374:     17.007000999997857
375: 
376:     The fourth-order derivative of a 3rd-order polynomial is zero:
377: 
378:     >>> np.polyder(p, 2)
379:     poly1d([6, 2])
380:     >>> np.polyder(p, 3)
381:     poly1d([6])
382:     >>> np.polyder(p, 4)
383:     poly1d([ 0.])
384: 
385:     '''
386:     m = int(m)
387:     if m < 0:
388:         raise ValueError("Order of derivative must be positive (see polyint)")
389: 
390:     truepoly = isinstance(p, poly1d)
391:     p = NX.asarray(p)
392:     n = len(p) - 1
393:     y = p[:-1] * NX.arange(n, 0, -1)
394:     if m == 0:
395:         val = p
396:     else:
397:         val = polyder(y, m - 1)
398:     if truepoly:
399:         val = poly1d(val)
400:     return val
401: 
402: def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
403:     '''
404:     Least squares polynomial fit.
405: 
406:     Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
407:     to points `(x, y)`. Returns a vector of coefficients `p` that minimises
408:     the squared error.
409: 
410:     Parameters
411:     ----------
412:     x : array_like, shape (M,)
413:         x-coordinates of the M sample points ``(x[i], y[i])``.
414:     y : array_like, shape (M,) or (M, K)
415:         y-coordinates of the sample points. Several data sets of sample
416:         points sharing the same x-coordinates can be fitted at once by
417:         passing in a 2D-array that contains one dataset per column.
418:     deg : int
419:         Degree of the fitting polynomial
420:     rcond : float, optional
421:         Relative condition number of the fit. Singular values smaller than
422:         this relative to the largest singular value will be ignored. The
423:         default value is len(x)*eps, where eps is the relative precision of
424:         the float type, about 2e-16 in most cases.
425:     full : bool, optional
426:         Switch determining nature of return value. When it is False (the
427:         default) just the coefficients are returned, when True diagnostic
428:         information from the singular value decomposition is also returned.
429:     w : array_like, shape (M,), optional
430:         Weights to apply to the y-coordinates of the sample points. For
431:         gaussian uncertainties, use 1/sigma (not 1/sigma**2).
432:     cov : bool, optional
433:         Return the estimate and the covariance matrix of the estimate
434:         If full is True, then cov is not returned.
435: 
436:     Returns
437:     -------
438:     p : ndarray, shape (M,) or (M, K)
439:         Polynomial coefficients, highest power first.  If `y` was 2-D, the
440:         coefficients for `k`-th data set are in ``p[:,k]``.
441: 
442:     residuals, rank, singular_values, rcond :
443:         Present only if `full` = True.  Residuals of the least-squares fit,
444:         the effective rank of the scaled Vandermonde coefficient matrix,
445:         its singular values, and the specified value of `rcond`. For more
446:         details, see `linalg.lstsq`.
447: 
448:     V : ndarray, shape (M,M) or (M,M,K)
449:         Present only if `full` = False and `cov`=True.  The covariance
450:         matrix of the polynomial coefficient estimates.  The diagonal of
451:         this matrix are the variance estimates for each coefficient.  If y
452:         is a 2-D array, then the covariance matrix for the `k`-th data set
453:         are in ``V[:,:,k]``
454: 
455: 
456:     Warns
457:     -----
458:     RankWarning
459:         The rank of the coefficient matrix in the least-squares fit is
460:         deficient. The warning is only raised if `full` = False.
461: 
462:         The warnings can be turned off by
463: 
464:         >>> import warnings
465:         >>> warnings.simplefilter('ignore', np.RankWarning)
466: 
467:     See Also
468:     --------
469:     polyval : Compute polynomial values.
470:     linalg.lstsq : Computes a least-squares fit.
471:     scipy.interpolate.UnivariateSpline : Computes spline fits.
472: 
473:     Notes
474:     -----
475:     The solution minimizes the squared error
476: 
477:     .. math ::
478:         E = \\sum_{j=0}^k |p(x_j) - y_j|^2
479: 
480:     in the equations::
481: 
482:         x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
483:         x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
484:         ...
485:         x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]
486: 
487:     The coefficient matrix of the coefficients `p` is a Vandermonde matrix.
488: 
489:     `polyfit` issues a `RankWarning` when the least-squares fit is badly
490:     conditioned. This implies that the best fit is not well-defined due
491:     to numerical error. The results may be improved by lowering the polynomial
492:     degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
493:     can also be set to a value smaller than its default, but the resulting
494:     fit may be spurious: including contributions from the small singular
495:     values can add numerical noise to the result.
496: 
497:     Note that fitting polynomial coefficients is inherently badly conditioned
498:     when the degree of the polynomial is large or the interval of sample points
499:     is badly centered. The quality of the fit should always be checked in these
500:     cases. When polynomial fits are not satisfactory, splines may be a good
501:     alternative.
502: 
503:     References
504:     ----------
505:     .. [1] Wikipedia, "Curve fitting",
506:            http://en.wikipedia.org/wiki/Curve_fitting
507:     .. [2] Wikipedia, "Polynomial interpolation",
508:            http://en.wikipedia.org/wiki/Polynomial_interpolation
509: 
510:     Examples
511:     --------
512:     >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
513:     >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
514:     >>> z = np.polyfit(x, y, 3)
515:     >>> z
516:     array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])
517: 
518:     It is convenient to use `poly1d` objects for dealing with polynomials:
519: 
520:     >>> p = np.poly1d(z)
521:     >>> p(0.5)
522:     0.6143849206349179
523:     >>> p(3.5)
524:     -0.34732142857143039
525:     >>> p(10)
526:     22.579365079365115
527: 
528:     High-order polynomials may oscillate wildly:
529: 
530:     >>> p30 = np.poly1d(np.polyfit(x, y, 30))
531:     /... RankWarning: Polyfit may be poorly conditioned...
532:     >>> p30(4)
533:     -0.80000000000000204
534:     >>> p30(5)
535:     -0.99999999999999445
536:     >>> p30(4.5)
537:     -0.10547061179440398
538: 
539:     Illustration:
540: 
541:     >>> import matplotlib.pyplot as plt
542:     >>> xp = np.linspace(-2, 6, 100)
543:     >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
544:     >>> plt.ylim(-2,2)
545:     (-2, 2)
546:     >>> plt.show()
547: 
548:     '''
549:     order = int(deg) + 1
550:     x = NX.asarray(x) + 0.0
551:     y = NX.asarray(y) + 0.0
552: 
553:     # check arguments.
554:     if deg < 0:
555:         raise ValueError("expected deg >= 0")
556:     if x.ndim != 1:
557:         raise TypeError("expected 1D vector for x")
558:     if x.size == 0:
559:         raise TypeError("expected non-empty vector for x")
560:     if y.ndim < 1 or y.ndim > 2:
561:         raise TypeError("expected 1D or 2D array for y")
562:     if x.shape[0] != y.shape[0]:
563:         raise TypeError("expected x and y to have same length")
564: 
565:     # set rcond
566:     if rcond is None:
567:         rcond = len(x)*finfo(x.dtype).eps
568: 
569:     # set up least squares equation for powers of x
570:     lhs = vander(x, order)
571:     rhs = y
572: 
573:     # apply weighting
574:     if w is not None:
575:         w = NX.asarray(w) + 0.0
576:         if w.ndim != 1:
577:             raise TypeError("expected a 1-d array for weights")
578:         if w.shape[0] != y.shape[0]:
579:             raise TypeError("expected w and y to have the same length")
580:         lhs *= w[:, NX.newaxis]
581:         if rhs.ndim == 2:
582:             rhs *= w[:, NX.newaxis]
583:         else:
584:             rhs *= w
585: 
586:     # scale lhs to improve condition number and solve
587:     scale = NX.sqrt((lhs*lhs).sum(axis=0))
588:     lhs /= scale
589:     c, resids, rank, s = lstsq(lhs, rhs, rcond)
590:     c = (c.T/scale).T  # broadcast scale coefficients
591: 
592:     # warn on rank reduction, which indicates an ill conditioned matrix
593:     if rank != order and not full:
594:         msg = "Polyfit may be poorly conditioned"
595:         warnings.warn(msg, RankWarning)
596: 
597:     if full:
598:         return c, resids, rank, s, rcond
599:     elif cov:
600:         Vbase = inv(dot(lhs.T, lhs))
601:         Vbase /= NX.outer(scale, scale)
602:         # Some literature ignores the extra -2.0 factor in the denominator, but
603:         #  it is included here because the covariance of Multivariate Student-T
604:         #  (which is implied by a Bayesian uncertainty analysis) includes it.
605:         #  Plus, it gives a slightly more conservative estimate of uncertainty.
606:         fac = resids / (len(x) - order - 2.0)
607:         if y.ndim == 1:
608:             return c, Vbase * fac
609:         else:
610:             return c, Vbase[:,:, NX.newaxis] * fac
611:     else:
612:         return c
613: 
614: 
615: def polyval(p, x):
616:     '''
617:     Evaluate a polynomial at specific values.
618: 
619:     If `p` is of length N, this function returns the value:
620: 
621:         ``p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]``
622: 
623:     If `x` is a sequence, then `p(x)` is returned for each element of `x`.
624:     If `x` is another polynomial then the composite polynomial `p(x(t))`
625:     is returned.
626: 
627:     Parameters
628:     ----------
629:     p : array_like or poly1d object
630:        1D array of polynomial coefficients (including coefficients equal
631:        to zero) from highest degree to the constant term, or an
632:        instance of poly1d.
633:     x : array_like or poly1d object
634:        A number, an array of numbers, or an instance of poly1d, at
635:        which to evaluate `p`.
636: 
637:     Returns
638:     -------
639:     values : ndarray or poly1d
640:        If `x` is a poly1d instance, the result is the composition of the two
641:        polynomials, i.e., `x` is "substituted" in `p` and the simplified
642:        result is returned. In addition, the type of `x` - array_like or
643:        poly1d - governs the type of the output: `x` array_like => `values`
644:        array_like, `x` a poly1d object => `values` is also.
645: 
646:     See Also
647:     --------
648:     poly1d: A polynomial class.
649: 
650:     Notes
651:     -----
652:     Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
653:     for polynomials of high degree the values may be inaccurate due to
654:     rounding errors. Use carefully.
655: 
656:     References
657:     ----------
658:     .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
659:        trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
660:        Reinhold Co., 1985, pg. 720.
661: 
662:     Examples
663:     --------
664:     >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
665:     76
666:     >>> np.polyval([3,0,1], np.poly1d(5))
667:     poly1d([ 76.])
668:     >>> np.polyval(np.poly1d([3,0,1]), 5)
669:     76
670:     >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
671:     poly1d([ 76.])
672: 
673:     '''
674:     p = NX.asarray(p)
675:     if isinstance(x, poly1d):
676:         y = 0
677:     else:
678:         x = NX.asarray(x)
679:         y = NX.zeros_like(x)
680:     for i in range(len(p)):
681:         y = y * x + p[i]
682:     return y
683: 
684: def polyadd(a1, a2):
685:     '''
686:     Find the sum of two polynomials.
687: 
688:     Returns the polynomial resulting from the sum of two input polynomials.
689:     Each input must be either a poly1d object or a 1D sequence of polynomial
690:     coefficients, from highest to lowest degree.
691: 
692:     Parameters
693:     ----------
694:     a1, a2 : array_like or poly1d object
695:         Input polynomials.
696: 
697:     Returns
698:     -------
699:     out : ndarray or poly1d object
700:         The sum of the inputs. If either input is a poly1d object, then the
701:         output is also a poly1d object. Otherwise, it is a 1D array of
702:         polynomial coefficients from highest to lowest degree.
703: 
704:     See Also
705:     --------
706:     poly1d : A one-dimensional polynomial class.
707:     poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval
708: 
709:     Examples
710:     --------
711:     >>> np.polyadd([1, 2], [9, 5, 4])
712:     array([9, 6, 6])
713: 
714:     Using poly1d objects:
715: 
716:     >>> p1 = np.poly1d([1, 2])
717:     >>> p2 = np.poly1d([9, 5, 4])
718:     >>> print(p1)
719:     1 x + 2
720:     >>> print(p2)
721:        2
722:     9 x + 5 x + 4
723:     >>> print(np.polyadd(p1, p2))
724:        2
725:     9 x + 6 x + 6
726: 
727:     '''
728:     truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
729:     a1 = atleast_1d(a1)
730:     a2 = atleast_1d(a2)
731:     diff = len(a2) - len(a1)
732:     if diff == 0:
733:         val = a1 + a2
734:     elif diff > 0:
735:         zr = NX.zeros(diff, a1.dtype)
736:         val = NX.concatenate((zr, a1)) + a2
737:     else:
738:         zr = NX.zeros(abs(diff), a2.dtype)
739:         val = a1 + NX.concatenate((zr, a2))
740:     if truepoly:
741:         val = poly1d(val)
742:     return val
743: 
744: def polysub(a1, a2):
745:     '''
746:     Difference (subtraction) of two polynomials.
747: 
748:     Given two polynomials `a1` and `a2`, returns ``a1 - a2``.
749:     `a1` and `a2` can be either array_like sequences of the polynomials'
750:     coefficients (including coefficients equal to zero), or `poly1d` objects.
751: 
752:     Parameters
753:     ----------
754:     a1, a2 : array_like or poly1d
755:         Minuend and subtrahend polynomials, respectively.
756: 
757:     Returns
758:     -------
759:     out : ndarray or poly1d
760:         Array or `poly1d` object of the difference polynomial's coefficients.
761: 
762:     See Also
763:     --------
764:     polyval, polydiv, polymul, polyadd
765: 
766:     Examples
767:     --------
768:     .. math:: (2 x^2 + 10 x - 2) - (3 x^2 + 10 x -4) = (-x^2 + 2)
769: 
770:     >>> np.polysub([2, 10, -2], [3, 10, -4])
771:     array([-1,  0,  2])
772: 
773:     '''
774:     truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
775:     a1 = atleast_1d(a1)
776:     a2 = atleast_1d(a2)
777:     diff = len(a2) - len(a1)
778:     if diff == 0:
779:         val = a1 - a2
780:     elif diff > 0:
781:         zr = NX.zeros(diff, a1.dtype)
782:         val = NX.concatenate((zr, a1)) - a2
783:     else:
784:         zr = NX.zeros(abs(diff), a2.dtype)
785:         val = a1 - NX.concatenate((zr, a2))
786:     if truepoly:
787:         val = poly1d(val)
788:     return val
789: 
790: 
791: def polymul(a1, a2):
792:     '''
793:     Find the product of two polynomials.
794: 
795:     Finds the polynomial resulting from the multiplication of the two input
796:     polynomials. Each input must be either a poly1d object or a 1D sequence
797:     of polynomial coefficients, from highest to lowest degree.
798: 
799:     Parameters
800:     ----------
801:     a1, a2 : array_like or poly1d object
802:         Input polynomials.
803: 
804:     Returns
805:     -------
806:     out : ndarray or poly1d object
807:         The polynomial resulting from the multiplication of the inputs. If
808:         either inputs is a poly1d object, then the output is also a poly1d
809:         object. Otherwise, it is a 1D array of polynomial coefficients from
810:         highest to lowest degree.
811: 
812:     See Also
813:     --------
814:     poly1d : A one-dimensional polynomial class.
815:     poly, polyadd, polyder, polydiv, polyfit, polyint, polysub,
816:     polyval
817:     convolve : Array convolution. Same output as polymul, but has parameter
818:                for overlap mode.
819: 
820:     Examples
821:     --------
822:     >>> np.polymul([1, 2, 3], [9, 5, 1])
823:     array([ 9, 23, 38, 17,  3])
824: 
825:     Using poly1d objects:
826: 
827:     >>> p1 = np.poly1d([1, 2, 3])
828:     >>> p2 = np.poly1d([9, 5, 1])
829:     >>> print(p1)
830:        2
831:     1 x + 2 x + 3
832:     >>> print(p2)
833:        2
834:     9 x + 5 x + 1
835:     >>> print(np.polymul(p1, p2))
836:        4      3      2
837:     9 x + 23 x + 38 x + 17 x + 3
838: 
839:     '''
840:     truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
841:     a1, a2 = poly1d(a1), poly1d(a2)
842:     val = NX.convolve(a1, a2)
843:     if truepoly:
844:         val = poly1d(val)
845:     return val
846: 
847: def polydiv(u, v):
848:     '''
849:     Returns the quotient and remainder of polynomial division.
850: 
851:     The input arrays are the coefficients (including any coefficients
852:     equal to zero) of the "numerator" (dividend) and "denominator"
853:     (divisor) polynomials, respectively.
854: 
855:     Parameters
856:     ----------
857:     u : array_like or poly1d
858:         Dividend polynomial's coefficients.
859: 
860:     v : array_like or poly1d
861:         Divisor polynomial's coefficients.
862: 
863:     Returns
864:     -------
865:     q : ndarray
866:         Coefficients, including those equal to zero, of the quotient.
867:     r : ndarray
868:         Coefficients, including those equal to zero, of the remainder.
869: 
870:     See Also
871:     --------
872:     poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub,
873:     polyval
874: 
875:     Notes
876:     -----
877:     Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
878:     not equal `v.ndim`. In other words, all four possible combinations -
879:     ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
880:     ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.
881: 
882:     Examples
883:     --------
884:     .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25
885: 
886:     >>> x = np.array([3.0, 5.0, 2.0])
887:     >>> y = np.array([2.0, 1.0])
888:     >>> np.polydiv(x, y)
889:     (array([ 1.5 ,  1.75]), array([ 0.25]))
890: 
891:     '''
892:     truepoly = (isinstance(u, poly1d) or isinstance(u, poly1d))
893:     u = atleast_1d(u) + 0.0
894:     v = atleast_1d(v) + 0.0
895:     # w has the common type
896:     w = u[0] + v[0]
897:     m = len(u) - 1
898:     n = len(v) - 1
899:     scale = 1. / v[0]
900:     q = NX.zeros((max(m - n + 1, 1),), w.dtype)
901:     r = u.copy()
902:     for k in range(0, m-n+1):
903:         d = scale * r[k]
904:         q[k] = d
905:         r[k:k+n+1] -= d*v
906:     while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
907:         r = r[1:]
908:     if truepoly:
909:         return poly1d(q), poly1d(r)
910:     return q, r
911: 
912: _poly_mat = re.compile(r"[*][*]([0-9]*)")
913: def _raise_power(astr, wrap=70):
914:     n = 0
915:     line1 = ''
916:     line2 = ''
917:     output = ' '
918:     while True:
919:         mat = _poly_mat.search(astr, n)
920:         if mat is None:
921:             break
922:         span = mat.span()
923:         power = mat.groups()[0]
924:         partstr = astr[n:span[0]]
925:         n = span[1]
926:         toadd2 = partstr + ' '*(len(power)-1)
927:         toadd1 = ' '*(len(partstr)-1) + power
928:         if ((len(line2) + len(toadd2) > wrap) or
929:                 (len(line1) + len(toadd1) > wrap)):
930:             output += line1 + "\n" + line2 + "\n "
931:             line1 = toadd1
932:             line2 = toadd2
933:         else:
934:             line2 += partstr + ' '*(len(power)-1)
935:             line1 += ' '*(len(partstr)-1) + power
936:     output += line1 + "\n" + line2
937:     return output + astr[n:]
938: 
939: 
940: class poly1d(object):
941:     '''
942:     A one-dimensional polynomial class.
943: 
944:     A convenience class, used to encapsulate "natural" operations on
945:     polynomials so that said operations may take on their customary
946:     form in code (see Examples).
947: 
948:     Parameters
949:     ----------
950:     c_or_r : array_like
951:         The polynomial's coefficients, in decreasing powers, or if
952:         the value of the second parameter is True, the polynomial's
953:         roots (values where the polynomial evaluates to 0).  For example,
954:         ``poly1d([1, 2, 3])`` returns an object that represents
955:         :math:`x^2 + 2x + 3`, whereas ``poly1d([1, 2, 3], True)`` returns
956:         one that represents :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6`.
957:     r : bool, optional
958:         If True, `c_or_r` specifies the polynomial's roots; the default
959:         is False.
960:     variable : str, optional
961:         Changes the variable used when printing `p` from `x` to `variable`
962:         (see Examples).
963: 
964:     Examples
965:     --------
966:     Construct the polynomial :math:`x^2 + 2x + 3`:
967: 
968:     >>> p = np.poly1d([1, 2, 3])
969:     >>> print(np.poly1d(p))
970:        2
971:     1 x + 2 x + 3
972: 
973:     Evaluate the polynomial at :math:`x = 0.5`:
974: 
975:     >>> p(0.5)
976:     4.25
977: 
978:     Find the roots:
979: 
980:     >>> p.r
981:     array([-1.+1.41421356j, -1.-1.41421356j])
982:     >>> p(p.r)
983:     array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j])
984: 
985:     These numbers in the previous line represent (0, 0) to machine precision
986: 
987:     Show the coefficients:
988: 
989:     >>> p.c
990:     array([1, 2, 3])
991: 
992:     Display the order (the leading zero-coefficients are removed):
993: 
994:     >>> p.order
995:     2
996: 
997:     Show the coefficient of the k-th power in the polynomial
998:     (which is equivalent to ``p.c[-(i+1)]``):
999: 
1000:     >>> p[1]
1001:     2
1002: 
1003:     Polynomials can be added, subtracted, multiplied, and divided
1004:     (returns quotient and remainder):
1005: 
1006:     >>> p * p
1007:     poly1d([ 1,  4, 10, 12,  9])
1008: 
1009:     >>> (p**3 + 4) / p
1010:     (poly1d([  1.,   4.,  10.,  12.,   9.]), poly1d([ 4.]))
1011: 
1012:     ``asarray(p)`` gives the coefficient array, so polynomials can be
1013:     used in all functions that accept arrays:
1014: 
1015:     >>> p**2 # square of polynomial
1016:     poly1d([ 1,  4, 10, 12,  9])
1017: 
1018:     >>> np.square(p) # square of individual coefficients
1019:     array([1, 4, 9])
1020: 
1021:     The variable used in the string representation of `p` can be modified,
1022:     using the `variable` parameter:
1023: 
1024:     >>> p = np.poly1d([1,2,3], variable='z')
1025:     >>> print(p)
1026:        2
1027:     1 z + 2 z + 3
1028: 
1029:     Construct a polynomial from its roots:
1030: 
1031:     >>> np.poly1d([1, 2], True)
1032:     poly1d([ 1, -3,  2])
1033: 
1034:     This is the same polynomial as obtained by:
1035: 
1036:     >>> np.poly1d([1, -1]) * np.poly1d([1, -2])
1037:     poly1d([ 1, -3,  2])
1038: 
1039:     '''
1040:     coeffs = None
1041:     order = None
1042:     variable = None
1043:     __hash__ = None
1044: 
1045:     def __init__(self, c_or_r, r=0, variable=None):
1046:         if isinstance(c_or_r, poly1d):
1047:             for key in c_or_r.__dict__.keys():
1048:                 self.__dict__[key] = c_or_r.__dict__[key]
1049:             if variable is not None:
1050:                 self.__dict__['variable'] = variable
1051:             return
1052:         if r:
1053:             c_or_r = poly(c_or_r)
1054:         c_or_r = atleast_1d(c_or_r)
1055:         if len(c_or_r.shape) > 1:
1056:             raise ValueError("Polynomial must be 1d only.")
1057:         c_or_r = trim_zeros(c_or_r, trim='f')
1058:         if len(c_or_r) == 0:
1059:             c_or_r = NX.array([0.])
1060:         self.__dict__['coeffs'] = c_or_r
1061:         self.__dict__['order'] = len(c_or_r) - 1
1062:         if variable is None:
1063:             variable = 'x'
1064:         self.__dict__['variable'] = variable
1065: 
1066:     def __array__(self, t=None):
1067:         if t:
1068:             return NX.asarray(self.coeffs, t)
1069:         else:
1070:             return NX.asarray(self.coeffs)
1071: 
1072:     def __repr__(self):
1073:         vals = repr(self.coeffs)
1074:         vals = vals[6:-1]
1075:         return "poly1d(%s)" % vals
1076: 
1077:     def __len__(self):
1078:         return self.order
1079: 
1080:     def __str__(self):
1081:         thestr = "0"
1082:         var = self.variable
1083: 
1084:         # Remove leading zeros
1085:         coeffs = self.coeffs[NX.logical_or.accumulate(self.coeffs != 0)]
1086:         N = len(coeffs)-1
1087: 
1088:         def fmt_float(q):
1089:             s = '%.4g' % q
1090:             if s.endswith('.0000'):
1091:                 s = s[:-5]
1092:             return s
1093: 
1094:         for k in range(len(coeffs)):
1095:             if not iscomplex(coeffs[k]):
1096:                 coefstr = fmt_float(real(coeffs[k]))
1097:             elif real(coeffs[k]) == 0:
1098:                 coefstr = '%sj' % fmt_float(imag(coeffs[k]))
1099:             else:
1100:                 coefstr = '(%s + %sj)' % (fmt_float(real(coeffs[k])),
1101:                                           fmt_float(imag(coeffs[k])))
1102: 
1103:             power = (N-k)
1104:             if power == 0:
1105:                 if coefstr != '0':
1106:                     newstr = '%s' % (coefstr,)
1107:                 else:
1108:                     if k == 0:
1109:                         newstr = '0'
1110:                     else:
1111:                         newstr = ''
1112:             elif power == 1:
1113:                 if coefstr == '0':
1114:                     newstr = ''
1115:                 elif coefstr == 'b':
1116:                     newstr = var
1117:                 else:
1118:                     newstr = '%s %s' % (coefstr, var)
1119:             else:
1120:                 if coefstr == '0':
1121:                     newstr = ''
1122:                 elif coefstr == 'b':
1123:                     newstr = '%s**%d' % (var, power,)
1124:                 else:
1125:                     newstr = '%s %s**%d' % (coefstr, var, power)
1126: 
1127:             if k > 0:
1128:                 if newstr != '':
1129:                     if newstr.startswith('-'):
1130:                         thestr = "%s - %s" % (thestr, newstr[1:])
1131:                     else:
1132:                         thestr = "%s + %s" % (thestr, newstr)
1133:             else:
1134:                 thestr = newstr
1135:         return _raise_power(thestr)
1136: 
1137:     def __call__(self, val):
1138:         return polyval(self.coeffs, val)
1139: 
1140:     def __neg__(self):
1141:         return poly1d(-self.coeffs)
1142: 
1143:     def __pos__(self):
1144:         return self
1145: 
1146:     def __mul__(self, other):
1147:         if isscalar(other):
1148:             return poly1d(self.coeffs * other)
1149:         else:
1150:             other = poly1d(other)
1151:             return poly1d(polymul(self.coeffs, other.coeffs))
1152: 
1153:     def __rmul__(self, other):
1154:         if isscalar(other):
1155:             return poly1d(other * self.coeffs)
1156:         else:
1157:             other = poly1d(other)
1158:             return poly1d(polymul(self.coeffs, other.coeffs))
1159: 
1160:     def __add__(self, other):
1161:         other = poly1d(other)
1162:         return poly1d(polyadd(self.coeffs, other.coeffs))
1163: 
1164:     def __radd__(self, other):
1165:         other = poly1d(other)
1166:         return poly1d(polyadd(self.coeffs, other.coeffs))
1167: 
1168:     def __pow__(self, val):
1169:         if not isscalar(val) or int(val) != val or val < 0:
1170:             raise ValueError("Power to non-negative integers only.")
1171:         res = [1]
1172:         for _ in range(val):
1173:             res = polymul(self.coeffs, res)
1174:         return poly1d(res)
1175: 
1176:     def __sub__(self, other):
1177:         other = poly1d(other)
1178:         return poly1d(polysub(self.coeffs, other.coeffs))
1179: 
1180:     def __rsub__(self, other):
1181:         other = poly1d(other)
1182:         return poly1d(polysub(other.coeffs, self.coeffs))
1183: 
1184:     def __div__(self, other):
1185:         if isscalar(other):
1186:             return poly1d(self.coeffs/other)
1187:         else:
1188:             other = poly1d(other)
1189:             return polydiv(self, other)
1190: 
1191:     __truediv__ = __div__
1192: 
1193:     def __rdiv__(self, other):
1194:         if isscalar(other):
1195:             return poly1d(other/self.coeffs)
1196:         else:
1197:             other = poly1d(other)
1198:             return polydiv(other, self)
1199: 
1200:     __rtruediv__ = __rdiv__
1201: 
1202:     def __eq__(self, other):
1203:         if self.coeffs.shape != other.coeffs.shape:
1204:             return False
1205:         return (self.coeffs == other.coeffs).all()
1206: 
1207:     def __ne__(self, other):
1208:         return not self.__eq__(other)
1209: 
1210:     def __setattr__(self, key, val):
1211:         raise ValueError("Attributes cannot be changed this way.")
1212: 
1213:     def __getattr__(self, key):
1214:         if key in ['r', 'roots']:
1215:             return roots(self.coeffs)
1216:         elif key in ['c', 'coef', 'coefficients']:
1217:             return self.coeffs
1218:         elif key in ['o']:
1219:             return self.order
1220:         else:
1221:             try:
1222:                 return self.__dict__[key]
1223:             except KeyError:
1224:                 raise AttributeError(
1225:                     "'%s' has no attribute '%s'" % (self.__class__, key))
1226: 
1227:     def __getitem__(self, val):
1228:         ind = self.order - val
1229:         if val > self.order:
1230:             return 0
1231:         if val < 0:
1232:             return 0
1233:         return self.coeffs[ind]
1234: 
1235:     def __setitem__(self, key, val):
1236:         ind = self.order - key
1237:         if key < 0:
1238:             raise ValueError("Does not support negative powers.")
1239:         if key > self.order:
1240:             zr = NX.zeros(key-self.order, self.coeffs.dtype)
1241:             self.__dict__['coeffs'] = NX.concatenate((zr, self.coeffs))
1242:             self.__dict__['order'] = key
1243:             ind = 0
1244:         self.__dict__['coeffs'][ind] = val
1245:         return
1246: 
1247:     def __iter__(self):
1248:         return iter(self.coeffs)
1249: 
1250:     def integ(self, m=1, k=0):
1251:         '''
1252:         Return an antiderivative (indefinite integral) of this polynomial.
1253: 
1254:         Refer to `polyint` for full documentation.
1255: 
1256:         See Also
1257:         --------
1258:         polyint : equivalent function
1259: 
1260:         '''
1261:         return poly1d(polyint(self.coeffs, m=m, k=k))
1262: 
1263:     def deriv(self, m=1):
1264:         '''
1265:         Return a derivative of this polynomial.
1266: 
1267:         Refer to `polyder` for full documentation.
1268: 
1269:         See Also
1270:         --------
1271:         polyder : equivalent function
1272: 
1273:         '''
1274:         return poly1d(polyder(self.coeffs, m=m))
1275: 
1276: # Stuff to do on module import
1277: 
1278: warnings.simplefilter('always', RankWarning)
1279: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_120583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nFunctions to operate on polynomials.\n\n')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd', 'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d', 'polyfit', 'RankWarning']
module_type_store.set_exportable_members(['poly', 'roots', 'polyint', 'polyder', 'polyadd', 'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d', 'polyfit', 'RankWarning'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_120584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_120585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120585)
# Adding element type (line 7)
str_120586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'str', 'roots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120586)
# Adding element type (line 7)
str_120587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 28), 'str', 'polyint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120587)
# Adding element type (line 7)
str_120588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 39), 'str', 'polyder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120588)
# Adding element type (line 7)
str_120589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 50), 'str', 'polyadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120589)
# Adding element type (line 7)
str_120590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'polysub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120590)
# Adding element type (line 7)
str_120591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 22), 'str', 'polymul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120591)
# Adding element type (line 7)
str_120592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'str', 'polydiv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120592)
# Adding element type (line 7)
str_120593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 44), 'str', 'polyval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120593)
# Adding element type (line 7)
str_120594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 55), 'str', 'poly1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120594)
# Adding element type (line 7)
str_120595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'polyfit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120595)
# Adding element type (line 7)
str_120596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 22), 'str', 'RankWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_120584, str_120596)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_120584)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import re' statement (line 11)
import re

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy.core.numeric' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_120597 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.numeric')

if (type(import_120597) is not StypyTypeError):

    if (import_120597 != 'pyd_module'):
        __import__(import_120597)
        sys_modules_120598 = sys.modules[import_120597]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'NX', sys_modules_120598.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as NX

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'NX', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.numeric', import_120597)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.core import isscalar, abs, finfo, atleast_1d, hstack, dot, array, ones' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_120599 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core')

if (type(import_120599) is not StypyTypeError):

    if (import_120599 != 'pyd_module'):
        __import__(import_120599)
        sys_modules_120600 = sys.modules[import_120599]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core', sys_modules_120600.module_type_store, module_type_store, ['isscalar', 'abs', 'finfo', 'atleast_1d', 'hstack', 'dot', 'array', 'ones'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_120600, sys_modules_120600.module_type_store, module_type_store)
    else:
        from numpy.core import isscalar, abs, finfo, atleast_1d, hstack, dot, array, ones

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core', None, module_type_store, ['isscalar', 'abs', 'finfo', 'atleast_1d', 'hstack', 'dot', 'array', 'ones'], [isscalar, abs, finfo, atleast_1d, hstack, dot, array, ones])

else:
    # Assigning a type to the variable 'numpy.core' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core', import_120599)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from numpy.lib.twodim_base import diag, vander' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_120601 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib.twodim_base')

if (type(import_120601) is not StypyTypeError):

    if (import_120601 != 'pyd_module'):
        __import__(import_120601)
        sys_modules_120602 = sys.modules[import_120601]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib.twodim_base', sys_modules_120602.module_type_store, module_type_store, ['diag', 'vander'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_120602, sys_modules_120602.module_type_store, module_type_store)
    else:
        from numpy.lib.twodim_base import diag, vander

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib.twodim_base', None, module_type_store, ['diag', 'vander'], [diag, vander])

else:
    # Assigning a type to the variable 'numpy.lib.twodim_base' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib.twodim_base', import_120601)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.lib.function_base import trim_zeros, sort_complex' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_120603 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.function_base')

if (type(import_120603) is not StypyTypeError):

    if (import_120603 != 'pyd_module'):
        __import__(import_120603)
        sys_modules_120604 = sys.modules[import_120603]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.function_base', sys_modules_120604.module_type_store, module_type_store, ['trim_zeros', 'sort_complex'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_120604, sys_modules_120604.module_type_store, module_type_store)
    else:
        from numpy.lib.function_base import trim_zeros, sort_complex

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.function_base', None, module_type_store, ['trim_zeros', 'sort_complex'], [trim_zeros, sort_complex])

else:
    # Assigning a type to the variable 'numpy.lib.function_base' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.function_base', import_120603)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy.lib.type_check import iscomplex, real, imag, mintypecode' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_120605 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.lib.type_check')

if (type(import_120605) is not StypyTypeError):

    if (import_120605 != 'pyd_module'):
        __import__(import_120605)
        sys_modules_120606 = sys.modules[import_120605]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.lib.type_check', sys_modules_120606.module_type_store, module_type_store, ['iscomplex', 'real', 'imag', 'mintypecode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_120606, sys_modules_120606.module_type_store, module_type_store)
    else:
        from numpy.lib.type_check import iscomplex, real, imag, mintypecode

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.lib.type_check', None, module_type_store, ['iscomplex', 'real', 'imag', 'mintypecode'], [iscomplex, real, imag, mintypecode])

else:
    # Assigning a type to the variable 'numpy.lib.type_check' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.lib.type_check', import_120605)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.linalg import eigvals, lstsq, inv' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_120607 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.linalg')

if (type(import_120607) is not StypyTypeError):

    if (import_120607 != 'pyd_module'):
        __import__(import_120607)
        sys_modules_120608 = sys.modules[import_120607]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.linalg', sys_modules_120608.module_type_store, module_type_store, ['eigvals', 'lstsq', 'inv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_120608, sys_modules_120608.module_type_store, module_type_store)
    else:
        from numpy.linalg import eigvals, lstsq, inv

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.linalg', None, module_type_store, ['eigvals', 'lstsq', 'inv'], [eigvals, lstsq, inv])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.linalg', import_120607)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

# Declaration of the 'RankWarning' class
# Getting the type of 'UserWarning' (line 22)
UserWarning_120609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'UserWarning')

class RankWarning(UserWarning_120609, ):
    str_120610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '\n    Issued by `polyfit` when the Vandermonde matrix is rank deficient.\n\n    For more information, a way to suppress the warning, and an example of\n    `RankWarning` being issued, see `polyfit`.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 0, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RankWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RankWarning' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'RankWarning', RankWarning)

@norecursion
def poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poly'
    module_type_store = module_type_store.open_function_context('poly', 32, 0, False)
    
    # Passed parameters checking function
    poly.stypy_localization = localization
    poly.stypy_type_of_self = None
    poly.stypy_type_store = module_type_store
    poly.stypy_function_name = 'poly'
    poly.stypy_param_names_list = ['seq_of_zeros']
    poly.stypy_varargs_param_name = None
    poly.stypy_kwargs_param_name = None
    poly.stypy_call_defaults = defaults
    poly.stypy_call_varargs = varargs
    poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poly', ['seq_of_zeros'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poly', localization, ['seq_of_zeros'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poly(...)' code ##################

    str_120611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'str', '\n    Find the coefficients of a polynomial with the given sequence of roots.\n\n    Returns the coefficients of the polynomial whose leading coefficient\n    is one for the given sequence of zeros (multiple roots must be included\n    in the sequence as many times as their multiplicity; see Examples).\n    A square matrix (or array, which will be treated as a matrix) can also\n    be given, in which case the coefficients of the characteristic polynomial\n    of the matrix are returned.\n\n    Parameters\n    ----------\n    seq_of_zeros : array_like, shape (N,) or (N, N)\n        A sequence of polynomial roots, or a square array or matrix object.\n\n    Returns\n    -------\n    c : ndarray\n        1D array of polynomial coefficients from highest to lowest degree:\n\n        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``\n        where c[0] always equals 1.\n\n    Raises\n    ------\n    ValueError\n        If input is the wrong shape (the input must be a 1-D or square\n        2-D array).\n\n    See Also\n    --------\n    polyval : Compute polynomial values.\n    roots : Return the roots of a polynomial.\n    polyfit : Least squares polynomial fit.\n    poly1d : A one-dimensional polynomial class.\n\n    Notes\n    -----\n    Specifying the roots of a polynomial still leaves one degree of\n    freedom, typically represented by an undetermined leading\n    coefficient. [1]_ In the case of this function, that coefficient -\n    the first one in the returned array - is always taken as one. (If\n    for some reason you have one other point, the only automatic way\n    presently to leverage that information is to use ``polyfit``.)\n\n    The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`\n    matrix **A** is given by\n\n        :math:`p_a(t) = \\mathrm{det}(t\\, \\mathbf{I} - \\mathbf{A})`,\n\n    where **I** is the `n`-by-`n` identity matrix. [2]_\n\n    References\n    ----------\n    .. [1] M. Sullivan and M. Sullivan, III, "Algebra and Trignometry,\n       Enhanced With Graphing Utilities," Prentice-Hall, pg. 318, 1996.\n\n    .. [2] G. Strang, "Linear Algebra and Its Applications, 2nd Edition,"\n       Academic Press, pg. 182, 1980.\n\n    Examples\n    --------\n    Given a sequence of a polynomial\'s zeros:\n\n    >>> np.poly((0, 0, 0)) # Multiple root example\n    array([1, 0, 0, 0])\n\n    The line above represents z**3 + 0*z**2 + 0*z + 0.\n\n    >>> np.poly((-1./2, 0, 1./2))\n    array([ 1.  ,  0.  , -0.25,  0.  ])\n\n    The line above represents z**3 - z/4\n\n    >>> np.poly((np.random.random(1.)[0], 0, np.random.random(1.)[0]))\n    array([ 1.        , -0.77086955,  0.08618131,  0.        ]) #random\n\n    Given a square array object:\n\n    >>> P = np.array([[0, 1./3], [-1./2, 0]])\n    >>> np.poly(P)\n    array([ 1.        ,  0.        ,  0.16666667])\n\n    Or a square matrix object:\n\n    >>> np.poly(np.matrix(P))\n    array([ 1.        ,  0.        ,  0.16666667])\n\n    Note how in all cases the leading coefficient is always 1.\n\n    ')
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to atleast_1d(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'seq_of_zeros' (line 124)
    seq_of_zeros_120613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'seq_of_zeros', False)
    # Processing the call keyword arguments (line 124)
    kwargs_120614 = {}
    # Getting the type of 'atleast_1d' (line 124)
    atleast_1d_120612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 124)
    atleast_1d_call_result_120615 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), atleast_1d_120612, *[seq_of_zeros_120613], **kwargs_120614)
    
    # Assigning a type to the variable 'seq_of_zeros' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'seq_of_zeros', atleast_1d_call_result_120615)
    
    # Assigning a Attribute to a Name (line 125):
    
    # Assigning a Attribute to a Name (line 125):
    # Getting the type of 'seq_of_zeros' (line 125)
    seq_of_zeros_120616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'seq_of_zeros')
    # Obtaining the member 'shape' of a type (line 125)
    shape_120617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 9), seq_of_zeros_120616, 'shape')
    # Assigning a type to the variable 'sh' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'sh', shape_120617)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'sh' (line 127)
    sh_120619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'sh', False)
    # Processing the call keyword arguments (line 127)
    kwargs_120620 = {}
    # Getting the type of 'len' (line 127)
    len_120618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 7), 'len', False)
    # Calling len(args, kwargs) (line 127)
    len_call_result_120621 = invoke(stypy.reporting.localization.Localization(__file__, 127, 7), len_120618, *[sh_120619], **kwargs_120620)
    
    int_120622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'int')
    # Applying the binary operator '==' (line 127)
    result_eq_120623 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), '==', len_call_result_120621, int_120622)
    
    
    
    # Obtaining the type of the subscript
    int_120624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
    # Getting the type of 'sh' (line 127)
    sh_120625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'sh')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___120626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), sh_120625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_120627 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), getitem___120626, int_120624)
    
    
    # Obtaining the type of the subscript
    int_120628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 36), 'int')
    # Getting the type of 'sh' (line 127)
    sh_120629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'sh')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___120630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 33), sh_120629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_120631 = invoke(stypy.reporting.localization.Localization(__file__, 127, 33), getitem___120630, int_120628)
    
    # Applying the binary operator '==' (line 127)
    result_eq_120632 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 24), '==', subscript_call_result_120627, subscript_call_result_120631)
    
    # Applying the binary operator 'and' (line 127)
    result_and_keyword_120633 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), 'and', result_eq_120623, result_eq_120632)
    
    
    # Obtaining the type of the subscript
    int_120634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 46), 'int')
    # Getting the type of 'sh' (line 127)
    sh_120635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 43), 'sh')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___120636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 43), sh_120635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_120637 = invoke(stypy.reporting.localization.Localization(__file__, 127, 43), getitem___120636, int_120634)
    
    int_120638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 52), 'int')
    # Applying the binary operator '!=' (line 127)
    result_ne_120639 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 43), '!=', subscript_call_result_120637, int_120638)
    
    # Applying the binary operator 'and' (line 127)
    result_and_keyword_120640 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), 'and', result_and_keyword_120633, result_ne_120639)
    
    # Testing the type of an if condition (line 127)
    if_condition_120641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 4), result_and_keyword_120640)
    # Assigning a type to the variable 'if_condition_120641' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'if_condition_120641', if_condition_120641)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to eigvals(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'seq_of_zeros' (line 128)
    seq_of_zeros_120643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'seq_of_zeros', False)
    # Processing the call keyword arguments (line 128)
    kwargs_120644 = {}
    # Getting the type of 'eigvals' (line 128)
    eigvals_120642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'eigvals', False)
    # Calling eigvals(args, kwargs) (line 128)
    eigvals_call_result_120645 = invoke(stypy.reporting.localization.Localization(__file__, 128, 23), eigvals_120642, *[seq_of_zeros_120643], **kwargs_120644)
    
    # Assigning a type to the variable 'seq_of_zeros' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'seq_of_zeros', eigvals_call_result_120645)
    # SSA branch for the else part of an if statement (line 127)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'sh' (line 129)
    sh_120647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'sh', False)
    # Processing the call keyword arguments (line 129)
    kwargs_120648 = {}
    # Getting the type of 'len' (line 129)
    len_120646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'len', False)
    # Calling len(args, kwargs) (line 129)
    len_call_result_120649 = invoke(stypy.reporting.localization.Localization(__file__, 129, 9), len_120646, *[sh_120647], **kwargs_120648)
    
    int_120650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'int')
    # Applying the binary operator '==' (line 129)
    result_eq_120651 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 9), '==', len_call_result_120649, int_120650)
    
    # Testing the type of an if condition (line 129)
    if_condition_120652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 9), result_eq_120651)
    # Assigning a type to the variable 'if_condition_120652' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'if_condition_120652', if_condition_120652)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 130):
    
    # Assigning a Attribute to a Name (line 130):
    # Getting the type of 'seq_of_zeros' (line 130)
    seq_of_zeros_120653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'seq_of_zeros')
    # Obtaining the member 'dtype' of a type (line 130)
    dtype_120654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 13), seq_of_zeros_120653, 'dtype')
    # Assigning a type to the variable 'dt' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'dt', dtype_120654)
    
    
    # Getting the type of 'dt' (line 132)
    dt_120655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'dt')
    # Getting the type of 'object' (line 132)
    object_120656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'object')
    # Applying the binary operator '!=' (line 132)
    result_ne_120657 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), '!=', dt_120655, object_120656)
    
    # Testing the type of an if condition (line 132)
    if_condition_120658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_ne_120657)
    # Assigning a type to the variable 'if_condition_120658' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_120658', if_condition_120658)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to astype(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Call to mintypecode(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'dt' (line 133)
    dt_120662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 59), 'dt', False)
    # Obtaining the member 'char' of a type (line 133)
    char_120663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 59), dt_120662, 'char')
    # Processing the call keyword arguments (line 133)
    kwargs_120664 = {}
    # Getting the type of 'mintypecode' (line 133)
    mintypecode_120661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'mintypecode', False)
    # Calling mintypecode(args, kwargs) (line 133)
    mintypecode_call_result_120665 = invoke(stypy.reporting.localization.Localization(__file__, 133, 47), mintypecode_120661, *[char_120663], **kwargs_120664)
    
    # Processing the call keyword arguments (line 133)
    kwargs_120666 = {}
    # Getting the type of 'seq_of_zeros' (line 133)
    seq_of_zeros_120659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'seq_of_zeros', False)
    # Obtaining the member 'astype' of a type (line 133)
    astype_120660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 27), seq_of_zeros_120659, 'astype')
    # Calling astype(args, kwargs) (line 133)
    astype_call_result_120667 = invoke(stypy.reporting.localization.Localization(__file__, 133, 27), astype_120660, *[mintypecode_call_result_120665], **kwargs_120666)
    
    # Assigning a type to the variable 'seq_of_zeros' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'seq_of_zeros', astype_call_result_120667)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 129)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 135)
    # Processing the call arguments (line 135)
    str_120669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'str', 'input must be 1d or non-empty square 2d array.')
    # Processing the call keyword arguments (line 135)
    kwargs_120670 = {}
    # Getting the type of 'ValueError' (line 135)
    ValueError_120668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 135)
    ValueError_call_result_120671 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), ValueError_120668, *[str_120669], **kwargs_120670)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 8), ValueError_call_result_120671, 'raise parameter', BaseException)
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'seq_of_zeros' (line 137)
    seq_of_zeros_120673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'seq_of_zeros', False)
    # Processing the call keyword arguments (line 137)
    kwargs_120674 = {}
    # Getting the type of 'len' (line 137)
    len_120672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'len', False)
    # Calling len(args, kwargs) (line 137)
    len_call_result_120675 = invoke(stypy.reporting.localization.Localization(__file__, 137, 7), len_120672, *[seq_of_zeros_120673], **kwargs_120674)
    
    int_120676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 28), 'int')
    # Applying the binary operator '==' (line 137)
    result_eq_120677 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), '==', len_call_result_120675, int_120676)
    
    # Testing the type of an if condition (line 137)
    if_condition_120678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_eq_120677)
    # Assigning a type to the variable 'if_condition_120678' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_120678', if_condition_120678)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    float_120679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 15), 'float')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', float_120679)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 139):
    
    # Assigning a Attribute to a Name (line 139):
    # Getting the type of 'seq_of_zeros' (line 139)
    seq_of_zeros_120680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'seq_of_zeros')
    # Obtaining the member 'dtype' of a type (line 139)
    dtype_120681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 9), seq_of_zeros_120680, 'dtype')
    # Assigning a type to the variable 'dt' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'dt', dtype_120681)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to ones(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Obtaining an instance of the builtin type 'tuple' (line 140)
    tuple_120683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 140)
    # Adding element type (line 140)
    int_120684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 14), tuple_120683, int_120684)
    
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'dt' (line 140)
    dt_120685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'dt', False)
    keyword_120686 = dt_120685
    kwargs_120687 = {'dtype': keyword_120686}
    # Getting the type of 'ones' (line 140)
    ones_120682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'ones', False)
    # Calling ones(args, kwargs) (line 140)
    ones_call_result_120688 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), ones_120682, *[tuple_120683], **kwargs_120687)
    
    # Assigning a type to the variable 'a' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'a', ones_call_result_120688)
    
    
    # Call to range(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Call to len(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'seq_of_zeros' (line 141)
    seq_of_zeros_120691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'seq_of_zeros', False)
    # Processing the call keyword arguments (line 141)
    kwargs_120692 = {}
    # Getting the type of 'len' (line 141)
    len_120690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'len', False)
    # Calling len(args, kwargs) (line 141)
    len_call_result_120693 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), len_120690, *[seq_of_zeros_120691], **kwargs_120692)
    
    # Processing the call keyword arguments (line 141)
    kwargs_120694 = {}
    # Getting the type of 'range' (line 141)
    range_120689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'range', False)
    # Calling range(args, kwargs) (line 141)
    range_call_result_120695 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), range_120689, *[len_call_result_120693], **kwargs_120694)
    
    # Testing the type of a for loop iterable (line 141)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 4), range_call_result_120695)
    # Getting the type of the for loop variable (line 141)
    for_loop_var_120696 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 4), range_call_result_120695)
    # Assigning a type to the variable 'k' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'k', for_loop_var_120696)
    # SSA begins for a for statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 142):
    
    # Assigning a Call to a Name (line 142):
    
    # Call to convolve(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'a' (line 142)
    a_120699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'a', False)
    
    # Call to array(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_120701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    # Adding element type (line 142)
    int_120702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), list_120701, int_120702)
    # Adding element type (line 142)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 142)
    k_120703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 51), 'k', False)
    # Getting the type of 'seq_of_zeros' (line 142)
    seq_of_zeros_120704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'seq_of_zeros', False)
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___120705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 38), seq_of_zeros_120704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_120706 = invoke(stypy.reporting.localization.Localization(__file__, 142, 38), getitem___120705, k_120703)
    
    # Applying the 'usub' unary operator (line 142)
    result___neg___120707 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 37), 'usub', subscript_call_result_120706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), list_120701, result___neg___120707)
    
    # Processing the call keyword arguments (line 142)
    # Getting the type of 'dt' (line 142)
    dt_120708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 62), 'dt', False)
    keyword_120709 = dt_120708
    kwargs_120710 = {'dtype': keyword_120709}
    # Getting the type of 'array' (line 142)
    array_120700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'array', False)
    # Calling array(args, kwargs) (line 142)
    array_call_result_120711 = invoke(stypy.reporting.localization.Localization(__file__, 142, 27), array_120700, *[list_120701], **kwargs_120710)
    
    # Processing the call keyword arguments (line 142)
    str_120712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'str', 'full')
    keyword_120713 = str_120712
    kwargs_120714 = {'mode': keyword_120713}
    # Getting the type of 'NX' (line 142)
    NX_120697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'NX', False)
    # Obtaining the member 'convolve' of a type (line 142)
    convolve_120698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), NX_120697, 'convolve')
    # Calling convolve(args, kwargs) (line 142)
    convolve_call_result_120715 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), convolve_120698, *[a_120699, array_call_result_120711], **kwargs_120714)
    
    # Assigning a type to the variable 'a' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'a', convolve_call_result_120715)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubclass(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'a' (line 145)
    a_120717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'a', False)
    # Obtaining the member 'dtype' of a type (line 145)
    dtype_120718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), a_120717, 'dtype')
    # Obtaining the member 'type' of a type (line 145)
    type_120719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), dtype_120718, 'type')
    # Getting the type of 'NX' (line 145)
    NX_120720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'NX', False)
    # Obtaining the member 'complexfloating' of a type (line 145)
    complexfloating_120721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), NX_120720, 'complexfloating')
    # Processing the call keyword arguments (line 145)
    kwargs_120722 = {}
    # Getting the type of 'issubclass' (line 145)
    issubclass_120716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 145)
    issubclass_call_result_120723 = invoke(stypy.reporting.localization.Localization(__file__, 145, 7), issubclass_120716, *[type_120719, complexfloating_120721], **kwargs_120722)
    
    # Testing the type of an if condition (line 145)
    if_condition_120724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 4), issubclass_call_result_120723)
    # Assigning a type to the variable 'if_condition_120724' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'if_condition_120724', if_condition_120724)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to asarray(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'seq_of_zeros' (line 147)
    seq_of_zeros_120727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'seq_of_zeros', False)
    # Getting the type of 'complex' (line 147)
    complex_120728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'complex', False)
    # Processing the call keyword arguments (line 147)
    kwargs_120729 = {}
    # Getting the type of 'NX' (line 147)
    NX_120725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 147)
    asarray_120726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 16), NX_120725, 'asarray')
    # Calling asarray(args, kwargs) (line 147)
    asarray_call_result_120730 = invoke(stypy.reporting.localization.Localization(__file__, 147, 16), asarray_120726, *[seq_of_zeros_120727, complex_120728], **kwargs_120729)
    
    # Assigning a type to the variable 'roots' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'roots', asarray_call_result_120730)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to sort_complex(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to compress(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Getting the type of 'roots' (line 148)
    roots_120734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 45), 'roots', False)
    # Obtaining the member 'imag' of a type (line 148)
    imag_120735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 45), roots_120734, 'imag')
    int_120736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 58), 'int')
    # Applying the binary operator '>' (line 148)
    result_gt_120737 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 45), '>', imag_120735, int_120736)
    
    # Getting the type of 'roots' (line 148)
    roots_120738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 61), 'roots', False)
    # Processing the call keyword arguments (line 148)
    kwargs_120739 = {}
    # Getting the type of 'NX' (line 148)
    NX_120732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'NX', False)
    # Obtaining the member 'compress' of a type (line 148)
    compress_120733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 33), NX_120732, 'compress')
    # Calling compress(args, kwargs) (line 148)
    compress_call_result_120740 = invoke(stypy.reporting.localization.Localization(__file__, 148, 33), compress_120733, *[result_gt_120737, roots_120738], **kwargs_120739)
    
    # Processing the call keyword arguments (line 148)
    kwargs_120741 = {}
    # Getting the type of 'sort_complex' (line 148)
    sort_complex_120731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'sort_complex', False)
    # Calling sort_complex(args, kwargs) (line 148)
    sort_complex_call_result_120742 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), sort_complex_120731, *[compress_call_result_120740], **kwargs_120741)
    
    # Assigning a type to the variable 'pos_roots' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'pos_roots', sort_complex_call_result_120742)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to conjugate(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Call to sort_complex(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Call to compress(...): (line 150)
    # Processing the call arguments (line 150)
    
    # Getting the type of 'roots' (line 150)
    roots_120748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 52), 'roots', False)
    # Obtaining the member 'imag' of a type (line 150)
    imag_120749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 52), roots_120748, 'imag')
    int_120750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 65), 'int')
    # Applying the binary operator '<' (line 150)
    result_lt_120751 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 52), '<', imag_120749, int_120750)
    
    # Getting the type of 'roots' (line 150)
    roots_120752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 68), 'roots', False)
    # Processing the call keyword arguments (line 150)
    kwargs_120753 = {}
    # Getting the type of 'NX' (line 150)
    NX_120746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 40), 'NX', False)
    # Obtaining the member 'compress' of a type (line 150)
    compress_120747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 40), NX_120746, 'compress')
    # Calling compress(args, kwargs) (line 150)
    compress_call_result_120754 = invoke(stypy.reporting.localization.Localization(__file__, 150, 40), compress_120747, *[result_lt_120751, roots_120752], **kwargs_120753)
    
    # Processing the call keyword arguments (line 149)
    kwargs_120755 = {}
    # Getting the type of 'sort_complex' (line 149)
    sort_complex_120745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'sort_complex', False)
    # Calling sort_complex(args, kwargs) (line 149)
    sort_complex_call_result_120756 = invoke(stypy.reporting.localization.Localization(__file__, 149, 33), sort_complex_120745, *[compress_call_result_120754], **kwargs_120755)
    
    # Processing the call keyword arguments (line 149)
    kwargs_120757 = {}
    # Getting the type of 'NX' (line 149)
    NX_120743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'NX', False)
    # Obtaining the member 'conjugate' of a type (line 149)
    conjugate_120744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), NX_120743, 'conjugate')
    # Calling conjugate(args, kwargs) (line 149)
    conjugate_call_result_120758 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), conjugate_120744, *[sort_complex_call_result_120756], **kwargs_120757)
    
    # Assigning a type to the variable 'neg_roots' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'neg_roots', conjugate_call_result_120758)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'pos_roots' (line 151)
    pos_roots_120760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'pos_roots', False)
    # Processing the call keyword arguments (line 151)
    kwargs_120761 = {}
    # Getting the type of 'len' (line 151)
    len_120759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'len', False)
    # Calling len(args, kwargs) (line 151)
    len_call_result_120762 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), len_120759, *[pos_roots_120760], **kwargs_120761)
    
    
    # Call to len(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'neg_roots' (line 151)
    neg_roots_120764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'neg_roots', False)
    # Processing the call keyword arguments (line 151)
    kwargs_120765 = {}
    # Getting the type of 'len' (line 151)
    len_120763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'len', False)
    # Calling len(args, kwargs) (line 151)
    len_call_result_120766 = invoke(stypy.reporting.localization.Localization(__file__, 151, 30), len_120763, *[neg_roots_120764], **kwargs_120765)
    
    # Applying the binary operator '==' (line 151)
    result_eq_120767 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 12), '==', len_call_result_120762, len_call_result_120766)
    
    
    # Call to alltrue(...): (line 152)
    # Processing the call arguments (line 152)
    
    # Getting the type of 'neg_roots' (line 152)
    neg_roots_120770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'neg_roots', False)
    # Getting the type of 'pos_roots' (line 152)
    pos_roots_120771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 40), 'pos_roots', False)
    # Applying the binary operator '==' (line 152)
    result_eq_120772 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 27), '==', neg_roots_120770, pos_roots_120771)
    
    # Processing the call keyword arguments (line 152)
    kwargs_120773 = {}
    # Getting the type of 'NX' (line 152)
    NX_120768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'NX', False)
    # Obtaining the member 'alltrue' of a type (line 152)
    alltrue_120769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), NX_120768, 'alltrue')
    # Calling alltrue(args, kwargs) (line 152)
    alltrue_call_result_120774 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), alltrue_120769, *[result_eq_120772], **kwargs_120773)
    
    # Applying the binary operator 'and' (line 151)
    result_and_keyword_120775 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 12), 'and', result_eq_120767, alltrue_call_result_120774)
    
    # Testing the type of an if condition (line 151)
    if_condition_120776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_and_keyword_120775)
    # Assigning a type to the variable 'if_condition_120776' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_120776', if_condition_120776)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to copy(...): (line 153)
    # Processing the call keyword arguments (line 153)
    kwargs_120780 = {}
    # Getting the type of 'a' (line 153)
    a_120777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'a', False)
    # Obtaining the member 'real' of a type (line 153)
    real_120778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), a_120777, 'real')
    # Obtaining the member 'copy' of a type (line 153)
    copy_120779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), real_120778, 'copy')
    # Calling copy(args, kwargs) (line 153)
    copy_call_result_120781 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), copy_120779, *[], **kwargs_120780)
    
    # Assigning a type to the variable 'a' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'a', copy_call_result_120781)
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 155)
    a_120782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', a_120782)
    
    # ################# End of 'poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poly' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_120783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120783)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poly'
    return stypy_return_type_120783

# Assigning a type to the variable 'poly' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'poly', poly)

@norecursion
def roots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'roots'
    module_type_store = module_type_store.open_function_context('roots', 157, 0, False)
    
    # Passed parameters checking function
    roots.stypy_localization = localization
    roots.stypy_type_of_self = None
    roots.stypy_type_store = module_type_store
    roots.stypy_function_name = 'roots'
    roots.stypy_param_names_list = ['p']
    roots.stypy_varargs_param_name = None
    roots.stypy_kwargs_param_name = None
    roots.stypy_call_defaults = defaults
    roots.stypy_call_varargs = varargs
    roots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'roots', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'roots', localization, ['p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'roots(...)' code ##################

    str_120784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, (-1)), 'str', '\n    Return the roots of a polynomial with coefficients given in p.\n\n    The values in the rank-1 array `p` are coefficients of a polynomial.\n    If the length of `p` is n+1 then the polynomial is described by::\n\n      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]\n\n    Parameters\n    ----------\n    p : array_like\n        Rank-1 array of polynomial coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        An array containing the complex roots of the polynomial.\n\n    Raises\n    ------\n    ValueError\n        When `p` cannot be converted to a rank-1 array.\n\n    See also\n    --------\n    poly : Find the coefficients of a polynomial with a given sequence\n           of roots.\n    polyval : Compute polynomial values.\n    polyfit : Least squares polynomial fit.\n    poly1d : A one-dimensional polynomial class.\n\n    Notes\n    -----\n    The algorithm relies on computing the eigenvalues of the\n    companion matrix [1]_.\n\n    References\n    ----------\n    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:\n        Cambridge University Press, 1999, pp. 146-7.\n\n    Examples\n    --------\n    >>> coeff = [3.2, 2, 1]\n    >>> np.roots(coeff)\n    array([-0.3125+0.46351241j, -0.3125-0.46351241j])\n\n    ')
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to atleast_1d(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'p' (line 207)
    p_120786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'p', False)
    # Processing the call keyword arguments (line 207)
    kwargs_120787 = {}
    # Getting the type of 'atleast_1d' (line 207)
    atleast_1d_120785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 207)
    atleast_1d_call_result_120788 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), atleast_1d_120785, *[p_120786], **kwargs_120787)
    
    # Assigning a type to the variable 'p' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'p', atleast_1d_call_result_120788)
    
    
    
    # Call to len(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'p' (line 208)
    p_120790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'p', False)
    # Obtaining the member 'shape' of a type (line 208)
    shape_120791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), p_120790, 'shape')
    # Processing the call keyword arguments (line 208)
    kwargs_120792 = {}
    # Getting the type of 'len' (line 208)
    len_120789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 7), 'len', False)
    # Calling len(args, kwargs) (line 208)
    len_call_result_120793 = invoke(stypy.reporting.localization.Localization(__file__, 208, 7), len_120789, *[shape_120791], **kwargs_120792)
    
    int_120794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'int')
    # Applying the binary operator '!=' (line 208)
    result_ne_120795 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 7), '!=', len_call_result_120793, int_120794)
    
    # Testing the type of an if condition (line 208)
    if_condition_120796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 4), result_ne_120795)
    # Assigning a type to the variable 'if_condition_120796' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'if_condition_120796', if_condition_120796)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 209)
    # Processing the call arguments (line 209)
    str_120798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 25), 'str', 'Input must be a rank-1 array.')
    # Processing the call keyword arguments (line 209)
    kwargs_120799 = {}
    # Getting the type of 'ValueError' (line 209)
    ValueError_120797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 209)
    ValueError_call_result_120800 = invoke(stypy.reporting.localization.Localization(__file__, 209, 14), ValueError_120797, *[str_120798], **kwargs_120799)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 8), ValueError_call_result_120800, 'raise parameter', BaseException)
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 212):
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_120801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 39), 'int')
    
    # Call to nonzero(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Call to ravel(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'p' (line 212)
    p_120806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 35), 'p', False)
    # Processing the call keyword arguments (line 212)
    kwargs_120807 = {}
    # Getting the type of 'NX' (line 212)
    NX_120804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'NX', False)
    # Obtaining the member 'ravel' of a type (line 212)
    ravel_120805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 26), NX_120804, 'ravel')
    # Calling ravel(args, kwargs) (line 212)
    ravel_call_result_120808 = invoke(stypy.reporting.localization.Localization(__file__, 212, 26), ravel_120805, *[p_120806], **kwargs_120807)
    
    # Processing the call keyword arguments (line 212)
    kwargs_120809 = {}
    # Getting the type of 'NX' (line 212)
    NX_120802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'NX', False)
    # Obtaining the member 'nonzero' of a type (line 212)
    nonzero_120803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), NX_120802, 'nonzero')
    # Calling nonzero(args, kwargs) (line 212)
    nonzero_call_result_120810 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), nonzero_120803, *[ravel_call_result_120808], **kwargs_120809)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___120811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), nonzero_call_result_120810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_120812 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), getitem___120811, int_120801)
    
    # Assigning a type to the variable 'non_zero' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'non_zero', subscript_call_result_120812)
    
    
    
    # Call to len(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'non_zero' (line 215)
    non_zero_120814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'non_zero', False)
    # Processing the call keyword arguments (line 215)
    kwargs_120815 = {}
    # Getting the type of 'len' (line 215)
    len_120813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'len', False)
    # Calling len(args, kwargs) (line 215)
    len_call_result_120816 = invoke(stypy.reporting.localization.Localization(__file__, 215, 7), len_120813, *[non_zero_120814], **kwargs_120815)
    
    int_120817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 24), 'int')
    # Applying the binary operator '==' (line 215)
    result_eq_120818 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), '==', len_call_result_120816, int_120817)
    
    # Testing the type of an if condition (line 215)
    if_condition_120819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_eq_120818)
    # Assigning a type to the variable 'if_condition_120819' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_120819', if_condition_120819)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_120822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    
    # Processing the call keyword arguments (line 216)
    kwargs_120823 = {}
    # Getting the type of 'NX' (line 216)
    NX_120820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'NX', False)
    # Obtaining the member 'array' of a type (line 216)
    array_120821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), NX_120820, 'array')
    # Calling array(args, kwargs) (line 216)
    array_call_result_120824 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), array_120821, *[list_120822], **kwargs_120823)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', array_call_result_120824)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 219):
    
    # Assigning a BinOp to a Name (line 219):
    
    # Call to len(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'p' (line 219)
    p_120826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'p', False)
    # Processing the call keyword arguments (line 219)
    kwargs_120827 = {}
    # Getting the type of 'len' (line 219)
    len_120825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'len', False)
    # Calling len(args, kwargs) (line 219)
    len_call_result_120828 = invoke(stypy.reporting.localization.Localization(__file__, 219, 21), len_120825, *[p_120826], **kwargs_120827)
    
    
    # Obtaining the type of the subscript
    int_120829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 39), 'int')
    # Getting the type of 'non_zero' (line 219)
    non_zero_120830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'non_zero')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___120831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 30), non_zero_120830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_120832 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), getitem___120831, int_120829)
    
    # Applying the binary operator '-' (line 219)
    result_sub_120833 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 21), '-', len_call_result_120828, subscript_call_result_120832)
    
    int_120834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 45), 'int')
    # Applying the binary operator '-' (line 219)
    result_sub_120835 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 43), '-', result_sub_120833, int_120834)
    
    # Assigning a type to the variable 'trailing_zeros' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'trailing_zeros', result_sub_120835)
    
    # Assigning a Subscript to a Name (line 222):
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 222)
    # Processing the call arguments (line 222)
    
    # Obtaining the type of the subscript
    int_120837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'int')
    # Getting the type of 'non_zero' (line 222)
    non_zero_120838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'non_zero', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___120839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 14), non_zero_120838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_120840 = invoke(stypy.reporting.localization.Localization(__file__, 222, 14), getitem___120839, int_120837)
    
    # Processing the call keyword arguments (line 222)
    kwargs_120841 = {}
    # Getting the type of 'int' (line 222)
    int_120836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 10), 'int', False)
    # Calling int(args, kwargs) (line 222)
    int_call_result_120842 = invoke(stypy.reporting.localization.Localization(__file__, 222, 10), int_120836, *[subscript_call_result_120840], **kwargs_120841)
    
    
    # Call to int(...): (line 222)
    # Processing the call arguments (line 222)
    
    # Obtaining the type of the subscript
    int_120844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 40), 'int')
    # Getting the type of 'non_zero' (line 222)
    non_zero_120845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'non_zero', False)
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___120846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), non_zero_120845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_120847 = invoke(stypy.reporting.localization.Localization(__file__, 222, 31), getitem___120846, int_120844)
    
    # Processing the call keyword arguments (line 222)
    kwargs_120848 = {}
    # Getting the type of 'int' (line 222)
    int_120843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'int', False)
    # Calling int(args, kwargs) (line 222)
    int_call_result_120849 = invoke(stypy.reporting.localization.Localization(__file__, 222, 27), int_120843, *[subscript_call_result_120847], **kwargs_120848)
    
    int_120850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 45), 'int')
    # Applying the binary operator '+' (line 222)
    result_add_120851 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 27), '+', int_call_result_120849, int_120850)
    
    slice_120852 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 222, 8), int_call_result_120842, result_add_120851, None)
    # Getting the type of 'p' (line 222)
    p_120853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'p')
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___120854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), p_120853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_120855 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), getitem___120854, slice_120852)
    
    # Assigning a type to the variable 'p' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'p', subscript_call_result_120855)
    
    
    
    # Call to issubclass(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'p' (line 225)
    p_120857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'p', False)
    # Obtaining the member 'dtype' of a type (line 225)
    dtype_120858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), p_120857, 'dtype')
    # Obtaining the member 'type' of a type (line 225)
    type_120859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), dtype_120858, 'type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 225)
    tuple_120860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 225)
    # Adding element type (line 225)
    # Getting the type of 'NX' (line 225)
    NX_120861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 37), 'NX', False)
    # Obtaining the member 'floating' of a type (line 225)
    floating_120862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 37), NX_120861, 'floating')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 37), tuple_120860, floating_120862)
    # Adding element type (line 225)
    # Getting the type of 'NX' (line 225)
    NX_120863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'NX', False)
    # Obtaining the member 'complexfloating' of a type (line 225)
    complexfloating_120864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 50), NX_120863, 'complexfloating')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 37), tuple_120860, complexfloating_120864)
    
    # Processing the call keyword arguments (line 225)
    kwargs_120865 = {}
    # Getting the type of 'issubclass' (line 225)
    issubclass_120856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 225)
    issubclass_call_result_120866 = invoke(stypy.reporting.localization.Localization(__file__, 225, 11), issubclass_120856, *[type_120859, tuple_120860], **kwargs_120865)
    
    # Applying the 'not' unary operator (line 225)
    result_not__120867 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 7), 'not', issubclass_call_result_120866)
    
    # Testing the type of an if condition (line 225)
    if_condition_120868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 4), result_not__120867)
    # Assigning a type to the variable 'if_condition_120868' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'if_condition_120868', if_condition_120868)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to astype(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'float' (line 226)
    float_120871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 'float', False)
    # Processing the call keyword arguments (line 226)
    kwargs_120872 = {}
    # Getting the type of 'p' (line 226)
    p_120869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'p', False)
    # Obtaining the member 'astype' of a type (line 226)
    astype_120870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), p_120869, 'astype')
    # Calling astype(args, kwargs) (line 226)
    astype_call_result_120873 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), astype_120870, *[float_120871], **kwargs_120872)
    
    # Assigning a type to the variable 'p' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'p', astype_call_result_120873)
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to len(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'p' (line 228)
    p_120875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'p', False)
    # Processing the call keyword arguments (line 228)
    kwargs_120876 = {}
    # Getting the type of 'len' (line 228)
    len_120874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'len', False)
    # Calling len(args, kwargs) (line 228)
    len_call_result_120877 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), len_120874, *[p_120875], **kwargs_120876)
    
    # Assigning a type to the variable 'N' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'N', len_call_result_120877)
    
    
    # Getting the type of 'N' (line 229)
    N_120878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 7), 'N')
    int_120879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 11), 'int')
    # Applying the binary operator '>' (line 229)
    result_gt_120880 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 7), '>', N_120878, int_120879)
    
    # Testing the type of an if condition (line 229)
    if_condition_120881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 4), result_gt_120880)
    # Assigning a type to the variable 'if_condition_120881' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'if_condition_120881', if_condition_120881)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to diag(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Call to ones(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Obtaining an instance of the builtin type 'tuple' (line 231)
    tuple_120885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 231)
    # Adding element type (line 231)
    # Getting the type of 'N' (line 231)
    N_120886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'N', False)
    int_120887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'int')
    # Applying the binary operator '-' (line 231)
    result_sub_120888 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 26), '-', N_120886, int_120887)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 26), tuple_120885, result_sub_120888)
    
    # Getting the type of 'p' (line 231)
    p_120889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 33), 'p', False)
    # Obtaining the member 'dtype' of a type (line 231)
    dtype_120890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 33), p_120889, 'dtype')
    # Processing the call keyword arguments (line 231)
    kwargs_120891 = {}
    # Getting the type of 'NX' (line 231)
    NX_120883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'NX', False)
    # Obtaining the member 'ones' of a type (line 231)
    ones_120884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 17), NX_120883, 'ones')
    # Calling ones(args, kwargs) (line 231)
    ones_call_result_120892 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), ones_120884, *[tuple_120885, dtype_120890], **kwargs_120891)
    
    int_120893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 43), 'int')
    # Processing the call keyword arguments (line 231)
    kwargs_120894 = {}
    # Getting the type of 'diag' (line 231)
    diag_120882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'diag', False)
    # Calling diag(args, kwargs) (line 231)
    diag_call_result_120895 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), diag_120882, *[ones_call_result_120892, int_120893], **kwargs_120894)
    
    # Assigning a type to the variable 'A' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'A', diag_call_result_120895)
    
    # Assigning a BinOp to a Subscript (line 232):
    
    # Assigning a BinOp to a Subscript (line 232):
    
    
    # Obtaining the type of the subscript
    int_120896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 20), 'int')
    slice_120897 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 232, 18), int_120896, None, None)
    # Getting the type of 'p' (line 232)
    p_120898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'p')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___120899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 18), p_120898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_120900 = invoke(stypy.reporting.localization.Localization(__file__, 232, 18), getitem___120899, slice_120897)
    
    # Applying the 'usub' unary operator (line 232)
    result___neg___120901 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 17), 'usub', subscript_call_result_120900)
    
    
    # Obtaining the type of the subscript
    int_120902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'int')
    # Getting the type of 'p' (line 232)
    p_120903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'p')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___120904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 26), p_120903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_120905 = invoke(stypy.reporting.localization.Localization(__file__, 232, 26), getitem___120904, int_120902)
    
    # Applying the binary operator 'div' (line 232)
    result_div_120906 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 17), 'div', result___neg___120901, subscript_call_result_120905)
    
    # Getting the type of 'A' (line 232)
    A_120907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'A')
    int_120908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 10), 'int')
    slice_120909 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 232, 8), None, None, None)
    # Storing an element on a container (line 232)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 8), A_120907, ((int_120908, slice_120909), result_div_120906))
    
    # Assigning a Call to a Name (line 233):
    
    # Assigning a Call to a Name (line 233):
    
    # Call to eigvals(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'A' (line 233)
    A_120911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'A', False)
    # Processing the call keyword arguments (line 233)
    kwargs_120912 = {}
    # Getting the type of 'eigvals' (line 233)
    eigvals_120910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'eigvals', False)
    # Calling eigvals(args, kwargs) (line 233)
    eigvals_call_result_120913 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), eigvals_120910, *[A_120911], **kwargs_120912)
    
    # Assigning a type to the variable 'roots' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'roots', eigvals_call_result_120913)
    # SSA branch for the else part of an if statement (line 229)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to array(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'list' (line 235)
    list_120916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 235)
    
    # Processing the call keyword arguments (line 235)
    kwargs_120917 = {}
    # Getting the type of 'NX' (line 235)
    NX_120914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'NX', False)
    # Obtaining the member 'array' of a type (line 235)
    array_120915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), NX_120914, 'array')
    # Calling array(args, kwargs) (line 235)
    array_call_result_120918 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), array_120915, *[list_120916], **kwargs_120917)
    
    # Assigning a type to the variable 'roots' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'roots', array_call_result_120918)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to hstack(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Obtaining an instance of the builtin type 'tuple' (line 238)
    tuple_120920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 238)
    # Adding element type (line 238)
    # Getting the type of 'roots' (line 238)
    roots_120921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 20), tuple_120920, roots_120921)
    # Adding element type (line 238)
    
    # Call to zeros(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'trailing_zeros' (line 238)
    trailing_zeros_120924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'trailing_zeros', False)
    # Getting the type of 'roots' (line 238)
    roots_120925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 52), 'roots', False)
    # Obtaining the member 'dtype' of a type (line 238)
    dtype_120926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 52), roots_120925, 'dtype')
    # Processing the call keyword arguments (line 238)
    kwargs_120927 = {}
    # Getting the type of 'NX' (line 238)
    NX_120922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'NX', False)
    # Obtaining the member 'zeros' of a type (line 238)
    zeros_120923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), NX_120922, 'zeros')
    # Calling zeros(args, kwargs) (line 238)
    zeros_call_result_120928 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), zeros_120923, *[trailing_zeros_120924, dtype_120926], **kwargs_120927)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 20), tuple_120920, zeros_call_result_120928)
    
    # Processing the call keyword arguments (line 238)
    kwargs_120929 = {}
    # Getting the type of 'hstack' (line 238)
    hstack_120919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'hstack', False)
    # Calling hstack(args, kwargs) (line 238)
    hstack_call_result_120930 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), hstack_120919, *[tuple_120920], **kwargs_120929)
    
    # Assigning a type to the variable 'roots' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'roots', hstack_call_result_120930)
    # Getting the type of 'roots' (line 239)
    roots_120931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'roots')
    # Assigning a type to the variable 'stypy_return_type' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type', roots_120931)
    
    # ################# End of 'roots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'roots' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_120932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120932)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'roots'
    return stypy_return_type_120932

# Assigning a type to the variable 'roots' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'roots', roots)

@norecursion
def polyint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_120933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 17), 'int')
    # Getting the type of 'None' (line 241)
    None_120934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'None')
    defaults = [int_120933, None_120934]
    # Create a new context for function 'polyint'
    module_type_store = module_type_store.open_function_context('polyint', 241, 0, False)
    
    # Passed parameters checking function
    polyint.stypy_localization = localization
    polyint.stypy_type_of_self = None
    polyint.stypy_type_store = module_type_store
    polyint.stypy_function_name = 'polyint'
    polyint.stypy_param_names_list = ['p', 'm', 'k']
    polyint.stypy_varargs_param_name = None
    polyint.stypy_kwargs_param_name = None
    polyint.stypy_call_defaults = defaults
    polyint.stypy_call_varargs = varargs
    polyint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyint', ['p', 'm', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyint', localization, ['p', 'm', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyint(...)' code ##################

    str_120935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, (-1)), 'str', '\n    Return an antiderivative (indefinite integral) of a polynomial.\n\n    The returned order `m` antiderivative `P` of polynomial `p` satisfies\n    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`\n    integration constants `k`. The constants determine the low-order\n    polynomial part\n\n    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}\n\n    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.\n\n    Parameters\n    ----------\n    p : array_like or poly1d\n        Polynomial to differentiate.\n        A sequence is interpreted as polynomial coefficients, see `poly1d`.\n    m : int, optional\n        Order of the antiderivative. (Default: 1)\n    k : list of `m` scalars or scalar, optional\n        Integration constants. They are given in the order of integration:\n        those corresponding to highest-order terms come first.\n\n        If ``None`` (default), all constants are assumed to be zero.\n        If `m = 1`, a single scalar can be given instead of a list.\n\n    See Also\n    --------\n    polyder : derivative of a polynomial\n    poly1d.integ : equivalent method\n\n    Examples\n    --------\n    The defining property of the antiderivative:\n\n    >>> p = np.poly1d([1,1,1])\n    >>> P = np.polyint(p)\n    >>> P\n    poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ])\n    >>> np.polyder(P) == p\n    True\n\n    The integration constants default to zero, but can be specified:\n\n    >>> P = np.polyint(p, 3)\n    >>> P(0)\n    0.0\n    >>> np.polyder(P)(0)\n    0.0\n    >>> np.polyder(P, 2)(0)\n    0.0\n    >>> P = np.polyint(p, 3, k=[6,5,3])\n    >>> P\n    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3. ,  5. ,  3. ])\n\n    Note that 3 = 6 / 2!, and that the constants are given in the order of\n    integrations. Constant of the highest-order polynomial term comes first:\n\n    >>> np.polyder(P, 2)(0)\n    6.0\n    >>> np.polyder(P, 1)(0)\n    5.0\n    >>> P(0)\n    3.0\n\n    ')
    
    # Assigning a Call to a Name (line 308):
    
    # Assigning a Call to a Name (line 308):
    
    # Call to int(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'm' (line 308)
    m_120937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'm', False)
    # Processing the call keyword arguments (line 308)
    kwargs_120938 = {}
    # Getting the type of 'int' (line 308)
    int_120936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'int', False)
    # Calling int(args, kwargs) (line 308)
    int_call_result_120939 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), int_120936, *[m_120937], **kwargs_120938)
    
    # Assigning a type to the variable 'm' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'm', int_call_result_120939)
    
    
    # Getting the type of 'm' (line 309)
    m_120940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 7), 'm')
    int_120941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 11), 'int')
    # Applying the binary operator '<' (line 309)
    result_lt_120942 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 7), '<', m_120940, int_120941)
    
    # Testing the type of an if condition (line 309)
    if_condition_120943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 4), result_lt_120942)
    # Assigning a type to the variable 'if_condition_120943' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'if_condition_120943', if_condition_120943)
    # SSA begins for if statement (line 309)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 310)
    # Processing the call arguments (line 310)
    str_120945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 25), 'str', 'Order of integral must be positive (see polyder)')
    # Processing the call keyword arguments (line 310)
    kwargs_120946 = {}
    # Getting the type of 'ValueError' (line 310)
    ValueError_120944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 310)
    ValueError_call_result_120947 = invoke(stypy.reporting.localization.Localization(__file__, 310, 14), ValueError_120944, *[str_120945], **kwargs_120946)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 310, 8), ValueError_call_result_120947, 'raise parameter', BaseException)
    # SSA join for if statement (line 309)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 311)
    # Getting the type of 'k' (line 311)
    k_120948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 7), 'k')
    # Getting the type of 'None' (line 311)
    None_120949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'None')
    
    (may_be_120950, more_types_in_union_120951) = may_be_none(k_120948, None_120949)

    if may_be_120950:

        if more_types_in_union_120951:
            # Runtime conditional SSA (line 311)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to zeros(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'm' (line 312)
        m_120954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 21), 'm', False)
        # Getting the type of 'float' (line 312)
        float_120955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'float', False)
        # Processing the call keyword arguments (line 312)
        kwargs_120956 = {}
        # Getting the type of 'NX' (line 312)
        NX_120952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'NX', False)
        # Obtaining the member 'zeros' of a type (line 312)
        zeros_120953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), NX_120952, 'zeros')
        # Calling zeros(args, kwargs) (line 312)
        zeros_call_result_120957 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), zeros_120953, *[m_120954, float_120955], **kwargs_120956)
        
        # Assigning a type to the variable 'k' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'k', zeros_call_result_120957)

        if more_types_in_union_120951:
            # SSA join for if statement (line 311)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to atleast_1d(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'k' (line 313)
    k_120959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'k', False)
    # Processing the call keyword arguments (line 313)
    kwargs_120960 = {}
    # Getting the type of 'atleast_1d' (line 313)
    atleast_1d_120958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 313)
    atleast_1d_call_result_120961 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), atleast_1d_120958, *[k_120959], **kwargs_120960)
    
    # Assigning a type to the variable 'k' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'k', atleast_1d_call_result_120961)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'k' (line 314)
    k_120963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'k', False)
    # Processing the call keyword arguments (line 314)
    kwargs_120964 = {}
    # Getting the type of 'len' (line 314)
    len_120962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 7), 'len', False)
    # Calling len(args, kwargs) (line 314)
    len_call_result_120965 = invoke(stypy.reporting.localization.Localization(__file__, 314, 7), len_120962, *[k_120963], **kwargs_120964)
    
    int_120966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 17), 'int')
    # Applying the binary operator '==' (line 314)
    result_eq_120967 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 7), '==', len_call_result_120965, int_120966)
    
    
    # Getting the type of 'm' (line 314)
    m_120968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 'm')
    int_120969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 27), 'int')
    # Applying the binary operator '>' (line 314)
    result_gt_120970 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 23), '>', m_120968, int_120969)
    
    # Applying the binary operator 'and' (line 314)
    result_and_keyword_120971 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 7), 'and', result_eq_120967, result_gt_120970)
    
    # Testing the type of an if condition (line 314)
    if_condition_120972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 4), result_and_keyword_120971)
    # Assigning a type to the variable 'if_condition_120972' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'if_condition_120972', if_condition_120972)
    # SSA begins for if statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 315):
    
    # Assigning a BinOp to a Name (line 315):
    
    # Obtaining the type of the subscript
    int_120973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 14), 'int')
    # Getting the type of 'k' (line 315)
    k_120974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'k')
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___120975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), k_120974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_120976 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), getitem___120975, int_120973)
    
    
    # Call to ones(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'm' (line 315)
    m_120979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 25), 'm', False)
    # Getting the type of 'float' (line 315)
    float_120980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 28), 'float', False)
    # Processing the call keyword arguments (line 315)
    kwargs_120981 = {}
    # Getting the type of 'NX' (line 315)
    NX_120977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'NX', False)
    # Obtaining the member 'ones' of a type (line 315)
    ones_120978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 17), NX_120977, 'ones')
    # Calling ones(args, kwargs) (line 315)
    ones_call_result_120982 = invoke(stypy.reporting.localization.Localization(__file__, 315, 17), ones_120978, *[m_120979, float_120980], **kwargs_120981)
    
    # Applying the binary operator '*' (line 315)
    result_mul_120983 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 12), '*', subscript_call_result_120976, ones_call_result_120982)
    
    # Assigning a type to the variable 'k' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'k', result_mul_120983)
    # SSA join for if statement (line 314)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'k' (line 316)
    k_120985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 'k', False)
    # Processing the call keyword arguments (line 316)
    kwargs_120986 = {}
    # Getting the type of 'len' (line 316)
    len_120984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 7), 'len', False)
    # Calling len(args, kwargs) (line 316)
    len_call_result_120987 = invoke(stypy.reporting.localization.Localization(__file__, 316, 7), len_120984, *[k_120985], **kwargs_120986)
    
    # Getting the type of 'm' (line 316)
    m_120988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'm')
    # Applying the binary operator '<' (line 316)
    result_lt_120989 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 7), '<', len_call_result_120987, m_120988)
    
    # Testing the type of an if condition (line 316)
    if_condition_120990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 4), result_lt_120989)
    # Assigning a type to the variable 'if_condition_120990' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'if_condition_120990', if_condition_120990)
    # SSA begins for if statement (line 316)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 317)
    # Processing the call arguments (line 317)
    str_120992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 14), 'str', 'k must be a scalar or a rank-1 array of length 1 or >m.')
    # Processing the call keyword arguments (line 317)
    kwargs_120993 = {}
    # Getting the type of 'ValueError' (line 317)
    ValueError_120991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 317)
    ValueError_call_result_120994 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), ValueError_120991, *[str_120992], **kwargs_120993)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 317, 8), ValueError_call_result_120994, 'raise parameter', BaseException)
    # SSA join for if statement (line 316)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to isinstance(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'p' (line 320)
    p_120996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'p', False)
    # Getting the type of 'poly1d' (line 320)
    poly1d_120997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 29), 'poly1d', False)
    # Processing the call keyword arguments (line 320)
    kwargs_120998 = {}
    # Getting the type of 'isinstance' (line 320)
    isinstance_120995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 320)
    isinstance_call_result_120999 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), isinstance_120995, *[p_120996, poly1d_120997], **kwargs_120998)
    
    # Assigning a type to the variable 'truepoly' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'truepoly', isinstance_call_result_120999)
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to asarray(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'p' (line 321)
    p_121002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'p', False)
    # Processing the call keyword arguments (line 321)
    kwargs_121003 = {}
    # Getting the type of 'NX' (line 321)
    NX_121000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 321)
    asarray_121001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), NX_121000, 'asarray')
    # Calling asarray(args, kwargs) (line 321)
    asarray_call_result_121004 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), asarray_121001, *[p_121002], **kwargs_121003)
    
    # Assigning a type to the variable 'p' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'p', asarray_call_result_121004)
    
    
    # Getting the type of 'm' (line 322)
    m_121005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 7), 'm')
    int_121006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 12), 'int')
    # Applying the binary operator '==' (line 322)
    result_eq_121007 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), '==', m_121005, int_121006)
    
    # Testing the type of an if condition (line 322)
    if_condition_121008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_eq_121007)
    # Assigning a type to the variable 'if_condition_121008' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_121008', if_condition_121008)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'truepoly' (line 323)
    truepoly_121009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'truepoly')
    # Testing the type of an if condition (line 323)
    if_condition_121010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), truepoly_121009)
    # Assigning a type to the variable 'if_condition_121010' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_121010', if_condition_121010)
    # SSA begins for if statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to poly1d(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'p' (line 324)
    p_121012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'p', False)
    # Processing the call keyword arguments (line 324)
    kwargs_121013 = {}
    # Getting the type of 'poly1d' (line 324)
    poly1d_121011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 324)
    poly1d_call_result_121014 = invoke(stypy.reporting.localization.Localization(__file__, 324, 19), poly1d_121011, *[p_121012], **kwargs_121013)
    
    # Assigning a type to the variable 'stypy_return_type' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'stypy_return_type', poly1d_call_result_121014)
    # SSA join for if statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'p' (line 325)
    p_121015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'stypy_return_type', p_121015)
    # SSA branch for the else part of an if statement (line 322)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to concatenate(...): (line 328)
    # Processing the call arguments (line 328)
    
    # Obtaining an instance of the builtin type 'tuple' (line 328)
    tuple_121018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 328)
    # Adding element type (line 328)
    
    # Call to __truediv__(...): (line 328)
    # Processing the call arguments (line 328)
    
    # Call to arange(...): (line 328)
    # Processing the call arguments (line 328)
    
    # Call to len(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'p' (line 328)
    p_121024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 56), 'p', False)
    # Processing the call keyword arguments (line 328)
    kwargs_121025 = {}
    # Getting the type of 'len' (line 328)
    len_121023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 52), 'len', False)
    # Calling len(args, kwargs) (line 328)
    len_call_result_121026 = invoke(stypy.reporting.localization.Localization(__file__, 328, 52), len_121023, *[p_121024], **kwargs_121025)
    
    int_121027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 60), 'int')
    int_121028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 63), 'int')
    # Processing the call keyword arguments (line 328)
    kwargs_121029 = {}
    # Getting the type of 'NX' (line 328)
    NX_121021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 42), 'NX', False)
    # Obtaining the member 'arange' of a type (line 328)
    arange_121022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 42), NX_121021, 'arange')
    # Calling arange(args, kwargs) (line 328)
    arange_call_result_121030 = invoke(stypy.reporting.localization.Localization(__file__, 328, 42), arange_121022, *[len_call_result_121026, int_121027, int_121028], **kwargs_121029)
    
    # Processing the call keyword arguments (line 328)
    kwargs_121031 = {}
    # Getting the type of 'p' (line 328)
    p_121019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'p', False)
    # Obtaining the member '__truediv__' of a type (line 328)
    truediv___121020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 28), p_121019, '__truediv__')
    # Calling __truediv__(args, kwargs) (line 328)
    truediv___call_result_121032 = invoke(stypy.reporting.localization.Localization(__file__, 328, 28), truediv___121020, *[arange_call_result_121030], **kwargs_121031)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 28), tuple_121018, truediv___call_result_121032)
    # Adding element type (line 328)
    
    # Obtaining an instance of the builtin type 'list' (line 328)
    list_121033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 69), 'list')
    # Adding type elements to the builtin type 'list' instance (line 328)
    # Adding element type (line 328)
    
    # Obtaining the type of the subscript
    int_121034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 72), 'int')
    # Getting the type of 'k' (line 328)
    k_121035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 70), 'k', False)
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___121036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 70), k_121035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_121037 = invoke(stypy.reporting.localization.Localization(__file__, 328, 70), getitem___121036, int_121034)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 69), list_121033, subscript_call_result_121037)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 28), tuple_121018, list_121033)
    
    # Processing the call keyword arguments (line 328)
    kwargs_121038 = {}
    # Getting the type of 'NX' (line 328)
    NX_121016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'NX', False)
    # Obtaining the member 'concatenate' of a type (line 328)
    concatenate_121017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), NX_121016, 'concatenate')
    # Calling concatenate(args, kwargs) (line 328)
    concatenate_call_result_121039 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), concatenate_121017, *[tuple_121018], **kwargs_121038)
    
    # Assigning a type to the variable 'y' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'y', concatenate_call_result_121039)
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to polyint(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'y' (line 329)
    y_121041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 22), 'y', False)
    # Getting the type of 'm' (line 329)
    m_121042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 25), 'm', False)
    int_121043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'int')
    # Applying the binary operator '-' (line 329)
    result_sub_121044 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 25), '-', m_121042, int_121043)
    
    # Processing the call keyword arguments (line 329)
    
    # Obtaining the type of the subscript
    int_121045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 36), 'int')
    slice_121046 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 329, 34), int_121045, None, None)
    # Getting the type of 'k' (line 329)
    k_121047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'k', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___121048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 34), k_121047, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_121049 = invoke(stypy.reporting.localization.Localization(__file__, 329, 34), getitem___121048, slice_121046)
    
    keyword_121050 = subscript_call_result_121049
    kwargs_121051 = {'k': keyword_121050}
    # Getting the type of 'polyint' (line 329)
    polyint_121040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'polyint', False)
    # Calling polyint(args, kwargs) (line 329)
    polyint_call_result_121052 = invoke(stypy.reporting.localization.Localization(__file__, 329, 14), polyint_121040, *[y_121041, result_sub_121044], **kwargs_121051)
    
    # Assigning a type to the variable 'val' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'val', polyint_call_result_121052)
    
    # Getting the type of 'truepoly' (line 330)
    truepoly_121053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'truepoly')
    # Testing the type of an if condition (line 330)
    if_condition_121054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), truepoly_121053)
    # Assigning a type to the variable 'if_condition_121054' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_121054', if_condition_121054)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to poly1d(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'val' (line 331)
    val_121056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'val', False)
    # Processing the call keyword arguments (line 331)
    kwargs_121057 = {}
    # Getting the type of 'poly1d' (line 331)
    poly1d_121055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 331)
    poly1d_call_result_121058 = invoke(stypy.reporting.localization.Localization(__file__, 331, 19), poly1d_121055, *[val_121056], **kwargs_121057)
    
    # Assigning a type to the variable 'stypy_return_type' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'stypy_return_type', poly1d_call_result_121058)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 332)
    val_121059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', val_121059)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polyint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyint' in the type store
    # Getting the type of 'stypy_return_type' (line 241)
    stypy_return_type_121060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121060)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyint'
    return stypy_return_type_121060

# Assigning a type to the variable 'polyint' (line 241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'polyint', polyint)

@norecursion
def polyder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_121061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 17), 'int')
    defaults = [int_121061]
    # Create a new context for function 'polyder'
    module_type_store = module_type_store.open_function_context('polyder', 334, 0, False)
    
    # Passed parameters checking function
    polyder.stypy_localization = localization
    polyder.stypy_type_of_self = None
    polyder.stypy_type_store = module_type_store
    polyder.stypy_function_name = 'polyder'
    polyder.stypy_param_names_list = ['p', 'm']
    polyder.stypy_varargs_param_name = None
    polyder.stypy_kwargs_param_name = None
    polyder.stypy_call_defaults = defaults
    polyder.stypy_call_varargs = varargs
    polyder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyder', ['p', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyder', localization, ['p', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyder(...)' code ##################

    str_121062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, (-1)), 'str', '\n    Return the derivative of the specified order of a polynomial.\n\n    Parameters\n    ----------\n    p : poly1d or sequence\n        Polynomial to differentiate.\n        A sequence is interpreted as polynomial coefficients, see `poly1d`.\n    m : int, optional\n        Order of differentiation (default: 1)\n\n    Returns\n    -------\n    der : poly1d\n        A new polynomial representing the derivative.\n\n    See Also\n    --------\n    polyint : Anti-derivative of a polynomial.\n    poly1d : Class for one-dimensional polynomials.\n\n    Examples\n    --------\n    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:\n\n    >>> p = np.poly1d([1,1,1,1])\n    >>> p2 = np.polyder(p)\n    >>> p2\n    poly1d([3, 2, 1])\n\n    which evaluates to:\n\n    >>> p2(2.)\n    17.0\n\n    We can verify this, approximating the derivative with\n    ``(f(x + h) - f(x))/h``:\n\n    >>> (p(2. + 0.001) - p(2.)) / 0.001\n    17.007000999997857\n\n    The fourth-order derivative of a 3rd-order polynomial is zero:\n\n    >>> np.polyder(p, 2)\n    poly1d([6, 2])\n    >>> np.polyder(p, 3)\n    poly1d([6])\n    >>> np.polyder(p, 4)\n    poly1d([ 0.])\n\n    ')
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to int(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'm' (line 386)
    m_121064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'm', False)
    # Processing the call keyword arguments (line 386)
    kwargs_121065 = {}
    # Getting the type of 'int' (line 386)
    int_121063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'int', False)
    # Calling int(args, kwargs) (line 386)
    int_call_result_121066 = invoke(stypy.reporting.localization.Localization(__file__, 386, 8), int_121063, *[m_121064], **kwargs_121065)
    
    # Assigning a type to the variable 'm' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'm', int_call_result_121066)
    
    
    # Getting the type of 'm' (line 387)
    m_121067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 7), 'm')
    int_121068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 11), 'int')
    # Applying the binary operator '<' (line 387)
    result_lt_121069 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 7), '<', m_121067, int_121068)
    
    # Testing the type of an if condition (line 387)
    if_condition_121070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 4), result_lt_121069)
    # Assigning a type to the variable 'if_condition_121070' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'if_condition_121070', if_condition_121070)
    # SSA begins for if statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 388)
    # Processing the call arguments (line 388)
    str_121072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 25), 'str', 'Order of derivative must be positive (see polyint)')
    # Processing the call keyword arguments (line 388)
    kwargs_121073 = {}
    # Getting the type of 'ValueError' (line 388)
    ValueError_121071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 388)
    ValueError_call_result_121074 = invoke(stypy.reporting.localization.Localization(__file__, 388, 14), ValueError_121071, *[str_121072], **kwargs_121073)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 388, 8), ValueError_call_result_121074, 'raise parameter', BaseException)
    # SSA join for if statement (line 387)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to isinstance(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'p' (line 390)
    p_121076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'p', False)
    # Getting the type of 'poly1d' (line 390)
    poly1d_121077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'poly1d', False)
    # Processing the call keyword arguments (line 390)
    kwargs_121078 = {}
    # Getting the type of 'isinstance' (line 390)
    isinstance_121075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 390)
    isinstance_call_result_121079 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), isinstance_121075, *[p_121076, poly1d_121077], **kwargs_121078)
    
    # Assigning a type to the variable 'truepoly' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'truepoly', isinstance_call_result_121079)
    
    # Assigning a Call to a Name (line 391):
    
    # Assigning a Call to a Name (line 391):
    
    # Call to asarray(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'p' (line 391)
    p_121082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'p', False)
    # Processing the call keyword arguments (line 391)
    kwargs_121083 = {}
    # Getting the type of 'NX' (line 391)
    NX_121080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 391)
    asarray_121081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), NX_121080, 'asarray')
    # Calling asarray(args, kwargs) (line 391)
    asarray_call_result_121084 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), asarray_121081, *[p_121082], **kwargs_121083)
    
    # Assigning a type to the variable 'p' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'p', asarray_call_result_121084)
    
    # Assigning a BinOp to a Name (line 392):
    
    # Assigning a BinOp to a Name (line 392):
    
    # Call to len(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'p' (line 392)
    p_121086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'p', False)
    # Processing the call keyword arguments (line 392)
    kwargs_121087 = {}
    # Getting the type of 'len' (line 392)
    len_121085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'len', False)
    # Calling len(args, kwargs) (line 392)
    len_call_result_121088 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), len_121085, *[p_121086], **kwargs_121087)
    
    int_121089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 17), 'int')
    # Applying the binary operator '-' (line 392)
    result_sub_121090 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 8), '-', len_call_result_121088, int_121089)
    
    # Assigning a type to the variable 'n' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'n', result_sub_121090)
    
    # Assigning a BinOp to a Name (line 393):
    
    # Assigning a BinOp to a Name (line 393):
    
    # Obtaining the type of the subscript
    int_121091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 11), 'int')
    slice_121092 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 8), None, int_121091, None)
    # Getting the type of 'p' (line 393)
    p_121093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'p')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___121094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), p_121093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_121095 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___121094, slice_121092)
    
    
    # Call to arange(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'n' (line 393)
    n_121098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 27), 'n', False)
    int_121099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 30), 'int')
    int_121100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 33), 'int')
    # Processing the call keyword arguments (line 393)
    kwargs_121101 = {}
    # Getting the type of 'NX' (line 393)
    NX_121096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'NX', False)
    # Obtaining the member 'arange' of a type (line 393)
    arange_121097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 17), NX_121096, 'arange')
    # Calling arange(args, kwargs) (line 393)
    arange_call_result_121102 = invoke(stypy.reporting.localization.Localization(__file__, 393, 17), arange_121097, *[n_121098, int_121099, int_121100], **kwargs_121101)
    
    # Applying the binary operator '*' (line 393)
    result_mul_121103 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 8), '*', subscript_call_result_121095, arange_call_result_121102)
    
    # Assigning a type to the variable 'y' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'y', result_mul_121103)
    
    
    # Getting the type of 'm' (line 394)
    m_121104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 7), 'm')
    int_121105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 12), 'int')
    # Applying the binary operator '==' (line 394)
    result_eq_121106 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 7), '==', m_121104, int_121105)
    
    # Testing the type of an if condition (line 394)
    if_condition_121107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 4), result_eq_121106)
    # Assigning a type to the variable 'if_condition_121107' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'if_condition_121107', if_condition_121107)
    # SSA begins for if statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 395):
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'p' (line 395)
    p_121108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'p')
    # Assigning a type to the variable 'val' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'val', p_121108)
    # SSA branch for the else part of an if statement (line 394)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 397):
    
    # Assigning a Call to a Name (line 397):
    
    # Call to polyder(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'y' (line 397)
    y_121110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'y', False)
    # Getting the type of 'm' (line 397)
    m_121111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 25), 'm', False)
    int_121112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 29), 'int')
    # Applying the binary operator '-' (line 397)
    result_sub_121113 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 25), '-', m_121111, int_121112)
    
    # Processing the call keyword arguments (line 397)
    kwargs_121114 = {}
    # Getting the type of 'polyder' (line 397)
    polyder_121109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), 'polyder', False)
    # Calling polyder(args, kwargs) (line 397)
    polyder_call_result_121115 = invoke(stypy.reporting.localization.Localization(__file__, 397, 14), polyder_121109, *[y_121110, result_sub_121113], **kwargs_121114)
    
    # Assigning a type to the variable 'val' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'val', polyder_call_result_121115)
    # SSA join for if statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'truepoly' (line 398)
    truepoly_121116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 7), 'truepoly')
    # Testing the type of an if condition (line 398)
    if_condition_121117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 4), truepoly_121116)
    # Assigning a type to the variable 'if_condition_121117' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'if_condition_121117', if_condition_121117)
    # SSA begins for if statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Call to a Name (line 399):
    
    # Call to poly1d(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'val' (line 399)
    val_121119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'val', False)
    # Processing the call keyword arguments (line 399)
    kwargs_121120 = {}
    # Getting the type of 'poly1d' (line 399)
    poly1d_121118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 14), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 399)
    poly1d_call_result_121121 = invoke(stypy.reporting.localization.Localization(__file__, 399, 14), poly1d_121118, *[val_121119], **kwargs_121120)
    
    # Assigning a type to the variable 'val' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'val', poly1d_call_result_121121)
    # SSA join for if statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 400)
    val_121122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type', val_121122)
    
    # ################# End of 'polyder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyder' in the type store
    # Getting the type of 'stypy_return_type' (line 334)
    stypy_return_type_121123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyder'
    return stypy_return_type_121123

# Assigning a type to the variable 'polyder' (line 334)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'polyder', polyder)

@norecursion
def polyfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 402)
    None_121124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 29), 'None')
    # Getting the type of 'False' (line 402)
    False_121125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 40), 'False')
    # Getting the type of 'None' (line 402)
    None_121126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 49), 'None')
    # Getting the type of 'False' (line 402)
    False_121127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 59), 'False')
    defaults = [None_121124, False_121125, None_121126, False_121127]
    # Create a new context for function 'polyfit'
    module_type_store = module_type_store.open_function_context('polyfit', 402, 0, False)
    
    # Passed parameters checking function
    polyfit.stypy_localization = localization
    polyfit.stypy_type_of_self = None
    polyfit.stypy_type_store = module_type_store
    polyfit.stypy_function_name = 'polyfit'
    polyfit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w', 'cov']
    polyfit.stypy_varargs_param_name = None
    polyfit.stypy_kwargs_param_name = None
    polyfit.stypy_call_defaults = defaults
    polyfit.stypy_call_varargs = varargs
    polyfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyfit', ['x', 'y', 'deg', 'rcond', 'full', 'w', 'cov'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyfit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w', 'cov'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyfit(...)' code ##################

    str_121128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, (-1)), 'str', '\n    Least squares polynomial fit.\n\n    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`\n    to points `(x, y)`. Returns a vector of coefficients `p` that minimises\n    the squared error.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int\n        Degree of the fitting polynomial\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (M,), optional\n        Weights to apply to the y-coordinates of the sample points. For\n        gaussian uncertainties, use 1/sigma (not 1/sigma**2).\n    cov : bool, optional\n        Return the estimate and the covariance matrix of the estimate\n        If full is True, then cov is not returned.\n\n    Returns\n    -------\n    p : ndarray, shape (M,) or (M, K)\n        Polynomial coefficients, highest power first.  If `y` was 2-D, the\n        coefficients for `k`-th data set are in ``p[:,k]``.\n\n    residuals, rank, singular_values, rcond :\n        Present only if `full` = True.  Residuals of the least-squares fit,\n        the effective rank of the scaled Vandermonde coefficient matrix,\n        its singular values, and the specified value of `rcond`. For more\n        details, see `linalg.lstsq`.\n\n    V : ndarray, shape (M,M) or (M,M,K)\n        Present only if `full` = False and `cov`=True.  The covariance\n        matrix of the polynomial coefficient estimates.  The diagonal of\n        this matrix are the variance estimates for each coefficient.  If y\n        is a 2-D array, then the covariance matrix for the `k`-th data set\n        are in ``V[:,:,k]``\n\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if `full` = False.\n\n        The warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', np.RankWarning)\n\n    See Also\n    --------\n    polyval : Compute polynomial values.\n    linalg.lstsq : Computes a least-squares fit.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution minimizes the squared error\n\n    .. math ::\n        E = \\sum_{j=0}^k |p(x_j) - y_j|^2\n\n    in the equations::\n\n        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]\n        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]\n        ...\n        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]\n\n    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.\n\n    `polyfit` issues a `RankWarning` when the least-squares fit is badly\n    conditioned. This implies that the best fit is not well-defined due\n    to numerical error. The results may be improved by lowering the polynomial\n    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter\n    can also be set to a value smaller than its default, but the resulting\n    fit may be spurious: including contributions from the small singular\n    values can add numerical noise to the result.\n\n    Note that fitting polynomial coefficients is inherently badly conditioned\n    when the degree of the polynomial is large or the interval of sample points\n    is badly centered. The quality of the fit should always be checked in these\n    cases. When polynomial fits are not satisfactory, splines may be a good\n    alternative.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           http://en.wikipedia.org/wiki/Curve_fitting\n    .. [2] Wikipedia, "Polynomial interpolation",\n           http://en.wikipedia.org/wiki/Polynomial_interpolation\n\n    Examples\n    --------\n    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])\n    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])\n    >>> z = np.polyfit(x, y, 3)\n    >>> z\n    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])\n\n    It is convenient to use `poly1d` objects for dealing with polynomials:\n\n    >>> p = np.poly1d(z)\n    >>> p(0.5)\n    0.6143849206349179\n    >>> p(3.5)\n    -0.34732142857143039\n    >>> p(10)\n    22.579365079365115\n\n    High-order polynomials may oscillate wildly:\n\n    >>> p30 = np.poly1d(np.polyfit(x, y, 30))\n    /... RankWarning: Polyfit may be poorly conditioned...\n    >>> p30(4)\n    -0.80000000000000204\n    >>> p30(5)\n    -0.99999999999999445\n    >>> p30(4.5)\n    -0.10547061179440398\n\n    Illustration:\n\n    >>> import matplotlib.pyplot as plt\n    >>> xp = np.linspace(-2, 6, 100)\n    >>> _ = plt.plot(x, y, \'.\', xp, p(xp), \'-\', xp, p30(xp), \'--\')\n    >>> plt.ylim(-2,2)\n    (-2, 2)\n    >>> plt.show()\n\n    ')
    
    # Assigning a BinOp to a Name (line 549):
    
    # Assigning a BinOp to a Name (line 549):
    
    # Call to int(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'deg' (line 549)
    deg_121130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'deg', False)
    # Processing the call keyword arguments (line 549)
    kwargs_121131 = {}
    # Getting the type of 'int' (line 549)
    int_121129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'int', False)
    # Calling int(args, kwargs) (line 549)
    int_call_result_121132 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), int_121129, *[deg_121130], **kwargs_121131)
    
    int_121133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 23), 'int')
    # Applying the binary operator '+' (line 549)
    result_add_121134 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 12), '+', int_call_result_121132, int_121133)
    
    # Assigning a type to the variable 'order' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'order', result_add_121134)
    
    # Assigning a BinOp to a Name (line 550):
    
    # Assigning a BinOp to a Name (line 550):
    
    # Call to asarray(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'x' (line 550)
    x_121137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'x', False)
    # Processing the call keyword arguments (line 550)
    kwargs_121138 = {}
    # Getting the type of 'NX' (line 550)
    NX_121135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 550)
    asarray_121136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), NX_121135, 'asarray')
    # Calling asarray(args, kwargs) (line 550)
    asarray_call_result_121139 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), asarray_121136, *[x_121137], **kwargs_121138)
    
    float_121140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 24), 'float')
    # Applying the binary operator '+' (line 550)
    result_add_121141 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 8), '+', asarray_call_result_121139, float_121140)
    
    # Assigning a type to the variable 'x' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'x', result_add_121141)
    
    # Assigning a BinOp to a Name (line 551):
    
    # Assigning a BinOp to a Name (line 551):
    
    # Call to asarray(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'y' (line 551)
    y_121144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 19), 'y', False)
    # Processing the call keyword arguments (line 551)
    kwargs_121145 = {}
    # Getting the type of 'NX' (line 551)
    NX_121142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 551)
    asarray_121143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 8), NX_121142, 'asarray')
    # Calling asarray(args, kwargs) (line 551)
    asarray_call_result_121146 = invoke(stypy.reporting.localization.Localization(__file__, 551, 8), asarray_121143, *[y_121144], **kwargs_121145)
    
    float_121147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 24), 'float')
    # Applying the binary operator '+' (line 551)
    result_add_121148 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 8), '+', asarray_call_result_121146, float_121147)
    
    # Assigning a type to the variable 'y' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'y', result_add_121148)
    
    
    # Getting the type of 'deg' (line 554)
    deg_121149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 7), 'deg')
    int_121150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 13), 'int')
    # Applying the binary operator '<' (line 554)
    result_lt_121151 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 7), '<', deg_121149, int_121150)
    
    # Testing the type of an if condition (line 554)
    if_condition_121152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 4), result_lt_121151)
    # Assigning a type to the variable 'if_condition_121152' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'if_condition_121152', if_condition_121152)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 555)
    # Processing the call arguments (line 555)
    str_121154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 555)
    kwargs_121155 = {}
    # Getting the type of 'ValueError' (line 555)
    ValueError_121153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 555)
    ValueError_call_result_121156 = invoke(stypy.reporting.localization.Localization(__file__, 555, 14), ValueError_121153, *[str_121154], **kwargs_121155)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 555, 8), ValueError_call_result_121156, 'raise parameter', BaseException)
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 556)
    x_121157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 556)
    ndim_121158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 7), x_121157, 'ndim')
    int_121159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 17), 'int')
    # Applying the binary operator '!=' (line 556)
    result_ne_121160 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 7), '!=', ndim_121158, int_121159)
    
    # Testing the type of an if condition (line 556)
    if_condition_121161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 4), result_ne_121160)
    # Assigning a type to the variable 'if_condition_121161' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'if_condition_121161', if_condition_121161)
    # SSA begins for if statement (line 556)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 557)
    # Processing the call arguments (line 557)
    str_121163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 557)
    kwargs_121164 = {}
    # Getting the type of 'TypeError' (line 557)
    TypeError_121162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 557)
    TypeError_call_result_121165 = invoke(stypy.reporting.localization.Localization(__file__, 557, 14), TypeError_121162, *[str_121163], **kwargs_121164)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 557, 8), TypeError_call_result_121165, 'raise parameter', BaseException)
    # SSA join for if statement (line 556)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 558)
    x_121166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 7), 'x')
    # Obtaining the member 'size' of a type (line 558)
    size_121167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 7), x_121166, 'size')
    int_121168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 17), 'int')
    # Applying the binary operator '==' (line 558)
    result_eq_121169 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 7), '==', size_121167, int_121168)
    
    # Testing the type of an if condition (line 558)
    if_condition_121170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 558, 4), result_eq_121169)
    # Assigning a type to the variable 'if_condition_121170' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'if_condition_121170', if_condition_121170)
    # SSA begins for if statement (line 558)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 559)
    # Processing the call arguments (line 559)
    str_121172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 559)
    kwargs_121173 = {}
    # Getting the type of 'TypeError' (line 559)
    TypeError_121171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 559)
    TypeError_call_result_121174 = invoke(stypy.reporting.localization.Localization(__file__, 559, 14), TypeError_121171, *[str_121172], **kwargs_121173)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 559, 8), TypeError_call_result_121174, 'raise parameter', BaseException)
    # SSA join for if statement (line 558)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 560)
    y_121175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 560)
    ndim_121176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 7), y_121175, 'ndim')
    int_121177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 16), 'int')
    # Applying the binary operator '<' (line 560)
    result_lt_121178 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 7), '<', ndim_121176, int_121177)
    
    
    # Getting the type of 'y' (line 560)
    y_121179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 560)
    ndim_121180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 21), y_121179, 'ndim')
    int_121181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 30), 'int')
    # Applying the binary operator '>' (line 560)
    result_gt_121182 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 21), '>', ndim_121180, int_121181)
    
    # Applying the binary operator 'or' (line 560)
    result_or_keyword_121183 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 7), 'or', result_lt_121178, result_gt_121182)
    
    # Testing the type of an if condition (line 560)
    if_condition_121184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 4), result_or_keyword_121183)
    # Assigning a type to the variable 'if_condition_121184' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'if_condition_121184', if_condition_121184)
    # SSA begins for if statement (line 560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 561)
    # Processing the call arguments (line 561)
    str_121186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 561)
    kwargs_121187 = {}
    # Getting the type of 'TypeError' (line 561)
    TypeError_121185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 561)
    TypeError_call_result_121188 = invoke(stypy.reporting.localization.Localization(__file__, 561, 14), TypeError_121185, *[str_121186], **kwargs_121187)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 561, 8), TypeError_call_result_121188, 'raise parameter', BaseException)
    # SSA join for if statement (line 560)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_121189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 15), 'int')
    # Getting the type of 'x' (line 562)
    x_121190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 7), 'x')
    # Obtaining the member 'shape' of a type (line 562)
    shape_121191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 7), x_121190, 'shape')
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___121192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 7), shape_121191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_121193 = invoke(stypy.reporting.localization.Localization(__file__, 562, 7), getitem___121192, int_121189)
    
    
    # Obtaining the type of the subscript
    int_121194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 29), 'int')
    # Getting the type of 'y' (line 562)
    y_121195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 21), 'y')
    # Obtaining the member 'shape' of a type (line 562)
    shape_121196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 21), y_121195, 'shape')
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___121197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 21), shape_121196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_121198 = invoke(stypy.reporting.localization.Localization(__file__, 562, 21), getitem___121197, int_121194)
    
    # Applying the binary operator '!=' (line 562)
    result_ne_121199 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 7), '!=', subscript_call_result_121193, subscript_call_result_121198)
    
    # Testing the type of an if condition (line 562)
    if_condition_121200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 4), result_ne_121199)
    # Assigning a type to the variable 'if_condition_121200' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'if_condition_121200', if_condition_121200)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 563)
    # Processing the call arguments (line 563)
    str_121202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 563)
    kwargs_121203 = {}
    # Getting the type of 'TypeError' (line 563)
    TypeError_121201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 563)
    TypeError_call_result_121204 = invoke(stypy.reporting.localization.Localization(__file__, 563, 14), TypeError_121201, *[str_121202], **kwargs_121203)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 563, 8), TypeError_call_result_121204, 'raise parameter', BaseException)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 566)
    # Getting the type of 'rcond' (line 566)
    rcond_121205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 7), 'rcond')
    # Getting the type of 'None' (line 566)
    None_121206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'None')
    
    (may_be_121207, more_types_in_union_121208) = may_be_none(rcond_121205, None_121206)

    if may_be_121207:

        if more_types_in_union_121208:
            # Runtime conditional SSA (line 566)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 567):
        
        # Assigning a BinOp to a Name (line 567):
        
        # Call to len(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'x' (line 567)
        x_121210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'x', False)
        # Processing the call keyword arguments (line 567)
        kwargs_121211 = {}
        # Getting the type of 'len' (line 567)
        len_121209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 16), 'len', False)
        # Calling len(args, kwargs) (line 567)
        len_call_result_121212 = invoke(stypy.reporting.localization.Localization(__file__, 567, 16), len_121209, *[x_121210], **kwargs_121211)
        
        
        # Call to finfo(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'x' (line 567)
        x_121214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 29), 'x', False)
        # Obtaining the member 'dtype' of a type (line 567)
        dtype_121215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 29), x_121214, 'dtype')
        # Processing the call keyword arguments (line 567)
        kwargs_121216 = {}
        # Getting the type of 'finfo' (line 567)
        finfo_121213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'finfo', False)
        # Calling finfo(args, kwargs) (line 567)
        finfo_call_result_121217 = invoke(stypy.reporting.localization.Localization(__file__, 567, 23), finfo_121213, *[dtype_121215], **kwargs_121216)
        
        # Obtaining the member 'eps' of a type (line 567)
        eps_121218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 23), finfo_call_result_121217, 'eps')
        # Applying the binary operator '*' (line 567)
        result_mul_121219 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 16), '*', len_call_result_121212, eps_121218)
        
        # Assigning a type to the variable 'rcond' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'rcond', result_mul_121219)

        if more_types_in_union_121208:
            # SSA join for if statement (line 566)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 570):
    
    # Assigning a Call to a Name (line 570):
    
    # Call to vander(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'x' (line 570)
    x_121221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 17), 'x', False)
    # Getting the type of 'order' (line 570)
    order_121222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 20), 'order', False)
    # Processing the call keyword arguments (line 570)
    kwargs_121223 = {}
    # Getting the type of 'vander' (line 570)
    vander_121220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 10), 'vander', False)
    # Calling vander(args, kwargs) (line 570)
    vander_call_result_121224 = invoke(stypy.reporting.localization.Localization(__file__, 570, 10), vander_121220, *[x_121221, order_121222], **kwargs_121223)
    
    # Assigning a type to the variable 'lhs' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'lhs', vander_call_result_121224)
    
    # Assigning a Name to a Name (line 571):
    
    # Assigning a Name to a Name (line 571):
    # Getting the type of 'y' (line 571)
    y_121225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 10), 'y')
    # Assigning a type to the variable 'rhs' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'rhs', y_121225)
    
    # Type idiom detected: calculating its left and rigth part (line 574)
    # Getting the type of 'w' (line 574)
    w_121226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'w')
    # Getting the type of 'None' (line 574)
    None_121227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'None')
    
    (may_be_121228, more_types_in_union_121229) = may_not_be_none(w_121226, None_121227)

    if may_be_121228:

        if more_types_in_union_121229:
            # Runtime conditional SSA (line 574)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 575):
        
        # Assigning a BinOp to a Name (line 575):
        
        # Call to asarray(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'w' (line 575)
        w_121232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'w', False)
        # Processing the call keyword arguments (line 575)
        kwargs_121233 = {}
        # Getting the type of 'NX' (line 575)
        NX_121230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'NX', False)
        # Obtaining the member 'asarray' of a type (line 575)
        asarray_121231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), NX_121230, 'asarray')
        # Calling asarray(args, kwargs) (line 575)
        asarray_call_result_121234 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), asarray_121231, *[w_121232], **kwargs_121233)
        
        float_121235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 28), 'float')
        # Applying the binary operator '+' (line 575)
        result_add_121236 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 12), '+', asarray_call_result_121234, float_121235)
        
        # Assigning a type to the variable 'w' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'w', result_add_121236)
        
        
        # Getting the type of 'w' (line 576)
        w_121237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 576)
        ndim_121238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 11), w_121237, 'ndim')
        int_121239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 21), 'int')
        # Applying the binary operator '!=' (line 576)
        result_ne_121240 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 11), '!=', ndim_121238, int_121239)
        
        # Testing the type of an if condition (line 576)
        if_condition_121241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 8), result_ne_121240)
        # Assigning a type to the variable 'if_condition_121241' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'if_condition_121241', if_condition_121241)
        # SSA begins for if statement (line 576)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 577)
        # Processing the call arguments (line 577)
        str_121243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 28), 'str', 'expected a 1-d array for weights')
        # Processing the call keyword arguments (line 577)
        kwargs_121244 = {}
        # Getting the type of 'TypeError' (line 577)
        TypeError_121242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 577)
        TypeError_call_result_121245 = invoke(stypy.reporting.localization.Localization(__file__, 577, 18), TypeError_121242, *[str_121243], **kwargs_121244)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 577, 12), TypeError_call_result_121245, 'raise parameter', BaseException)
        # SSA join for if statement (line 576)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_121246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 19), 'int')
        # Getting the type of 'w' (line 578)
        w_121247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), 'w')
        # Obtaining the member 'shape' of a type (line 578)
        shape_121248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 11), w_121247, 'shape')
        # Obtaining the member '__getitem__' of a type (line 578)
        getitem___121249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 11), shape_121248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 578)
        subscript_call_result_121250 = invoke(stypy.reporting.localization.Localization(__file__, 578, 11), getitem___121249, int_121246)
        
        
        # Obtaining the type of the subscript
        int_121251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 33), 'int')
        # Getting the type of 'y' (line 578)
        y_121252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 25), 'y')
        # Obtaining the member 'shape' of a type (line 578)
        shape_121253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 25), y_121252, 'shape')
        # Obtaining the member '__getitem__' of a type (line 578)
        getitem___121254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 25), shape_121253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 578)
        subscript_call_result_121255 = invoke(stypy.reporting.localization.Localization(__file__, 578, 25), getitem___121254, int_121251)
        
        # Applying the binary operator '!=' (line 578)
        result_ne_121256 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 11), '!=', subscript_call_result_121250, subscript_call_result_121255)
        
        # Testing the type of an if condition (line 578)
        if_condition_121257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 8), result_ne_121256)
        # Assigning a type to the variable 'if_condition_121257' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'if_condition_121257', if_condition_121257)
        # SSA begins for if statement (line 578)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 579)
        # Processing the call arguments (line 579)
        str_121259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 28), 'str', 'expected w and y to have the same length')
        # Processing the call keyword arguments (line 579)
        kwargs_121260 = {}
        # Getting the type of 'TypeError' (line 579)
        TypeError_121258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 579)
        TypeError_call_result_121261 = invoke(stypy.reporting.localization.Localization(__file__, 579, 18), TypeError_121258, *[str_121259], **kwargs_121260)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 579, 12), TypeError_call_result_121261, 'raise parameter', BaseException)
        # SSA join for if statement (line 578)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'lhs' (line 580)
        lhs_121262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'lhs')
        
        # Obtaining the type of the subscript
        slice_121263 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 580, 15), None, None, None)
        # Getting the type of 'NX' (line 580)
        NX_121264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), 'NX')
        # Obtaining the member 'newaxis' of a type (line 580)
        newaxis_121265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 20), NX_121264, 'newaxis')
        # Getting the type of 'w' (line 580)
        w_121266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'w')
        # Obtaining the member '__getitem__' of a type (line 580)
        getitem___121267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 15), w_121266, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 580)
        subscript_call_result_121268 = invoke(stypy.reporting.localization.Localization(__file__, 580, 15), getitem___121267, (slice_121263, newaxis_121265))
        
        # Applying the binary operator '*=' (line 580)
        result_imul_121269 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 8), '*=', lhs_121262, subscript_call_result_121268)
        # Assigning a type to the variable 'lhs' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'lhs', result_imul_121269)
        
        
        
        # Getting the type of 'rhs' (line 581)
        rhs_121270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 11), 'rhs')
        # Obtaining the member 'ndim' of a type (line 581)
        ndim_121271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 11), rhs_121270, 'ndim')
        int_121272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 23), 'int')
        # Applying the binary operator '==' (line 581)
        result_eq_121273 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 11), '==', ndim_121271, int_121272)
        
        # Testing the type of an if condition (line 581)
        if_condition_121274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 8), result_eq_121273)
        # Assigning a type to the variable 'if_condition_121274' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'if_condition_121274', if_condition_121274)
        # SSA begins for if statement (line 581)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'rhs' (line 582)
        rhs_121275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'rhs')
        
        # Obtaining the type of the subscript
        slice_121276 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 582, 19), None, None, None)
        # Getting the type of 'NX' (line 582)
        NX_121277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 24), 'NX')
        # Obtaining the member 'newaxis' of a type (line 582)
        newaxis_121278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 24), NX_121277, 'newaxis')
        # Getting the type of 'w' (line 582)
        w_121279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 19), 'w')
        # Obtaining the member '__getitem__' of a type (line 582)
        getitem___121280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 19), w_121279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 582)
        subscript_call_result_121281 = invoke(stypy.reporting.localization.Localization(__file__, 582, 19), getitem___121280, (slice_121276, newaxis_121278))
        
        # Applying the binary operator '*=' (line 582)
        result_imul_121282 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 12), '*=', rhs_121275, subscript_call_result_121281)
        # Assigning a type to the variable 'rhs' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'rhs', result_imul_121282)
        
        # SSA branch for the else part of an if statement (line 581)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'rhs' (line 584)
        rhs_121283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'rhs')
        # Getting the type of 'w' (line 584)
        w_121284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 19), 'w')
        # Applying the binary operator '*=' (line 584)
        result_imul_121285 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 12), '*=', rhs_121283, w_121284)
        # Assigning a type to the variable 'rhs' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'rhs', result_imul_121285)
        
        # SSA join for if statement (line 581)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_121229:
            # SSA join for if statement (line 574)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 587):
    
    # Assigning a Call to a Name (line 587):
    
    # Call to sqrt(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Call to sum(...): (line 587)
    # Processing the call keyword arguments (line 587)
    int_121292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 39), 'int')
    keyword_121293 = int_121292
    kwargs_121294 = {'axis': keyword_121293}
    # Getting the type of 'lhs' (line 587)
    lhs_121288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 21), 'lhs', False)
    # Getting the type of 'lhs' (line 587)
    lhs_121289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 25), 'lhs', False)
    # Applying the binary operator '*' (line 587)
    result_mul_121290 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 21), '*', lhs_121288, lhs_121289)
    
    # Obtaining the member 'sum' of a type (line 587)
    sum_121291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 21), result_mul_121290, 'sum')
    # Calling sum(args, kwargs) (line 587)
    sum_call_result_121295 = invoke(stypy.reporting.localization.Localization(__file__, 587, 21), sum_121291, *[], **kwargs_121294)
    
    # Processing the call keyword arguments (line 587)
    kwargs_121296 = {}
    # Getting the type of 'NX' (line 587)
    NX_121286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'NX', False)
    # Obtaining the member 'sqrt' of a type (line 587)
    sqrt_121287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 12), NX_121286, 'sqrt')
    # Calling sqrt(args, kwargs) (line 587)
    sqrt_call_result_121297 = invoke(stypy.reporting.localization.Localization(__file__, 587, 12), sqrt_121287, *[sum_call_result_121295], **kwargs_121296)
    
    # Assigning a type to the variable 'scale' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'scale', sqrt_call_result_121297)
    
    # Getting the type of 'lhs' (line 588)
    lhs_121298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'lhs')
    # Getting the type of 'scale' (line 588)
    scale_121299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 11), 'scale')
    # Applying the binary operator 'div=' (line 588)
    result_div_121300 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 4), 'div=', lhs_121298, scale_121299)
    # Assigning a type to the variable 'lhs' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'lhs', result_div_121300)
    
    
    # Assigning a Call to a Tuple (line 589):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'lhs' (line 589)
    lhs_121302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 31), 'lhs', False)
    # Getting the type of 'rhs' (line 589)
    rhs_121303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 36), 'rhs', False)
    # Getting the type of 'rcond' (line 589)
    rcond_121304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 41), 'rcond', False)
    # Processing the call keyword arguments (line 589)
    kwargs_121305 = {}
    # Getting the type of 'lstsq' (line 589)
    lstsq_121301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 25), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 589)
    lstsq_call_result_121306 = invoke(stypy.reporting.localization.Localization(__file__, 589, 25), lstsq_121301, *[lhs_121302, rhs_121303, rcond_121304], **kwargs_121305)
    
    # Assigning a type to the variable 'call_assignment_120576' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120576', lstsq_call_result_121306)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_121309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'int')
    # Processing the call keyword arguments
    kwargs_121310 = {}
    # Getting the type of 'call_assignment_120576' (line 589)
    call_assignment_120576_121307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120576', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___121308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), call_assignment_120576_121307, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_121311 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121308, *[int_121309], **kwargs_121310)
    
    # Assigning a type to the variable 'call_assignment_120577' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120577', getitem___call_result_121311)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_120577' (line 589)
    call_assignment_120577_121312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120577')
    # Assigning a type to the variable 'c' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'c', call_assignment_120577_121312)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_121315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'int')
    # Processing the call keyword arguments
    kwargs_121316 = {}
    # Getting the type of 'call_assignment_120576' (line 589)
    call_assignment_120576_121313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120576', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___121314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), call_assignment_120576_121313, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_121317 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121314, *[int_121315], **kwargs_121316)
    
    # Assigning a type to the variable 'call_assignment_120578' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120578', getitem___call_result_121317)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_120578' (line 589)
    call_assignment_120578_121318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120578')
    # Assigning a type to the variable 'resids' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 7), 'resids', call_assignment_120578_121318)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_121321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'int')
    # Processing the call keyword arguments
    kwargs_121322 = {}
    # Getting the type of 'call_assignment_120576' (line 589)
    call_assignment_120576_121319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120576', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___121320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), call_assignment_120576_121319, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_121323 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121320, *[int_121321], **kwargs_121322)
    
    # Assigning a type to the variable 'call_assignment_120579' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120579', getitem___call_result_121323)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_120579' (line 589)
    call_assignment_120579_121324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120579')
    # Assigning a type to the variable 'rank' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'rank', call_assignment_120579_121324)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_121327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'int')
    # Processing the call keyword arguments
    kwargs_121328 = {}
    # Getting the type of 'call_assignment_120576' (line 589)
    call_assignment_120576_121325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120576', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___121326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), call_assignment_120576_121325, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_121329 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121326, *[int_121327], **kwargs_121328)
    
    # Assigning a type to the variable 'call_assignment_120580' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120580', getitem___call_result_121329)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_120580' (line 589)
    call_assignment_120580_121330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_120580')
    # Assigning a type to the variable 's' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 's', call_assignment_120580_121330)
    
    # Assigning a Attribute to a Name (line 590):
    
    # Assigning a Attribute to a Name (line 590):
    # Getting the type of 'c' (line 590)
    c_121331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 9), 'c')
    # Obtaining the member 'T' of a type (line 590)
    T_121332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 9), c_121331, 'T')
    # Getting the type of 'scale' (line 590)
    scale_121333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 13), 'scale')
    # Applying the binary operator 'div' (line 590)
    result_div_121334 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 9), 'div', T_121332, scale_121333)
    
    # Obtaining the member 'T' of a type (line 590)
    T_121335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 9), result_div_121334, 'T')
    # Assigning a type to the variable 'c' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'c', T_121335)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 593)
    rank_121336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 7), 'rank')
    # Getting the type of 'order' (line 593)
    order_121337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 15), 'order')
    # Applying the binary operator '!=' (line 593)
    result_ne_121338 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 7), '!=', rank_121336, order_121337)
    
    
    # Getting the type of 'full' (line 593)
    full_121339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 29), 'full')
    # Applying the 'not' unary operator (line 593)
    result_not__121340 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 25), 'not', full_121339)
    
    # Applying the binary operator 'and' (line 593)
    result_and_keyword_121341 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 7), 'and', result_ne_121338, result_not__121340)
    
    # Testing the type of an if condition (line 593)
    if_condition_121342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 4), result_and_keyword_121341)
    # Assigning a type to the variable 'if_condition_121342' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'if_condition_121342', if_condition_121342)
    # SSA begins for if statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 594):
    
    # Assigning a Str to a Name (line 594):
    str_121343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 14), 'str', 'Polyfit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'msg', str_121343)
    
    # Call to warn(...): (line 595)
    # Processing the call arguments (line 595)
    # Getting the type of 'msg' (line 595)
    msg_121346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 22), 'msg', False)
    # Getting the type of 'RankWarning' (line 595)
    RankWarning_121347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 27), 'RankWarning', False)
    # Processing the call keyword arguments (line 595)
    kwargs_121348 = {}
    # Getting the type of 'warnings' (line 595)
    warnings_121344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 595)
    warn_121345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 8), warnings_121344, 'warn')
    # Calling warn(args, kwargs) (line 595)
    warn_call_result_121349 = invoke(stypy.reporting.localization.Localization(__file__, 595, 8), warn_121345, *[msg_121346, RankWarning_121347], **kwargs_121348)
    
    # SSA join for if statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 597)
    full_121350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 7), 'full')
    # Testing the type of an if condition (line 597)
    if_condition_121351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 4), full_121350)
    # Assigning a type to the variable 'if_condition_121351' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'if_condition_121351', if_condition_121351)
    # SSA begins for if statement (line 597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 598)
    tuple_121352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 598)
    # Adding element type (line 598)
    # Getting the type of 'c' (line 598)
    c_121353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121352, c_121353)
    # Adding element type (line 598)
    # Getting the type of 'resids' (line 598)
    resids_121354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 18), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121352, resids_121354)
    # Adding element type (line 598)
    # Getting the type of 'rank' (line 598)
    rank_121355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 26), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121352, rank_121355)
    # Adding element type (line 598)
    # Getting the type of 's' (line 598)
    s_121356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 32), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121352, s_121356)
    # Adding element type (line 598)
    # Getting the type of 'rcond' (line 598)
    rcond_121357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 35), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121352, rcond_121357)
    
    # Assigning a type to the variable 'stypy_return_type' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'stypy_return_type', tuple_121352)
    # SSA branch for the else part of an if statement (line 597)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'cov' (line 599)
    cov_121358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 9), 'cov')
    # Testing the type of an if condition (line 599)
    if_condition_121359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 9), cov_121358)
    # Assigning a type to the variable 'if_condition_121359' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 9), 'if_condition_121359', if_condition_121359)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 600):
    
    # Assigning a Call to a Name (line 600):
    
    # Call to inv(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Call to dot(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'lhs' (line 600)
    lhs_121362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'lhs', False)
    # Obtaining the member 'T' of a type (line 600)
    T_121363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 24), lhs_121362, 'T')
    # Getting the type of 'lhs' (line 600)
    lhs_121364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 31), 'lhs', False)
    # Processing the call keyword arguments (line 600)
    kwargs_121365 = {}
    # Getting the type of 'dot' (line 600)
    dot_121361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 20), 'dot', False)
    # Calling dot(args, kwargs) (line 600)
    dot_call_result_121366 = invoke(stypy.reporting.localization.Localization(__file__, 600, 20), dot_121361, *[T_121363, lhs_121364], **kwargs_121365)
    
    # Processing the call keyword arguments (line 600)
    kwargs_121367 = {}
    # Getting the type of 'inv' (line 600)
    inv_121360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'inv', False)
    # Calling inv(args, kwargs) (line 600)
    inv_call_result_121368 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), inv_121360, *[dot_call_result_121366], **kwargs_121367)
    
    # Assigning a type to the variable 'Vbase' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'Vbase', inv_call_result_121368)
    
    # Getting the type of 'Vbase' (line 601)
    Vbase_121369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'Vbase')
    
    # Call to outer(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'scale' (line 601)
    scale_121372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 26), 'scale', False)
    # Getting the type of 'scale' (line 601)
    scale_121373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 33), 'scale', False)
    # Processing the call keyword arguments (line 601)
    kwargs_121374 = {}
    # Getting the type of 'NX' (line 601)
    NX_121370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 17), 'NX', False)
    # Obtaining the member 'outer' of a type (line 601)
    outer_121371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 17), NX_121370, 'outer')
    # Calling outer(args, kwargs) (line 601)
    outer_call_result_121375 = invoke(stypy.reporting.localization.Localization(__file__, 601, 17), outer_121371, *[scale_121372, scale_121373], **kwargs_121374)
    
    # Applying the binary operator 'div=' (line 601)
    result_div_121376 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 8), 'div=', Vbase_121369, outer_call_result_121375)
    # Assigning a type to the variable 'Vbase' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'Vbase', result_div_121376)
    
    
    # Assigning a BinOp to a Name (line 606):
    
    # Assigning a BinOp to a Name (line 606):
    # Getting the type of 'resids' (line 606)
    resids_121377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 14), 'resids')
    
    # Call to len(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'x' (line 606)
    x_121379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 28), 'x', False)
    # Processing the call keyword arguments (line 606)
    kwargs_121380 = {}
    # Getting the type of 'len' (line 606)
    len_121378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 24), 'len', False)
    # Calling len(args, kwargs) (line 606)
    len_call_result_121381 = invoke(stypy.reporting.localization.Localization(__file__, 606, 24), len_121378, *[x_121379], **kwargs_121380)
    
    # Getting the type of 'order' (line 606)
    order_121382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 33), 'order')
    # Applying the binary operator '-' (line 606)
    result_sub_121383 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 24), '-', len_call_result_121381, order_121382)
    
    float_121384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 41), 'float')
    # Applying the binary operator '-' (line 606)
    result_sub_121385 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 39), '-', result_sub_121383, float_121384)
    
    # Applying the binary operator 'div' (line 606)
    result_div_121386 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 14), 'div', resids_121377, result_sub_121385)
    
    # Assigning a type to the variable 'fac' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'fac', result_div_121386)
    
    
    # Getting the type of 'y' (line 607)
    y_121387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 11), 'y')
    # Obtaining the member 'ndim' of a type (line 607)
    ndim_121388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 11), y_121387, 'ndim')
    int_121389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 21), 'int')
    # Applying the binary operator '==' (line 607)
    result_eq_121390 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 11), '==', ndim_121388, int_121389)
    
    # Testing the type of an if condition (line 607)
    if_condition_121391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 8), result_eq_121390)
    # Assigning a type to the variable 'if_condition_121391' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'if_condition_121391', if_condition_121391)
    # SSA begins for if statement (line 607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 608)
    tuple_121392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 608)
    # Adding element type (line 608)
    # Getting the type of 'c' (line 608)
    c_121393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 19), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 19), tuple_121392, c_121393)
    # Adding element type (line 608)
    # Getting the type of 'Vbase' (line 608)
    Vbase_121394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 22), 'Vbase')
    # Getting the type of 'fac' (line 608)
    fac_121395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 30), 'fac')
    # Applying the binary operator '*' (line 608)
    result_mul_121396 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 22), '*', Vbase_121394, fac_121395)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 19), tuple_121392, result_mul_121396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'stypy_return_type', tuple_121392)
    # SSA branch for the else part of an if statement (line 607)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 610)
    tuple_121397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 610)
    # Adding element type (line 610)
    # Getting the type of 'c' (line 610)
    c_121398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 19), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 19), tuple_121397, c_121398)
    # Adding element type (line 610)
    
    # Obtaining the type of the subscript
    slice_121399 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 610, 22), None, None, None)
    slice_121400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 610, 22), None, None, None)
    # Getting the type of 'NX' (line 610)
    NX_121401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 33), 'NX')
    # Obtaining the member 'newaxis' of a type (line 610)
    newaxis_121402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 33), NX_121401, 'newaxis')
    # Getting the type of 'Vbase' (line 610)
    Vbase_121403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 22), 'Vbase')
    # Obtaining the member '__getitem__' of a type (line 610)
    getitem___121404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 22), Vbase_121403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 610)
    subscript_call_result_121405 = invoke(stypy.reporting.localization.Localization(__file__, 610, 22), getitem___121404, (slice_121399, slice_121400, newaxis_121402))
    
    # Getting the type of 'fac' (line 610)
    fac_121406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 47), 'fac')
    # Applying the binary operator '*' (line 610)
    result_mul_121407 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 22), '*', subscript_call_result_121405, fac_121406)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 19), tuple_121397, result_mul_121407)
    
    # Assigning a type to the variable 'stypy_return_type' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'stypy_return_type', tuple_121397)
    # SSA join for if statement (line 607)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 599)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 612)
    c_121408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'stypy_return_type', c_121408)
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'polyfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyfit' in the type store
    # Getting the type of 'stypy_return_type' (line 402)
    stypy_return_type_121409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyfit'
    return stypy_return_type_121409

# Assigning a type to the variable 'polyfit' (line 402)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'polyfit', polyfit)

@norecursion
def polyval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyval'
    module_type_store = module_type_store.open_function_context('polyval', 615, 0, False)
    
    # Passed parameters checking function
    polyval.stypy_localization = localization
    polyval.stypy_type_of_self = None
    polyval.stypy_type_store = module_type_store
    polyval.stypy_function_name = 'polyval'
    polyval.stypy_param_names_list = ['p', 'x']
    polyval.stypy_varargs_param_name = None
    polyval.stypy_kwargs_param_name = None
    polyval.stypy_call_defaults = defaults
    polyval.stypy_call_varargs = varargs
    polyval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyval', ['p', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyval', localization, ['p', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyval(...)' code ##################

    str_121410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, (-1)), 'str', '\n    Evaluate a polynomial at specific values.\n\n    If `p` is of length N, this function returns the value:\n\n        ``p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]``\n\n    If `x` is a sequence, then `p(x)` is returned for each element of `x`.\n    If `x` is another polynomial then the composite polynomial `p(x(t))`\n    is returned.\n\n    Parameters\n    ----------\n    p : array_like or poly1d object\n       1D array of polynomial coefficients (including coefficients equal\n       to zero) from highest degree to the constant term, or an\n       instance of poly1d.\n    x : array_like or poly1d object\n       A number, an array of numbers, or an instance of poly1d, at\n       which to evaluate `p`.\n\n    Returns\n    -------\n    values : ndarray or poly1d\n       If `x` is a poly1d instance, the result is the composition of the two\n       polynomials, i.e., `x` is "substituted" in `p` and the simplified\n       result is returned. In addition, the type of `x` - array_like or\n       poly1d - governs the type of the output: `x` array_like => `values`\n       array_like, `x` a poly1d object => `values` is also.\n\n    See Also\n    --------\n    poly1d: A polynomial class.\n\n    Notes\n    -----\n    Horner\'s scheme [1]_ is used to evaluate the polynomial. Even so,\n    for polynomials of high degree the values may be inaccurate due to\n    rounding errors. Use carefully.\n\n    References\n    ----------\n    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.\n       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand\n       Reinhold Co., 1985, pg. 720.\n\n    Examples\n    --------\n    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1\n    76\n    >>> np.polyval([3,0,1], np.poly1d(5))\n    poly1d([ 76.])\n    >>> np.polyval(np.poly1d([3,0,1]), 5)\n    76\n    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))\n    poly1d([ 76.])\n\n    ')
    
    # Assigning a Call to a Name (line 674):
    
    # Assigning a Call to a Name (line 674):
    
    # Call to asarray(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'p' (line 674)
    p_121413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 19), 'p', False)
    # Processing the call keyword arguments (line 674)
    kwargs_121414 = {}
    # Getting the type of 'NX' (line 674)
    NX_121411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 674)
    asarray_121412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 8), NX_121411, 'asarray')
    # Calling asarray(args, kwargs) (line 674)
    asarray_call_result_121415 = invoke(stypy.reporting.localization.Localization(__file__, 674, 8), asarray_121412, *[p_121413], **kwargs_121414)
    
    # Assigning a type to the variable 'p' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'p', asarray_call_result_121415)
    
    
    # Call to isinstance(...): (line 675)
    # Processing the call arguments (line 675)
    # Getting the type of 'x' (line 675)
    x_121417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 18), 'x', False)
    # Getting the type of 'poly1d' (line 675)
    poly1d_121418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 21), 'poly1d', False)
    # Processing the call keyword arguments (line 675)
    kwargs_121419 = {}
    # Getting the type of 'isinstance' (line 675)
    isinstance_121416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 675)
    isinstance_call_result_121420 = invoke(stypy.reporting.localization.Localization(__file__, 675, 7), isinstance_121416, *[x_121417, poly1d_121418], **kwargs_121419)
    
    # Testing the type of an if condition (line 675)
    if_condition_121421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 4), isinstance_call_result_121420)
    # Assigning a type to the variable 'if_condition_121421' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'if_condition_121421', if_condition_121421)
    # SSA begins for if statement (line 675)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 676):
    
    # Assigning a Num to a Name (line 676):
    int_121422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 12), 'int')
    # Assigning a type to the variable 'y' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'y', int_121422)
    # SSA branch for the else part of an if statement (line 675)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 678):
    
    # Assigning a Call to a Name (line 678):
    
    # Call to asarray(...): (line 678)
    # Processing the call arguments (line 678)
    # Getting the type of 'x' (line 678)
    x_121425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 23), 'x', False)
    # Processing the call keyword arguments (line 678)
    kwargs_121426 = {}
    # Getting the type of 'NX' (line 678)
    NX_121423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'NX', False)
    # Obtaining the member 'asarray' of a type (line 678)
    asarray_121424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 12), NX_121423, 'asarray')
    # Calling asarray(args, kwargs) (line 678)
    asarray_call_result_121427 = invoke(stypy.reporting.localization.Localization(__file__, 678, 12), asarray_121424, *[x_121425], **kwargs_121426)
    
    # Assigning a type to the variable 'x' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'x', asarray_call_result_121427)
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to zeros_like(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'x' (line 679)
    x_121430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 26), 'x', False)
    # Processing the call keyword arguments (line 679)
    kwargs_121431 = {}
    # Getting the type of 'NX' (line 679)
    NX_121428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'NX', False)
    # Obtaining the member 'zeros_like' of a type (line 679)
    zeros_like_121429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 12), NX_121428, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 679)
    zeros_like_call_result_121432 = invoke(stypy.reporting.localization.Localization(__file__, 679, 12), zeros_like_121429, *[x_121430], **kwargs_121431)
    
    # Assigning a type to the variable 'y' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'y', zeros_like_call_result_121432)
    # SSA join for if statement (line 675)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 680)
    # Processing the call arguments (line 680)
    
    # Call to len(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'p' (line 680)
    p_121435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 23), 'p', False)
    # Processing the call keyword arguments (line 680)
    kwargs_121436 = {}
    # Getting the type of 'len' (line 680)
    len_121434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 19), 'len', False)
    # Calling len(args, kwargs) (line 680)
    len_call_result_121437 = invoke(stypy.reporting.localization.Localization(__file__, 680, 19), len_121434, *[p_121435], **kwargs_121436)
    
    # Processing the call keyword arguments (line 680)
    kwargs_121438 = {}
    # Getting the type of 'range' (line 680)
    range_121433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 13), 'range', False)
    # Calling range(args, kwargs) (line 680)
    range_call_result_121439 = invoke(stypy.reporting.localization.Localization(__file__, 680, 13), range_121433, *[len_call_result_121437], **kwargs_121438)
    
    # Testing the type of a for loop iterable (line 680)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 680, 4), range_call_result_121439)
    # Getting the type of the for loop variable (line 680)
    for_loop_var_121440 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 680, 4), range_call_result_121439)
    # Assigning a type to the variable 'i' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'i', for_loop_var_121440)
    # SSA begins for a for statement (line 680)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 681):
    
    # Assigning a BinOp to a Name (line 681):
    # Getting the type of 'y' (line 681)
    y_121441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), 'y')
    # Getting the type of 'x' (line 681)
    x_121442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'x')
    # Applying the binary operator '*' (line 681)
    result_mul_121443 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 12), '*', y_121441, x_121442)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 681)
    i_121444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 22), 'i')
    # Getting the type of 'p' (line 681)
    p_121445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'p')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___121446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 20), p_121445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_121447 = invoke(stypy.reporting.localization.Localization(__file__, 681, 20), getitem___121446, i_121444)
    
    # Applying the binary operator '+' (line 681)
    result_add_121448 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 12), '+', result_mul_121443, subscript_call_result_121447)
    
    # Assigning a type to the variable 'y' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'y', result_add_121448)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'y' (line 682)
    y_121449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'stypy_return_type', y_121449)
    
    # ################# End of 'polyval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyval' in the type store
    # Getting the type of 'stypy_return_type' (line 615)
    stypy_return_type_121450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyval'
    return stypy_return_type_121450

# Assigning a type to the variable 'polyval' (line 615)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 0), 'polyval', polyval)

@norecursion
def polyadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polyadd'
    module_type_store = module_type_store.open_function_context('polyadd', 684, 0, False)
    
    # Passed parameters checking function
    polyadd.stypy_localization = localization
    polyadd.stypy_type_of_self = None
    polyadd.stypy_type_store = module_type_store
    polyadd.stypy_function_name = 'polyadd'
    polyadd.stypy_param_names_list = ['a1', 'a2']
    polyadd.stypy_varargs_param_name = None
    polyadd.stypy_kwargs_param_name = None
    polyadd.stypy_call_defaults = defaults
    polyadd.stypy_call_varargs = varargs
    polyadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyadd', ['a1', 'a2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyadd', localization, ['a1', 'a2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyadd(...)' code ##################

    str_121451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, (-1)), 'str', '\n    Find the sum of two polynomials.\n\n    Returns the polynomial resulting from the sum of two input polynomials.\n    Each input must be either a poly1d object or a 1D sequence of polynomial\n    coefficients, from highest to lowest degree.\n\n    Parameters\n    ----------\n    a1, a2 : array_like or poly1d object\n        Input polynomials.\n\n    Returns\n    -------\n    out : ndarray or poly1d object\n        The sum of the inputs. If either input is a poly1d object, then the\n        output is also a poly1d object. Otherwise, it is a 1D array of\n        polynomial coefficients from highest to lowest degree.\n\n    See Also\n    --------\n    poly1d : A one-dimensional polynomial class.\n    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval\n\n    Examples\n    --------\n    >>> np.polyadd([1, 2], [9, 5, 4])\n    array([9, 6, 6])\n\n    Using poly1d objects:\n\n    >>> p1 = np.poly1d([1, 2])\n    >>> p2 = np.poly1d([9, 5, 4])\n    >>> print(p1)\n    1 x + 2\n    >>> print(p2)\n       2\n    9 x + 5 x + 4\n    >>> print(np.polyadd(p1, p2))\n       2\n    9 x + 6 x + 6\n\n    ')
    
    # Assigning a BoolOp to a Name (line 728):
    
    # Assigning a BoolOp to a Name (line 728):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'a1' (line 728)
    a1_121453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 27), 'a1', False)
    # Getting the type of 'poly1d' (line 728)
    poly1d_121454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 31), 'poly1d', False)
    # Processing the call keyword arguments (line 728)
    kwargs_121455 = {}
    # Getting the type of 'isinstance' (line 728)
    isinstance_121452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 728)
    isinstance_call_result_121456 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), isinstance_121452, *[a1_121453, poly1d_121454], **kwargs_121455)
    
    
    # Call to isinstance(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'a2' (line 728)
    a2_121458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 53), 'a2', False)
    # Getting the type of 'poly1d' (line 728)
    poly1d_121459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 57), 'poly1d', False)
    # Processing the call keyword arguments (line 728)
    kwargs_121460 = {}
    # Getting the type of 'isinstance' (line 728)
    isinstance_121457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 42), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 728)
    isinstance_call_result_121461 = invoke(stypy.reporting.localization.Localization(__file__, 728, 42), isinstance_121457, *[a2_121458, poly1d_121459], **kwargs_121460)
    
    # Applying the binary operator 'or' (line 728)
    result_or_keyword_121462 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 16), 'or', isinstance_call_result_121456, isinstance_call_result_121461)
    
    # Assigning a type to the variable 'truepoly' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'truepoly', result_or_keyword_121462)
    
    # Assigning a Call to a Name (line 729):
    
    # Assigning a Call to a Name (line 729):
    
    # Call to atleast_1d(...): (line 729)
    # Processing the call arguments (line 729)
    # Getting the type of 'a1' (line 729)
    a1_121464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 20), 'a1', False)
    # Processing the call keyword arguments (line 729)
    kwargs_121465 = {}
    # Getting the type of 'atleast_1d' (line 729)
    atleast_1d_121463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 9), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 729)
    atleast_1d_call_result_121466 = invoke(stypy.reporting.localization.Localization(__file__, 729, 9), atleast_1d_121463, *[a1_121464], **kwargs_121465)
    
    # Assigning a type to the variable 'a1' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'a1', atleast_1d_call_result_121466)
    
    # Assigning a Call to a Name (line 730):
    
    # Assigning a Call to a Name (line 730):
    
    # Call to atleast_1d(...): (line 730)
    # Processing the call arguments (line 730)
    # Getting the type of 'a2' (line 730)
    a2_121468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 20), 'a2', False)
    # Processing the call keyword arguments (line 730)
    kwargs_121469 = {}
    # Getting the type of 'atleast_1d' (line 730)
    atleast_1d_121467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 9), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 730)
    atleast_1d_call_result_121470 = invoke(stypy.reporting.localization.Localization(__file__, 730, 9), atleast_1d_121467, *[a2_121468], **kwargs_121469)
    
    # Assigning a type to the variable 'a2' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'a2', atleast_1d_call_result_121470)
    
    # Assigning a BinOp to a Name (line 731):
    
    # Assigning a BinOp to a Name (line 731):
    
    # Call to len(...): (line 731)
    # Processing the call arguments (line 731)
    # Getting the type of 'a2' (line 731)
    a2_121472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 15), 'a2', False)
    # Processing the call keyword arguments (line 731)
    kwargs_121473 = {}
    # Getting the type of 'len' (line 731)
    len_121471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 11), 'len', False)
    # Calling len(args, kwargs) (line 731)
    len_call_result_121474 = invoke(stypy.reporting.localization.Localization(__file__, 731, 11), len_121471, *[a2_121472], **kwargs_121473)
    
    
    # Call to len(...): (line 731)
    # Processing the call arguments (line 731)
    # Getting the type of 'a1' (line 731)
    a1_121476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 25), 'a1', False)
    # Processing the call keyword arguments (line 731)
    kwargs_121477 = {}
    # Getting the type of 'len' (line 731)
    len_121475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 21), 'len', False)
    # Calling len(args, kwargs) (line 731)
    len_call_result_121478 = invoke(stypy.reporting.localization.Localization(__file__, 731, 21), len_121475, *[a1_121476], **kwargs_121477)
    
    # Applying the binary operator '-' (line 731)
    result_sub_121479 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 11), '-', len_call_result_121474, len_call_result_121478)
    
    # Assigning a type to the variable 'diff' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'diff', result_sub_121479)
    
    
    # Getting the type of 'diff' (line 732)
    diff_121480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 7), 'diff')
    int_121481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 15), 'int')
    # Applying the binary operator '==' (line 732)
    result_eq_121482 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 7), '==', diff_121480, int_121481)
    
    # Testing the type of an if condition (line 732)
    if_condition_121483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 732, 4), result_eq_121482)
    # Assigning a type to the variable 'if_condition_121483' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'if_condition_121483', if_condition_121483)
    # SSA begins for if statement (line 732)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 733):
    
    # Assigning a BinOp to a Name (line 733):
    # Getting the type of 'a1' (line 733)
    a1_121484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 14), 'a1')
    # Getting the type of 'a2' (line 733)
    a2_121485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 19), 'a2')
    # Applying the binary operator '+' (line 733)
    result_add_121486 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 14), '+', a1_121484, a2_121485)
    
    # Assigning a type to the variable 'val' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'val', result_add_121486)
    # SSA branch for the else part of an if statement (line 732)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'diff' (line 734)
    diff_121487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 9), 'diff')
    int_121488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 16), 'int')
    # Applying the binary operator '>' (line 734)
    result_gt_121489 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 9), '>', diff_121487, int_121488)
    
    # Testing the type of an if condition (line 734)
    if_condition_121490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 9), result_gt_121489)
    # Assigning a type to the variable 'if_condition_121490' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 9), 'if_condition_121490', if_condition_121490)
    # SSA begins for if statement (line 734)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 735):
    
    # Assigning a Call to a Name (line 735):
    
    # Call to zeros(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'diff' (line 735)
    diff_121493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 22), 'diff', False)
    # Getting the type of 'a1' (line 735)
    a1_121494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 28), 'a1', False)
    # Obtaining the member 'dtype' of a type (line 735)
    dtype_121495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 28), a1_121494, 'dtype')
    # Processing the call keyword arguments (line 735)
    kwargs_121496 = {}
    # Getting the type of 'NX' (line 735)
    NX_121491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 13), 'NX', False)
    # Obtaining the member 'zeros' of a type (line 735)
    zeros_121492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 13), NX_121491, 'zeros')
    # Calling zeros(args, kwargs) (line 735)
    zeros_call_result_121497 = invoke(stypy.reporting.localization.Localization(__file__, 735, 13), zeros_121492, *[diff_121493, dtype_121495], **kwargs_121496)
    
    # Assigning a type to the variable 'zr' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'zr', zeros_call_result_121497)
    
    # Assigning a BinOp to a Name (line 736):
    
    # Assigning a BinOp to a Name (line 736):
    
    # Call to concatenate(...): (line 736)
    # Processing the call arguments (line 736)
    
    # Obtaining an instance of the builtin type 'tuple' (line 736)
    tuple_121500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 736)
    # Adding element type (line 736)
    # Getting the type of 'zr' (line 736)
    zr_121501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 30), 'zr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 30), tuple_121500, zr_121501)
    # Adding element type (line 736)
    # Getting the type of 'a1' (line 736)
    a1_121502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 34), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 30), tuple_121500, a1_121502)
    
    # Processing the call keyword arguments (line 736)
    kwargs_121503 = {}
    # Getting the type of 'NX' (line 736)
    NX_121498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 14), 'NX', False)
    # Obtaining the member 'concatenate' of a type (line 736)
    concatenate_121499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 14), NX_121498, 'concatenate')
    # Calling concatenate(args, kwargs) (line 736)
    concatenate_call_result_121504 = invoke(stypy.reporting.localization.Localization(__file__, 736, 14), concatenate_121499, *[tuple_121500], **kwargs_121503)
    
    # Getting the type of 'a2' (line 736)
    a2_121505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 41), 'a2')
    # Applying the binary operator '+' (line 736)
    result_add_121506 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 14), '+', concatenate_call_result_121504, a2_121505)
    
    # Assigning a type to the variable 'val' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'val', result_add_121506)
    # SSA branch for the else part of an if statement (line 734)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 738):
    
    # Assigning a Call to a Name (line 738):
    
    # Call to zeros(...): (line 738)
    # Processing the call arguments (line 738)
    
    # Call to abs(...): (line 738)
    # Processing the call arguments (line 738)
    # Getting the type of 'diff' (line 738)
    diff_121510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 26), 'diff', False)
    # Processing the call keyword arguments (line 738)
    kwargs_121511 = {}
    # Getting the type of 'abs' (line 738)
    abs_121509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 22), 'abs', False)
    # Calling abs(args, kwargs) (line 738)
    abs_call_result_121512 = invoke(stypy.reporting.localization.Localization(__file__, 738, 22), abs_121509, *[diff_121510], **kwargs_121511)
    
    # Getting the type of 'a2' (line 738)
    a2_121513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 33), 'a2', False)
    # Obtaining the member 'dtype' of a type (line 738)
    dtype_121514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 33), a2_121513, 'dtype')
    # Processing the call keyword arguments (line 738)
    kwargs_121515 = {}
    # Getting the type of 'NX' (line 738)
    NX_121507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 13), 'NX', False)
    # Obtaining the member 'zeros' of a type (line 738)
    zeros_121508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 13), NX_121507, 'zeros')
    # Calling zeros(args, kwargs) (line 738)
    zeros_call_result_121516 = invoke(stypy.reporting.localization.Localization(__file__, 738, 13), zeros_121508, *[abs_call_result_121512, dtype_121514], **kwargs_121515)
    
    # Assigning a type to the variable 'zr' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'zr', zeros_call_result_121516)
    
    # Assigning a BinOp to a Name (line 739):
    
    # Assigning a BinOp to a Name (line 739):
    # Getting the type of 'a1' (line 739)
    a1_121517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 14), 'a1')
    
    # Call to concatenate(...): (line 739)
    # Processing the call arguments (line 739)
    
    # Obtaining an instance of the builtin type 'tuple' (line 739)
    tuple_121520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 739)
    # Adding element type (line 739)
    # Getting the type of 'zr' (line 739)
    zr_121521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 35), 'zr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 35), tuple_121520, zr_121521)
    # Adding element type (line 739)
    # Getting the type of 'a2' (line 739)
    a2_121522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 39), 'a2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 35), tuple_121520, a2_121522)
    
    # Processing the call keyword arguments (line 739)
    kwargs_121523 = {}
    # Getting the type of 'NX' (line 739)
    NX_121518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 19), 'NX', False)
    # Obtaining the member 'concatenate' of a type (line 739)
    concatenate_121519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 19), NX_121518, 'concatenate')
    # Calling concatenate(args, kwargs) (line 739)
    concatenate_call_result_121524 = invoke(stypy.reporting.localization.Localization(__file__, 739, 19), concatenate_121519, *[tuple_121520], **kwargs_121523)
    
    # Applying the binary operator '+' (line 739)
    result_add_121525 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 14), '+', a1_121517, concatenate_call_result_121524)
    
    # Assigning a type to the variable 'val' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'val', result_add_121525)
    # SSA join for if statement (line 734)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 732)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'truepoly' (line 740)
    truepoly_121526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 7), 'truepoly')
    # Testing the type of an if condition (line 740)
    if_condition_121527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 4), truepoly_121526)
    # Assigning a type to the variable 'if_condition_121527' (line 740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 4), 'if_condition_121527', if_condition_121527)
    # SSA begins for if statement (line 740)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to poly1d(...): (line 741)
    # Processing the call arguments (line 741)
    # Getting the type of 'val' (line 741)
    val_121529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 21), 'val', False)
    # Processing the call keyword arguments (line 741)
    kwargs_121530 = {}
    # Getting the type of 'poly1d' (line 741)
    poly1d_121528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 14), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 741)
    poly1d_call_result_121531 = invoke(stypy.reporting.localization.Localization(__file__, 741, 14), poly1d_121528, *[val_121529], **kwargs_121530)
    
    # Assigning a type to the variable 'val' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'val', poly1d_call_result_121531)
    # SSA join for if statement (line 740)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 742)
    val_121532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 11), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'stypy_return_type', val_121532)
    
    # ################# End of 'polyadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyadd' in the type store
    # Getting the type of 'stypy_return_type' (line 684)
    stypy_return_type_121533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyadd'
    return stypy_return_type_121533

# Assigning a type to the variable 'polyadd' (line 684)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'polyadd', polyadd)

@norecursion
def polysub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polysub'
    module_type_store = module_type_store.open_function_context('polysub', 744, 0, False)
    
    # Passed parameters checking function
    polysub.stypy_localization = localization
    polysub.stypy_type_of_self = None
    polysub.stypy_type_store = module_type_store
    polysub.stypy_function_name = 'polysub'
    polysub.stypy_param_names_list = ['a1', 'a2']
    polysub.stypy_varargs_param_name = None
    polysub.stypy_kwargs_param_name = None
    polysub.stypy_call_defaults = defaults
    polysub.stypy_call_varargs = varargs
    polysub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polysub', ['a1', 'a2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polysub', localization, ['a1', 'a2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polysub(...)' code ##################

    str_121534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, (-1)), 'str', "\n    Difference (subtraction) of two polynomials.\n\n    Given two polynomials `a1` and `a2`, returns ``a1 - a2``.\n    `a1` and `a2` can be either array_like sequences of the polynomials'\n    coefficients (including coefficients equal to zero), or `poly1d` objects.\n\n    Parameters\n    ----------\n    a1, a2 : array_like or poly1d\n        Minuend and subtrahend polynomials, respectively.\n\n    Returns\n    -------\n    out : ndarray or poly1d\n        Array or `poly1d` object of the difference polynomial's coefficients.\n\n    See Also\n    --------\n    polyval, polydiv, polymul, polyadd\n\n    Examples\n    --------\n    .. math:: (2 x^2 + 10 x - 2) - (3 x^2 + 10 x -4) = (-x^2 + 2)\n\n    >>> np.polysub([2, 10, -2], [3, 10, -4])\n    array([-1,  0,  2])\n\n    ")
    
    # Assigning a BoolOp to a Name (line 774):
    
    # Assigning a BoolOp to a Name (line 774):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'a1' (line 774)
    a1_121536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 27), 'a1', False)
    # Getting the type of 'poly1d' (line 774)
    poly1d_121537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 31), 'poly1d', False)
    # Processing the call keyword arguments (line 774)
    kwargs_121538 = {}
    # Getting the type of 'isinstance' (line 774)
    isinstance_121535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 774)
    isinstance_call_result_121539 = invoke(stypy.reporting.localization.Localization(__file__, 774, 16), isinstance_121535, *[a1_121536, poly1d_121537], **kwargs_121538)
    
    
    # Call to isinstance(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'a2' (line 774)
    a2_121541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 53), 'a2', False)
    # Getting the type of 'poly1d' (line 774)
    poly1d_121542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 57), 'poly1d', False)
    # Processing the call keyword arguments (line 774)
    kwargs_121543 = {}
    # Getting the type of 'isinstance' (line 774)
    isinstance_121540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 42), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 774)
    isinstance_call_result_121544 = invoke(stypy.reporting.localization.Localization(__file__, 774, 42), isinstance_121540, *[a2_121541, poly1d_121542], **kwargs_121543)
    
    # Applying the binary operator 'or' (line 774)
    result_or_keyword_121545 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 16), 'or', isinstance_call_result_121539, isinstance_call_result_121544)
    
    # Assigning a type to the variable 'truepoly' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'truepoly', result_or_keyword_121545)
    
    # Assigning a Call to a Name (line 775):
    
    # Assigning a Call to a Name (line 775):
    
    # Call to atleast_1d(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'a1' (line 775)
    a1_121547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 20), 'a1', False)
    # Processing the call keyword arguments (line 775)
    kwargs_121548 = {}
    # Getting the type of 'atleast_1d' (line 775)
    atleast_1d_121546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 9), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 775)
    atleast_1d_call_result_121549 = invoke(stypy.reporting.localization.Localization(__file__, 775, 9), atleast_1d_121546, *[a1_121547], **kwargs_121548)
    
    # Assigning a type to the variable 'a1' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 4), 'a1', atleast_1d_call_result_121549)
    
    # Assigning a Call to a Name (line 776):
    
    # Assigning a Call to a Name (line 776):
    
    # Call to atleast_1d(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'a2' (line 776)
    a2_121551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 20), 'a2', False)
    # Processing the call keyword arguments (line 776)
    kwargs_121552 = {}
    # Getting the type of 'atleast_1d' (line 776)
    atleast_1d_121550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 9), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 776)
    atleast_1d_call_result_121553 = invoke(stypy.reporting.localization.Localization(__file__, 776, 9), atleast_1d_121550, *[a2_121551], **kwargs_121552)
    
    # Assigning a type to the variable 'a2' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'a2', atleast_1d_call_result_121553)
    
    # Assigning a BinOp to a Name (line 777):
    
    # Assigning a BinOp to a Name (line 777):
    
    # Call to len(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'a2' (line 777)
    a2_121555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 15), 'a2', False)
    # Processing the call keyword arguments (line 777)
    kwargs_121556 = {}
    # Getting the type of 'len' (line 777)
    len_121554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 11), 'len', False)
    # Calling len(args, kwargs) (line 777)
    len_call_result_121557 = invoke(stypy.reporting.localization.Localization(__file__, 777, 11), len_121554, *[a2_121555], **kwargs_121556)
    
    
    # Call to len(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'a1' (line 777)
    a1_121559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 25), 'a1', False)
    # Processing the call keyword arguments (line 777)
    kwargs_121560 = {}
    # Getting the type of 'len' (line 777)
    len_121558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 21), 'len', False)
    # Calling len(args, kwargs) (line 777)
    len_call_result_121561 = invoke(stypy.reporting.localization.Localization(__file__, 777, 21), len_121558, *[a1_121559], **kwargs_121560)
    
    # Applying the binary operator '-' (line 777)
    result_sub_121562 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 11), '-', len_call_result_121557, len_call_result_121561)
    
    # Assigning a type to the variable 'diff' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'diff', result_sub_121562)
    
    
    # Getting the type of 'diff' (line 778)
    diff_121563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 7), 'diff')
    int_121564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 15), 'int')
    # Applying the binary operator '==' (line 778)
    result_eq_121565 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 7), '==', diff_121563, int_121564)
    
    # Testing the type of an if condition (line 778)
    if_condition_121566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 778, 4), result_eq_121565)
    # Assigning a type to the variable 'if_condition_121566' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'if_condition_121566', if_condition_121566)
    # SSA begins for if statement (line 778)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 779):
    
    # Assigning a BinOp to a Name (line 779):
    # Getting the type of 'a1' (line 779)
    a1_121567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 14), 'a1')
    # Getting the type of 'a2' (line 779)
    a2_121568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 19), 'a2')
    # Applying the binary operator '-' (line 779)
    result_sub_121569 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 14), '-', a1_121567, a2_121568)
    
    # Assigning a type to the variable 'val' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'val', result_sub_121569)
    # SSA branch for the else part of an if statement (line 778)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'diff' (line 780)
    diff_121570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 9), 'diff')
    int_121571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 16), 'int')
    # Applying the binary operator '>' (line 780)
    result_gt_121572 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 9), '>', diff_121570, int_121571)
    
    # Testing the type of an if condition (line 780)
    if_condition_121573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 9), result_gt_121572)
    # Assigning a type to the variable 'if_condition_121573' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 9), 'if_condition_121573', if_condition_121573)
    # SSA begins for if statement (line 780)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 781):
    
    # Assigning a Call to a Name (line 781):
    
    # Call to zeros(...): (line 781)
    # Processing the call arguments (line 781)
    # Getting the type of 'diff' (line 781)
    diff_121576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 22), 'diff', False)
    # Getting the type of 'a1' (line 781)
    a1_121577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 28), 'a1', False)
    # Obtaining the member 'dtype' of a type (line 781)
    dtype_121578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 28), a1_121577, 'dtype')
    # Processing the call keyword arguments (line 781)
    kwargs_121579 = {}
    # Getting the type of 'NX' (line 781)
    NX_121574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 13), 'NX', False)
    # Obtaining the member 'zeros' of a type (line 781)
    zeros_121575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 13), NX_121574, 'zeros')
    # Calling zeros(args, kwargs) (line 781)
    zeros_call_result_121580 = invoke(stypy.reporting.localization.Localization(__file__, 781, 13), zeros_121575, *[diff_121576, dtype_121578], **kwargs_121579)
    
    # Assigning a type to the variable 'zr' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 8), 'zr', zeros_call_result_121580)
    
    # Assigning a BinOp to a Name (line 782):
    
    # Assigning a BinOp to a Name (line 782):
    
    # Call to concatenate(...): (line 782)
    # Processing the call arguments (line 782)
    
    # Obtaining an instance of the builtin type 'tuple' (line 782)
    tuple_121583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 782)
    # Adding element type (line 782)
    # Getting the type of 'zr' (line 782)
    zr_121584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 30), 'zr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 30), tuple_121583, zr_121584)
    # Adding element type (line 782)
    # Getting the type of 'a1' (line 782)
    a1_121585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 34), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 30), tuple_121583, a1_121585)
    
    # Processing the call keyword arguments (line 782)
    kwargs_121586 = {}
    # Getting the type of 'NX' (line 782)
    NX_121581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 14), 'NX', False)
    # Obtaining the member 'concatenate' of a type (line 782)
    concatenate_121582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 14), NX_121581, 'concatenate')
    # Calling concatenate(args, kwargs) (line 782)
    concatenate_call_result_121587 = invoke(stypy.reporting.localization.Localization(__file__, 782, 14), concatenate_121582, *[tuple_121583], **kwargs_121586)
    
    # Getting the type of 'a2' (line 782)
    a2_121588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 41), 'a2')
    # Applying the binary operator '-' (line 782)
    result_sub_121589 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 14), '-', concatenate_call_result_121587, a2_121588)
    
    # Assigning a type to the variable 'val' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'val', result_sub_121589)
    # SSA branch for the else part of an if statement (line 780)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 784):
    
    # Assigning a Call to a Name (line 784):
    
    # Call to zeros(...): (line 784)
    # Processing the call arguments (line 784)
    
    # Call to abs(...): (line 784)
    # Processing the call arguments (line 784)
    # Getting the type of 'diff' (line 784)
    diff_121593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 26), 'diff', False)
    # Processing the call keyword arguments (line 784)
    kwargs_121594 = {}
    # Getting the type of 'abs' (line 784)
    abs_121592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 22), 'abs', False)
    # Calling abs(args, kwargs) (line 784)
    abs_call_result_121595 = invoke(stypy.reporting.localization.Localization(__file__, 784, 22), abs_121592, *[diff_121593], **kwargs_121594)
    
    # Getting the type of 'a2' (line 784)
    a2_121596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 33), 'a2', False)
    # Obtaining the member 'dtype' of a type (line 784)
    dtype_121597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 33), a2_121596, 'dtype')
    # Processing the call keyword arguments (line 784)
    kwargs_121598 = {}
    # Getting the type of 'NX' (line 784)
    NX_121590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 13), 'NX', False)
    # Obtaining the member 'zeros' of a type (line 784)
    zeros_121591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 13), NX_121590, 'zeros')
    # Calling zeros(args, kwargs) (line 784)
    zeros_call_result_121599 = invoke(stypy.reporting.localization.Localization(__file__, 784, 13), zeros_121591, *[abs_call_result_121595, dtype_121597], **kwargs_121598)
    
    # Assigning a type to the variable 'zr' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'zr', zeros_call_result_121599)
    
    # Assigning a BinOp to a Name (line 785):
    
    # Assigning a BinOp to a Name (line 785):
    # Getting the type of 'a1' (line 785)
    a1_121600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 14), 'a1')
    
    # Call to concatenate(...): (line 785)
    # Processing the call arguments (line 785)
    
    # Obtaining an instance of the builtin type 'tuple' (line 785)
    tuple_121603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 785)
    # Adding element type (line 785)
    # Getting the type of 'zr' (line 785)
    zr_121604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 35), 'zr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 785, 35), tuple_121603, zr_121604)
    # Adding element type (line 785)
    # Getting the type of 'a2' (line 785)
    a2_121605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 39), 'a2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 785, 35), tuple_121603, a2_121605)
    
    # Processing the call keyword arguments (line 785)
    kwargs_121606 = {}
    # Getting the type of 'NX' (line 785)
    NX_121601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 19), 'NX', False)
    # Obtaining the member 'concatenate' of a type (line 785)
    concatenate_121602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 19), NX_121601, 'concatenate')
    # Calling concatenate(args, kwargs) (line 785)
    concatenate_call_result_121607 = invoke(stypy.reporting.localization.Localization(__file__, 785, 19), concatenate_121602, *[tuple_121603], **kwargs_121606)
    
    # Applying the binary operator '-' (line 785)
    result_sub_121608 = python_operator(stypy.reporting.localization.Localization(__file__, 785, 14), '-', a1_121600, concatenate_call_result_121607)
    
    # Assigning a type to the variable 'val' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'val', result_sub_121608)
    # SSA join for if statement (line 780)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 778)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'truepoly' (line 786)
    truepoly_121609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 7), 'truepoly')
    # Testing the type of an if condition (line 786)
    if_condition_121610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 786, 4), truepoly_121609)
    # Assigning a type to the variable 'if_condition_121610' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'if_condition_121610', if_condition_121610)
    # SSA begins for if statement (line 786)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 787):
    
    # Assigning a Call to a Name (line 787):
    
    # Call to poly1d(...): (line 787)
    # Processing the call arguments (line 787)
    # Getting the type of 'val' (line 787)
    val_121612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 21), 'val', False)
    # Processing the call keyword arguments (line 787)
    kwargs_121613 = {}
    # Getting the type of 'poly1d' (line 787)
    poly1d_121611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 14), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 787)
    poly1d_call_result_121614 = invoke(stypy.reporting.localization.Localization(__file__, 787, 14), poly1d_121611, *[val_121612], **kwargs_121613)
    
    # Assigning a type to the variable 'val' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'val', poly1d_call_result_121614)
    # SSA join for if statement (line 786)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 788)
    val_121615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'stypy_return_type', val_121615)
    
    # ################# End of 'polysub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polysub' in the type store
    # Getting the type of 'stypy_return_type' (line 744)
    stypy_return_type_121616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polysub'
    return stypy_return_type_121616

# Assigning a type to the variable 'polysub' (line 744)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 0), 'polysub', polysub)

@norecursion
def polymul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polymul'
    module_type_store = module_type_store.open_function_context('polymul', 791, 0, False)
    
    # Passed parameters checking function
    polymul.stypy_localization = localization
    polymul.stypy_type_of_self = None
    polymul.stypy_type_store = module_type_store
    polymul.stypy_function_name = 'polymul'
    polymul.stypy_param_names_list = ['a1', 'a2']
    polymul.stypy_varargs_param_name = None
    polymul.stypy_kwargs_param_name = None
    polymul.stypy_call_defaults = defaults
    polymul.stypy_call_varargs = varargs
    polymul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polymul', ['a1', 'a2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polymul', localization, ['a1', 'a2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polymul(...)' code ##################

    str_121617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, (-1)), 'str', '\n    Find the product of two polynomials.\n\n    Finds the polynomial resulting from the multiplication of the two input\n    polynomials. Each input must be either a poly1d object or a 1D sequence\n    of polynomial coefficients, from highest to lowest degree.\n\n    Parameters\n    ----------\n    a1, a2 : array_like or poly1d object\n        Input polynomials.\n\n    Returns\n    -------\n    out : ndarray or poly1d object\n        The polynomial resulting from the multiplication of the inputs. If\n        either inputs is a poly1d object, then the output is also a poly1d\n        object. Otherwise, it is a 1D array of polynomial coefficients from\n        highest to lowest degree.\n\n    See Also\n    --------\n    poly1d : A one-dimensional polynomial class.\n    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub,\n    polyval\n    convolve : Array convolution. Same output as polymul, but has parameter\n               for overlap mode.\n\n    Examples\n    --------\n    >>> np.polymul([1, 2, 3], [9, 5, 1])\n    array([ 9, 23, 38, 17,  3])\n\n    Using poly1d objects:\n\n    >>> p1 = np.poly1d([1, 2, 3])\n    >>> p2 = np.poly1d([9, 5, 1])\n    >>> print(p1)\n       2\n    1 x + 2 x + 3\n    >>> print(p2)\n       2\n    9 x + 5 x + 1\n    >>> print(np.polymul(p1, p2))\n       4      3      2\n    9 x + 23 x + 38 x + 17 x + 3\n\n    ')
    
    # Assigning a BoolOp to a Name (line 840):
    
    # Assigning a BoolOp to a Name (line 840):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 840)
    # Processing the call arguments (line 840)
    # Getting the type of 'a1' (line 840)
    a1_121619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 27), 'a1', False)
    # Getting the type of 'poly1d' (line 840)
    poly1d_121620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 31), 'poly1d', False)
    # Processing the call keyword arguments (line 840)
    kwargs_121621 = {}
    # Getting the type of 'isinstance' (line 840)
    isinstance_121618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 840)
    isinstance_call_result_121622 = invoke(stypy.reporting.localization.Localization(__file__, 840, 16), isinstance_121618, *[a1_121619, poly1d_121620], **kwargs_121621)
    
    
    # Call to isinstance(...): (line 840)
    # Processing the call arguments (line 840)
    # Getting the type of 'a2' (line 840)
    a2_121624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 53), 'a2', False)
    # Getting the type of 'poly1d' (line 840)
    poly1d_121625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 57), 'poly1d', False)
    # Processing the call keyword arguments (line 840)
    kwargs_121626 = {}
    # Getting the type of 'isinstance' (line 840)
    isinstance_121623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 42), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 840)
    isinstance_call_result_121627 = invoke(stypy.reporting.localization.Localization(__file__, 840, 42), isinstance_121623, *[a2_121624, poly1d_121625], **kwargs_121626)
    
    # Applying the binary operator 'or' (line 840)
    result_or_keyword_121628 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 16), 'or', isinstance_call_result_121622, isinstance_call_result_121627)
    
    # Assigning a type to the variable 'truepoly' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 4), 'truepoly', result_or_keyword_121628)
    
    # Assigning a Tuple to a Tuple (line 841):
    
    # Assigning a Call to a Name (line 841):
    
    # Call to poly1d(...): (line 841)
    # Processing the call arguments (line 841)
    # Getting the type of 'a1' (line 841)
    a1_121630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 20), 'a1', False)
    # Processing the call keyword arguments (line 841)
    kwargs_121631 = {}
    # Getting the type of 'poly1d' (line 841)
    poly1d_121629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 13), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 841)
    poly1d_call_result_121632 = invoke(stypy.reporting.localization.Localization(__file__, 841, 13), poly1d_121629, *[a1_121630], **kwargs_121631)
    
    # Assigning a type to the variable 'tuple_assignment_120581' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'tuple_assignment_120581', poly1d_call_result_121632)
    
    # Assigning a Call to a Name (line 841):
    
    # Call to poly1d(...): (line 841)
    # Processing the call arguments (line 841)
    # Getting the type of 'a2' (line 841)
    a2_121634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 32), 'a2', False)
    # Processing the call keyword arguments (line 841)
    kwargs_121635 = {}
    # Getting the type of 'poly1d' (line 841)
    poly1d_121633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 25), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 841)
    poly1d_call_result_121636 = invoke(stypy.reporting.localization.Localization(__file__, 841, 25), poly1d_121633, *[a2_121634], **kwargs_121635)
    
    # Assigning a type to the variable 'tuple_assignment_120582' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'tuple_assignment_120582', poly1d_call_result_121636)
    
    # Assigning a Name to a Name (line 841):
    # Getting the type of 'tuple_assignment_120581' (line 841)
    tuple_assignment_120581_121637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'tuple_assignment_120581')
    # Assigning a type to the variable 'a1' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'a1', tuple_assignment_120581_121637)
    
    # Assigning a Name to a Name (line 841):
    # Getting the type of 'tuple_assignment_120582' (line 841)
    tuple_assignment_120582_121638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'tuple_assignment_120582')
    # Assigning a type to the variable 'a2' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'a2', tuple_assignment_120582_121638)
    
    # Assigning a Call to a Name (line 842):
    
    # Assigning a Call to a Name (line 842):
    
    # Call to convolve(...): (line 842)
    # Processing the call arguments (line 842)
    # Getting the type of 'a1' (line 842)
    a1_121641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 22), 'a1', False)
    # Getting the type of 'a2' (line 842)
    a2_121642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 26), 'a2', False)
    # Processing the call keyword arguments (line 842)
    kwargs_121643 = {}
    # Getting the type of 'NX' (line 842)
    NX_121639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 10), 'NX', False)
    # Obtaining the member 'convolve' of a type (line 842)
    convolve_121640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 10), NX_121639, 'convolve')
    # Calling convolve(args, kwargs) (line 842)
    convolve_call_result_121644 = invoke(stypy.reporting.localization.Localization(__file__, 842, 10), convolve_121640, *[a1_121641, a2_121642], **kwargs_121643)
    
    # Assigning a type to the variable 'val' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'val', convolve_call_result_121644)
    
    # Getting the type of 'truepoly' (line 843)
    truepoly_121645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 7), 'truepoly')
    # Testing the type of an if condition (line 843)
    if_condition_121646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 843, 4), truepoly_121645)
    # Assigning a type to the variable 'if_condition_121646' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'if_condition_121646', if_condition_121646)
    # SSA begins for if statement (line 843)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 844):
    
    # Assigning a Call to a Name (line 844):
    
    # Call to poly1d(...): (line 844)
    # Processing the call arguments (line 844)
    # Getting the type of 'val' (line 844)
    val_121648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 21), 'val', False)
    # Processing the call keyword arguments (line 844)
    kwargs_121649 = {}
    # Getting the type of 'poly1d' (line 844)
    poly1d_121647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 14), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 844)
    poly1d_call_result_121650 = invoke(stypy.reporting.localization.Localization(__file__, 844, 14), poly1d_121647, *[val_121648], **kwargs_121649)
    
    # Assigning a type to the variable 'val' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'val', poly1d_call_result_121650)
    # SSA join for if statement (line 843)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 845)
    val_121651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 11), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'stypy_return_type', val_121651)
    
    # ################# End of 'polymul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polymul' in the type store
    # Getting the type of 'stypy_return_type' (line 791)
    stypy_return_type_121652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polymul'
    return stypy_return_type_121652

# Assigning a type to the variable 'polymul' (line 791)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 0), 'polymul', polymul)

@norecursion
def polydiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polydiv'
    module_type_store = module_type_store.open_function_context('polydiv', 847, 0, False)
    
    # Passed parameters checking function
    polydiv.stypy_localization = localization
    polydiv.stypy_type_of_self = None
    polydiv.stypy_type_store = module_type_store
    polydiv.stypy_function_name = 'polydiv'
    polydiv.stypy_param_names_list = ['u', 'v']
    polydiv.stypy_varargs_param_name = None
    polydiv.stypy_kwargs_param_name = None
    polydiv.stypy_call_defaults = defaults
    polydiv.stypy_call_varargs = varargs
    polydiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polydiv', ['u', 'v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polydiv', localization, ['u', 'v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polydiv(...)' code ##################

    str_121653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, (-1)), 'str', '\n    Returns the quotient and remainder of polynomial division.\n\n    The input arrays are the coefficients (including any coefficients\n    equal to zero) of the "numerator" (dividend) and "denominator"\n    (divisor) polynomials, respectively.\n\n    Parameters\n    ----------\n    u : array_like or poly1d\n        Dividend polynomial\'s coefficients.\n\n    v : array_like or poly1d\n        Divisor polynomial\'s coefficients.\n\n    Returns\n    -------\n    q : ndarray\n        Coefficients, including those equal to zero, of the quotient.\n    r : ndarray\n        Coefficients, including those equal to zero, of the remainder.\n\n    See Also\n    --------\n    poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub,\n    polyval\n\n    Notes\n    -----\n    Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need\n    not equal `v.ndim`. In other words, all four possible combinations -\n    ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,\n    ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.\n\n    Examples\n    --------\n    .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25\n\n    >>> x = np.array([3.0, 5.0, 2.0])\n    >>> y = np.array([2.0, 1.0])\n    >>> np.polydiv(x, y)\n    (array([ 1.5 ,  1.75]), array([ 0.25]))\n\n    ')
    
    # Assigning a BoolOp to a Name (line 892):
    
    # Assigning a BoolOp to a Name (line 892):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'u' (line 892)
    u_121655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 27), 'u', False)
    # Getting the type of 'poly1d' (line 892)
    poly1d_121656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 30), 'poly1d', False)
    # Processing the call keyword arguments (line 892)
    kwargs_121657 = {}
    # Getting the type of 'isinstance' (line 892)
    isinstance_121654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 892)
    isinstance_call_result_121658 = invoke(stypy.reporting.localization.Localization(__file__, 892, 16), isinstance_121654, *[u_121655, poly1d_121656], **kwargs_121657)
    
    
    # Call to isinstance(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'u' (line 892)
    u_121660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 52), 'u', False)
    # Getting the type of 'poly1d' (line 892)
    poly1d_121661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 55), 'poly1d', False)
    # Processing the call keyword arguments (line 892)
    kwargs_121662 = {}
    # Getting the type of 'isinstance' (line 892)
    isinstance_121659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 41), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 892)
    isinstance_call_result_121663 = invoke(stypy.reporting.localization.Localization(__file__, 892, 41), isinstance_121659, *[u_121660, poly1d_121661], **kwargs_121662)
    
    # Applying the binary operator 'or' (line 892)
    result_or_keyword_121664 = python_operator(stypy.reporting.localization.Localization(__file__, 892, 16), 'or', isinstance_call_result_121658, isinstance_call_result_121663)
    
    # Assigning a type to the variable 'truepoly' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 4), 'truepoly', result_or_keyword_121664)
    
    # Assigning a BinOp to a Name (line 893):
    
    # Assigning a BinOp to a Name (line 893):
    
    # Call to atleast_1d(...): (line 893)
    # Processing the call arguments (line 893)
    # Getting the type of 'u' (line 893)
    u_121666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 19), 'u', False)
    # Processing the call keyword arguments (line 893)
    kwargs_121667 = {}
    # Getting the type of 'atleast_1d' (line 893)
    atleast_1d_121665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 893)
    atleast_1d_call_result_121668 = invoke(stypy.reporting.localization.Localization(__file__, 893, 8), atleast_1d_121665, *[u_121666], **kwargs_121667)
    
    float_121669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 24), 'float')
    # Applying the binary operator '+' (line 893)
    result_add_121670 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 8), '+', atleast_1d_call_result_121668, float_121669)
    
    # Assigning a type to the variable 'u' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 4), 'u', result_add_121670)
    
    # Assigning a BinOp to a Name (line 894):
    
    # Assigning a BinOp to a Name (line 894):
    
    # Call to atleast_1d(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'v' (line 894)
    v_121672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 19), 'v', False)
    # Processing the call keyword arguments (line 894)
    kwargs_121673 = {}
    # Getting the type of 'atleast_1d' (line 894)
    atleast_1d_121671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 894)
    atleast_1d_call_result_121674 = invoke(stypy.reporting.localization.Localization(__file__, 894, 8), atleast_1d_121671, *[v_121672], **kwargs_121673)
    
    float_121675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 24), 'float')
    # Applying the binary operator '+' (line 894)
    result_add_121676 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 8), '+', atleast_1d_call_result_121674, float_121675)
    
    # Assigning a type to the variable 'v' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'v', result_add_121676)
    
    # Assigning a BinOp to a Name (line 896):
    
    # Assigning a BinOp to a Name (line 896):
    
    # Obtaining the type of the subscript
    int_121677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 10), 'int')
    # Getting the type of 'u' (line 896)
    u_121678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 8), 'u')
    # Obtaining the member '__getitem__' of a type (line 896)
    getitem___121679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 8), u_121678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 896)
    subscript_call_result_121680 = invoke(stypy.reporting.localization.Localization(__file__, 896, 8), getitem___121679, int_121677)
    
    
    # Obtaining the type of the subscript
    int_121681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 17), 'int')
    # Getting the type of 'v' (line 896)
    v_121682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 15), 'v')
    # Obtaining the member '__getitem__' of a type (line 896)
    getitem___121683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 15), v_121682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 896)
    subscript_call_result_121684 = invoke(stypy.reporting.localization.Localization(__file__, 896, 15), getitem___121683, int_121681)
    
    # Applying the binary operator '+' (line 896)
    result_add_121685 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 8), '+', subscript_call_result_121680, subscript_call_result_121684)
    
    # Assigning a type to the variable 'w' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'w', result_add_121685)
    
    # Assigning a BinOp to a Name (line 897):
    
    # Assigning a BinOp to a Name (line 897):
    
    # Call to len(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'u' (line 897)
    u_121687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 12), 'u', False)
    # Processing the call keyword arguments (line 897)
    kwargs_121688 = {}
    # Getting the type of 'len' (line 897)
    len_121686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'len', False)
    # Calling len(args, kwargs) (line 897)
    len_call_result_121689 = invoke(stypy.reporting.localization.Localization(__file__, 897, 8), len_121686, *[u_121687], **kwargs_121688)
    
    int_121690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 17), 'int')
    # Applying the binary operator '-' (line 897)
    result_sub_121691 = python_operator(stypy.reporting.localization.Localization(__file__, 897, 8), '-', len_call_result_121689, int_121690)
    
    # Assigning a type to the variable 'm' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'm', result_sub_121691)
    
    # Assigning a BinOp to a Name (line 898):
    
    # Assigning a BinOp to a Name (line 898):
    
    # Call to len(...): (line 898)
    # Processing the call arguments (line 898)
    # Getting the type of 'v' (line 898)
    v_121693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'v', False)
    # Processing the call keyword arguments (line 898)
    kwargs_121694 = {}
    # Getting the type of 'len' (line 898)
    len_121692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 8), 'len', False)
    # Calling len(args, kwargs) (line 898)
    len_call_result_121695 = invoke(stypy.reporting.localization.Localization(__file__, 898, 8), len_121692, *[v_121693], **kwargs_121694)
    
    int_121696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 17), 'int')
    # Applying the binary operator '-' (line 898)
    result_sub_121697 = python_operator(stypy.reporting.localization.Localization(__file__, 898, 8), '-', len_call_result_121695, int_121696)
    
    # Assigning a type to the variable 'n' (line 898)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 4), 'n', result_sub_121697)
    
    # Assigning a BinOp to a Name (line 899):
    
    # Assigning a BinOp to a Name (line 899):
    float_121698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'float')
    
    # Obtaining the type of the subscript
    int_121699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 19), 'int')
    # Getting the type of 'v' (line 899)
    v_121700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 17), 'v')
    # Obtaining the member '__getitem__' of a type (line 899)
    getitem___121701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 17), v_121700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 899)
    subscript_call_result_121702 = invoke(stypy.reporting.localization.Localization(__file__, 899, 17), getitem___121701, int_121699)
    
    # Applying the binary operator 'div' (line 899)
    result_div_121703 = python_operator(stypy.reporting.localization.Localization(__file__, 899, 12), 'div', float_121698, subscript_call_result_121702)
    
    # Assigning a type to the variable 'scale' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 4), 'scale', result_div_121703)
    
    # Assigning a Call to a Name (line 900):
    
    # Assigning a Call to a Name (line 900):
    
    # Call to zeros(...): (line 900)
    # Processing the call arguments (line 900)
    
    # Obtaining an instance of the builtin type 'tuple' (line 900)
    tuple_121706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 900)
    # Adding element type (line 900)
    
    # Call to max(...): (line 900)
    # Processing the call arguments (line 900)
    # Getting the type of 'm' (line 900)
    m_121708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 22), 'm', False)
    # Getting the type of 'n' (line 900)
    n_121709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 26), 'n', False)
    # Applying the binary operator '-' (line 900)
    result_sub_121710 = python_operator(stypy.reporting.localization.Localization(__file__, 900, 22), '-', m_121708, n_121709)
    
    int_121711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 30), 'int')
    # Applying the binary operator '+' (line 900)
    result_add_121712 = python_operator(stypy.reporting.localization.Localization(__file__, 900, 28), '+', result_sub_121710, int_121711)
    
    int_121713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 33), 'int')
    # Processing the call keyword arguments (line 900)
    kwargs_121714 = {}
    # Getting the type of 'max' (line 900)
    max_121707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 18), 'max', False)
    # Calling max(args, kwargs) (line 900)
    max_call_result_121715 = invoke(stypy.reporting.localization.Localization(__file__, 900, 18), max_121707, *[result_add_121712, int_121713], **kwargs_121714)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 900, 18), tuple_121706, max_call_result_121715)
    
    # Getting the type of 'w' (line 900)
    w_121716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 39), 'w', False)
    # Obtaining the member 'dtype' of a type (line 900)
    dtype_121717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 39), w_121716, 'dtype')
    # Processing the call keyword arguments (line 900)
    kwargs_121718 = {}
    # Getting the type of 'NX' (line 900)
    NX_121704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 8), 'NX', False)
    # Obtaining the member 'zeros' of a type (line 900)
    zeros_121705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 8), NX_121704, 'zeros')
    # Calling zeros(args, kwargs) (line 900)
    zeros_call_result_121719 = invoke(stypy.reporting.localization.Localization(__file__, 900, 8), zeros_121705, *[tuple_121706, dtype_121717], **kwargs_121718)
    
    # Assigning a type to the variable 'q' (line 900)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 4), 'q', zeros_call_result_121719)
    
    # Assigning a Call to a Name (line 901):
    
    # Assigning a Call to a Name (line 901):
    
    # Call to copy(...): (line 901)
    # Processing the call keyword arguments (line 901)
    kwargs_121722 = {}
    # Getting the type of 'u' (line 901)
    u_121720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'u', False)
    # Obtaining the member 'copy' of a type (line 901)
    copy_121721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 8), u_121720, 'copy')
    # Calling copy(args, kwargs) (line 901)
    copy_call_result_121723 = invoke(stypy.reporting.localization.Localization(__file__, 901, 8), copy_121721, *[], **kwargs_121722)
    
    # Assigning a type to the variable 'r' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 4), 'r', copy_call_result_121723)
    
    
    # Call to range(...): (line 902)
    # Processing the call arguments (line 902)
    int_121725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 19), 'int')
    # Getting the type of 'm' (line 902)
    m_121726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 22), 'm', False)
    # Getting the type of 'n' (line 902)
    n_121727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 24), 'n', False)
    # Applying the binary operator '-' (line 902)
    result_sub_121728 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 22), '-', m_121726, n_121727)
    
    int_121729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 26), 'int')
    # Applying the binary operator '+' (line 902)
    result_add_121730 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 25), '+', result_sub_121728, int_121729)
    
    # Processing the call keyword arguments (line 902)
    kwargs_121731 = {}
    # Getting the type of 'range' (line 902)
    range_121724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 13), 'range', False)
    # Calling range(args, kwargs) (line 902)
    range_call_result_121732 = invoke(stypy.reporting.localization.Localization(__file__, 902, 13), range_121724, *[int_121725, result_add_121730], **kwargs_121731)
    
    # Testing the type of a for loop iterable (line 902)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 902, 4), range_call_result_121732)
    # Getting the type of the for loop variable (line 902)
    for_loop_var_121733 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 902, 4), range_call_result_121732)
    # Assigning a type to the variable 'k' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 4), 'k', for_loop_var_121733)
    # SSA begins for a for statement (line 902)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 903):
    
    # Assigning a BinOp to a Name (line 903):
    # Getting the type of 'scale' (line 903)
    scale_121734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 12), 'scale')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 903)
    k_121735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 22), 'k')
    # Getting the type of 'r' (line 903)
    r_121736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 20), 'r')
    # Obtaining the member '__getitem__' of a type (line 903)
    getitem___121737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 20), r_121736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 903)
    subscript_call_result_121738 = invoke(stypy.reporting.localization.Localization(__file__, 903, 20), getitem___121737, k_121735)
    
    # Applying the binary operator '*' (line 903)
    result_mul_121739 = python_operator(stypy.reporting.localization.Localization(__file__, 903, 12), '*', scale_121734, subscript_call_result_121738)
    
    # Assigning a type to the variable 'd' (line 903)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'd', result_mul_121739)
    
    # Assigning a Name to a Subscript (line 904):
    
    # Assigning a Name to a Subscript (line 904):
    # Getting the type of 'd' (line 904)
    d_121740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 15), 'd')
    # Getting the type of 'q' (line 904)
    q_121741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 8), 'q')
    # Getting the type of 'k' (line 904)
    k_121742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 10), 'k')
    # Storing an element on a container (line 904)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 904, 8), q_121741, (k_121742, d_121740))
    
    # Getting the type of 'r' (line 905)
    r_121743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'r')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 905)
    k_121744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 10), 'k')
    # Getting the type of 'k' (line 905)
    k_121745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 12), 'k')
    # Getting the type of 'n' (line 905)
    n_121746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 14), 'n')
    # Applying the binary operator '+' (line 905)
    result_add_121747 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 12), '+', k_121745, n_121746)
    
    int_121748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 16), 'int')
    # Applying the binary operator '+' (line 905)
    result_add_121749 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 15), '+', result_add_121747, int_121748)
    
    slice_121750 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 905, 8), k_121744, result_add_121749, None)
    # Getting the type of 'r' (line 905)
    r_121751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'r')
    # Obtaining the member '__getitem__' of a type (line 905)
    getitem___121752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 8), r_121751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 905)
    subscript_call_result_121753 = invoke(stypy.reporting.localization.Localization(__file__, 905, 8), getitem___121752, slice_121750)
    
    # Getting the type of 'd' (line 905)
    d_121754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 22), 'd')
    # Getting the type of 'v' (line 905)
    v_121755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 24), 'v')
    # Applying the binary operator '*' (line 905)
    result_mul_121756 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 22), '*', d_121754, v_121755)
    
    # Applying the binary operator '-=' (line 905)
    result_isub_121757 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 8), '-=', subscript_call_result_121753, result_mul_121756)
    # Getting the type of 'r' (line 905)
    r_121758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'r')
    # Getting the type of 'k' (line 905)
    k_121759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 10), 'k')
    # Getting the type of 'k' (line 905)
    k_121760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 12), 'k')
    # Getting the type of 'n' (line 905)
    n_121761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 14), 'n')
    # Applying the binary operator '+' (line 905)
    result_add_121762 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 12), '+', k_121760, n_121761)
    
    int_121763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 16), 'int')
    # Applying the binary operator '+' (line 905)
    result_add_121764 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 15), '+', result_add_121762, int_121763)
    
    slice_121765 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 905, 8), k_121759, result_add_121764, None)
    # Storing an element on a container (line 905)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 8), r_121758, (slice_121765, result_isub_121757))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to allclose(...): (line 906)
    # Processing the call arguments (line 906)
    
    # Obtaining the type of the subscript
    int_121768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 24), 'int')
    # Getting the type of 'r' (line 906)
    r_121769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 22), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 906)
    getitem___121770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 22), r_121769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 906)
    subscript_call_result_121771 = invoke(stypy.reporting.localization.Localization(__file__, 906, 22), getitem___121770, int_121768)
    
    int_121772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 28), 'int')
    # Processing the call keyword arguments (line 906)
    float_121773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 36), 'float')
    keyword_121774 = float_121773
    kwargs_121775 = {'rtol': keyword_121774}
    # Getting the type of 'NX' (line 906)
    NX_121766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 10), 'NX', False)
    # Obtaining the member 'allclose' of a type (line 906)
    allclose_121767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 10), NX_121766, 'allclose')
    # Calling allclose(args, kwargs) (line 906)
    allclose_call_result_121776 = invoke(stypy.reporting.localization.Localization(__file__, 906, 10), allclose_121767, *[subscript_call_result_121771, int_121772], **kwargs_121775)
    
    
    
    # Obtaining the type of the subscript
    int_121777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 56), 'int')
    # Getting the type of 'r' (line 906)
    r_121778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 48), 'r')
    # Obtaining the member 'shape' of a type (line 906)
    shape_121779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 48), r_121778, 'shape')
    # Obtaining the member '__getitem__' of a type (line 906)
    getitem___121780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 48), shape_121779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 906)
    subscript_call_result_121781 = invoke(stypy.reporting.localization.Localization(__file__, 906, 48), getitem___121780, int_121777)
    
    int_121782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 62), 'int')
    # Applying the binary operator '>' (line 906)
    result_gt_121783 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 48), '>', subscript_call_result_121781, int_121782)
    
    # Applying the binary operator 'and' (line 906)
    result_and_keyword_121784 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 10), 'and', allclose_call_result_121776, result_gt_121783)
    
    # Testing the type of an if condition (line 906)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 906, 4), result_and_keyword_121784)
    # SSA begins for while statement (line 906)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 907):
    
    # Assigning a Subscript to a Name (line 907):
    
    # Obtaining the type of the subscript
    int_121785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 14), 'int')
    slice_121786 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 907, 12), int_121785, None, None)
    # Getting the type of 'r' (line 907)
    r_121787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 12), 'r')
    # Obtaining the member '__getitem__' of a type (line 907)
    getitem___121788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 12), r_121787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 907)
    subscript_call_result_121789 = invoke(stypy.reporting.localization.Localization(__file__, 907, 12), getitem___121788, slice_121786)
    
    # Assigning a type to the variable 'r' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 8), 'r', subscript_call_result_121789)
    # SSA join for while statement (line 906)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'truepoly' (line 908)
    truepoly_121790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 7), 'truepoly')
    # Testing the type of an if condition (line 908)
    if_condition_121791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 908, 4), truepoly_121790)
    # Assigning a type to the variable 'if_condition_121791' (line 908)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 4), 'if_condition_121791', if_condition_121791)
    # SSA begins for if statement (line 908)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 909)
    tuple_121792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 909)
    # Adding element type (line 909)
    
    # Call to poly1d(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'q' (line 909)
    q_121794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 22), 'q', False)
    # Processing the call keyword arguments (line 909)
    kwargs_121795 = {}
    # Getting the type of 'poly1d' (line 909)
    poly1d_121793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 15), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 909)
    poly1d_call_result_121796 = invoke(stypy.reporting.localization.Localization(__file__, 909, 15), poly1d_121793, *[q_121794], **kwargs_121795)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 909, 15), tuple_121792, poly1d_call_result_121796)
    # Adding element type (line 909)
    
    # Call to poly1d(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'r' (line 909)
    r_121798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 33), 'r', False)
    # Processing the call keyword arguments (line 909)
    kwargs_121799 = {}
    # Getting the type of 'poly1d' (line 909)
    poly1d_121797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 26), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 909)
    poly1d_call_result_121800 = invoke(stypy.reporting.localization.Localization(__file__, 909, 26), poly1d_121797, *[r_121798], **kwargs_121799)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 909, 15), tuple_121792, poly1d_call_result_121800)
    
    # Assigning a type to the variable 'stypy_return_type' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'stypy_return_type', tuple_121792)
    # SSA join for if statement (line 908)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 910)
    tuple_121801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 910)
    # Adding element type (line 910)
    # Getting the type of 'q' (line 910)
    q_121802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 11), 'q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 910, 11), tuple_121801, q_121802)
    # Adding element type (line 910)
    # Getting the type of 'r' (line 910)
    r_121803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 14), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 910, 11), tuple_121801, r_121803)
    
    # Assigning a type to the variable 'stypy_return_type' (line 910)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 4), 'stypy_return_type', tuple_121801)
    
    # ################# End of 'polydiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polydiv' in the type store
    # Getting the type of 'stypy_return_type' (line 847)
    stypy_return_type_121804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121804)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polydiv'
    return stypy_return_type_121804

# Assigning a type to the variable 'polydiv' (line 847)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 0), 'polydiv', polydiv)

# Assigning a Call to a Name (line 912):

# Assigning a Call to a Name (line 912):

# Call to compile(...): (line 912)
# Processing the call arguments (line 912)
str_121807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 23), 'str', '[*][*]([0-9]*)')
# Processing the call keyword arguments (line 912)
kwargs_121808 = {}
# Getting the type of 're' (line 912)
re_121805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 're', False)
# Obtaining the member 'compile' of a type (line 912)
compile_121806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 12), re_121805, 'compile')
# Calling compile(args, kwargs) (line 912)
compile_call_result_121809 = invoke(stypy.reporting.localization.Localization(__file__, 912, 12), compile_121806, *[str_121807], **kwargs_121808)

# Assigning a type to the variable '_poly_mat' (line 912)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 0), '_poly_mat', compile_call_result_121809)

@norecursion
def _raise_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_121810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 28), 'int')
    defaults = [int_121810]
    # Create a new context for function '_raise_power'
    module_type_store = module_type_store.open_function_context('_raise_power', 913, 0, False)
    
    # Passed parameters checking function
    _raise_power.stypy_localization = localization
    _raise_power.stypy_type_of_self = None
    _raise_power.stypy_type_store = module_type_store
    _raise_power.stypy_function_name = '_raise_power'
    _raise_power.stypy_param_names_list = ['astr', 'wrap']
    _raise_power.stypy_varargs_param_name = None
    _raise_power.stypy_kwargs_param_name = None
    _raise_power.stypy_call_defaults = defaults
    _raise_power.stypy_call_varargs = varargs
    _raise_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raise_power', ['astr', 'wrap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raise_power', localization, ['astr', 'wrap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raise_power(...)' code ##################

    
    # Assigning a Num to a Name (line 914):
    
    # Assigning a Num to a Name (line 914):
    int_121811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 8), 'int')
    # Assigning a type to the variable 'n' (line 914)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'n', int_121811)
    
    # Assigning a Str to a Name (line 915):
    
    # Assigning a Str to a Name (line 915):
    str_121812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 12), 'str', '')
    # Assigning a type to the variable 'line1' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'line1', str_121812)
    
    # Assigning a Str to a Name (line 916):
    
    # Assigning a Str to a Name (line 916):
    str_121813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 12), 'str', '')
    # Assigning a type to the variable 'line2' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'line2', str_121813)
    
    # Assigning a Str to a Name (line 917):
    
    # Assigning a Str to a Name (line 917):
    str_121814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 13), 'str', ' ')
    # Assigning a type to the variable 'output' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 4), 'output', str_121814)
    
    # Getting the type of 'True' (line 918)
    True_121815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 10), 'True')
    # Testing the type of an if condition (line 918)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 918, 4), True_121815)
    # SSA begins for while statement (line 918)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 919):
    
    # Assigning a Call to a Name (line 919):
    
    # Call to search(...): (line 919)
    # Processing the call arguments (line 919)
    # Getting the type of 'astr' (line 919)
    astr_121818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 31), 'astr', False)
    # Getting the type of 'n' (line 919)
    n_121819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 37), 'n', False)
    # Processing the call keyword arguments (line 919)
    kwargs_121820 = {}
    # Getting the type of '_poly_mat' (line 919)
    _poly_mat_121816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 14), '_poly_mat', False)
    # Obtaining the member 'search' of a type (line 919)
    search_121817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 14), _poly_mat_121816, 'search')
    # Calling search(args, kwargs) (line 919)
    search_call_result_121821 = invoke(stypy.reporting.localization.Localization(__file__, 919, 14), search_121817, *[astr_121818, n_121819], **kwargs_121820)
    
    # Assigning a type to the variable 'mat' (line 919)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'mat', search_call_result_121821)
    
    # Type idiom detected: calculating its left and rigth part (line 920)
    # Getting the type of 'mat' (line 920)
    mat_121822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 11), 'mat')
    # Getting the type of 'None' (line 920)
    None_121823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 18), 'None')
    
    (may_be_121824, more_types_in_union_121825) = may_be_none(mat_121822, None_121823)

    if may_be_121824:

        if more_types_in_union_121825:
            # Runtime conditional SSA (line 920)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_121825:
            # SSA join for if statement (line 920)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 922):
    
    # Assigning a Call to a Name (line 922):
    
    # Call to span(...): (line 922)
    # Processing the call keyword arguments (line 922)
    kwargs_121828 = {}
    # Getting the type of 'mat' (line 922)
    mat_121826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 15), 'mat', False)
    # Obtaining the member 'span' of a type (line 922)
    span_121827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 15), mat_121826, 'span')
    # Calling span(args, kwargs) (line 922)
    span_call_result_121829 = invoke(stypy.reporting.localization.Localization(__file__, 922, 15), span_121827, *[], **kwargs_121828)
    
    # Assigning a type to the variable 'span' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'span', span_call_result_121829)
    
    # Assigning a Subscript to a Name (line 923):
    
    # Assigning a Subscript to a Name (line 923):
    
    # Obtaining the type of the subscript
    int_121830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 29), 'int')
    
    # Call to groups(...): (line 923)
    # Processing the call keyword arguments (line 923)
    kwargs_121833 = {}
    # Getting the type of 'mat' (line 923)
    mat_121831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 16), 'mat', False)
    # Obtaining the member 'groups' of a type (line 923)
    groups_121832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 16), mat_121831, 'groups')
    # Calling groups(args, kwargs) (line 923)
    groups_call_result_121834 = invoke(stypy.reporting.localization.Localization(__file__, 923, 16), groups_121832, *[], **kwargs_121833)
    
    # Obtaining the member '__getitem__' of a type (line 923)
    getitem___121835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 16), groups_call_result_121834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 923)
    subscript_call_result_121836 = invoke(stypy.reporting.localization.Localization(__file__, 923, 16), getitem___121835, int_121830)
    
    # Assigning a type to the variable 'power' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'power', subscript_call_result_121836)
    
    # Assigning a Subscript to a Name (line 924):
    
    # Assigning a Subscript to a Name (line 924):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 924)
    n_121837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 23), 'n')
    
    # Obtaining the type of the subscript
    int_121838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 30), 'int')
    # Getting the type of 'span' (line 924)
    span_121839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 25), 'span')
    # Obtaining the member '__getitem__' of a type (line 924)
    getitem___121840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 25), span_121839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 924)
    subscript_call_result_121841 = invoke(stypy.reporting.localization.Localization(__file__, 924, 25), getitem___121840, int_121838)
    
    slice_121842 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 924, 18), n_121837, subscript_call_result_121841, None)
    # Getting the type of 'astr' (line 924)
    astr_121843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 18), 'astr')
    # Obtaining the member '__getitem__' of a type (line 924)
    getitem___121844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 18), astr_121843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 924)
    subscript_call_result_121845 = invoke(stypy.reporting.localization.Localization(__file__, 924, 18), getitem___121844, slice_121842)
    
    # Assigning a type to the variable 'partstr' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'partstr', subscript_call_result_121845)
    
    # Assigning a Subscript to a Name (line 925):
    
    # Assigning a Subscript to a Name (line 925):
    
    # Obtaining the type of the subscript
    int_121846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 17), 'int')
    # Getting the type of 'span' (line 925)
    span_121847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'span')
    # Obtaining the member '__getitem__' of a type (line 925)
    getitem___121848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 12), span_121847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 925)
    subscript_call_result_121849 = invoke(stypy.reporting.localization.Localization(__file__, 925, 12), getitem___121848, int_121846)
    
    # Assigning a type to the variable 'n' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'n', subscript_call_result_121849)
    
    # Assigning a BinOp to a Name (line 926):
    
    # Assigning a BinOp to a Name (line 926):
    # Getting the type of 'partstr' (line 926)
    partstr_121850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 17), 'partstr')
    str_121851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 27), 'str', ' ')
    
    # Call to len(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'power' (line 926)
    power_121853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 36), 'power', False)
    # Processing the call keyword arguments (line 926)
    kwargs_121854 = {}
    # Getting the type of 'len' (line 926)
    len_121852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 32), 'len', False)
    # Calling len(args, kwargs) (line 926)
    len_call_result_121855 = invoke(stypy.reporting.localization.Localization(__file__, 926, 32), len_121852, *[power_121853], **kwargs_121854)
    
    int_121856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 43), 'int')
    # Applying the binary operator '-' (line 926)
    result_sub_121857 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 32), '-', len_call_result_121855, int_121856)
    
    # Applying the binary operator '*' (line 926)
    result_mul_121858 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 27), '*', str_121851, result_sub_121857)
    
    # Applying the binary operator '+' (line 926)
    result_add_121859 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 17), '+', partstr_121850, result_mul_121858)
    
    # Assigning a type to the variable 'toadd2' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'toadd2', result_add_121859)
    
    # Assigning a BinOp to a Name (line 927):
    
    # Assigning a BinOp to a Name (line 927):
    str_121860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 17), 'str', ' ')
    
    # Call to len(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'partstr' (line 927)
    partstr_121862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 26), 'partstr', False)
    # Processing the call keyword arguments (line 927)
    kwargs_121863 = {}
    # Getting the type of 'len' (line 927)
    len_121861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 22), 'len', False)
    # Calling len(args, kwargs) (line 927)
    len_call_result_121864 = invoke(stypy.reporting.localization.Localization(__file__, 927, 22), len_121861, *[partstr_121862], **kwargs_121863)
    
    int_121865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 35), 'int')
    # Applying the binary operator '-' (line 927)
    result_sub_121866 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 22), '-', len_call_result_121864, int_121865)
    
    # Applying the binary operator '*' (line 927)
    result_mul_121867 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 17), '*', str_121860, result_sub_121866)
    
    # Getting the type of 'power' (line 927)
    power_121868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 40), 'power')
    # Applying the binary operator '+' (line 927)
    result_add_121869 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 17), '+', result_mul_121867, power_121868)
    
    # Assigning a type to the variable 'toadd1' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'toadd1', result_add_121869)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 'line2' (line 928)
    line2_121871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 17), 'line2', False)
    # Processing the call keyword arguments (line 928)
    kwargs_121872 = {}
    # Getting the type of 'len' (line 928)
    len_121870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 13), 'len', False)
    # Calling len(args, kwargs) (line 928)
    len_call_result_121873 = invoke(stypy.reporting.localization.Localization(__file__, 928, 13), len_121870, *[line2_121871], **kwargs_121872)
    
    
    # Call to len(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 'toadd2' (line 928)
    toadd2_121875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 30), 'toadd2', False)
    # Processing the call keyword arguments (line 928)
    kwargs_121876 = {}
    # Getting the type of 'len' (line 928)
    len_121874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 26), 'len', False)
    # Calling len(args, kwargs) (line 928)
    len_call_result_121877 = invoke(stypy.reporting.localization.Localization(__file__, 928, 26), len_121874, *[toadd2_121875], **kwargs_121876)
    
    # Applying the binary operator '+' (line 928)
    result_add_121878 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 13), '+', len_call_result_121873, len_call_result_121877)
    
    # Getting the type of 'wrap' (line 928)
    wrap_121879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 40), 'wrap')
    # Applying the binary operator '>' (line 928)
    result_gt_121880 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 13), '>', result_add_121878, wrap_121879)
    
    
    
    # Call to len(...): (line 929)
    # Processing the call arguments (line 929)
    # Getting the type of 'line1' (line 929)
    line1_121882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 21), 'line1', False)
    # Processing the call keyword arguments (line 929)
    kwargs_121883 = {}
    # Getting the type of 'len' (line 929)
    len_121881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 17), 'len', False)
    # Calling len(args, kwargs) (line 929)
    len_call_result_121884 = invoke(stypy.reporting.localization.Localization(__file__, 929, 17), len_121881, *[line1_121882], **kwargs_121883)
    
    
    # Call to len(...): (line 929)
    # Processing the call arguments (line 929)
    # Getting the type of 'toadd1' (line 929)
    toadd1_121886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 34), 'toadd1', False)
    # Processing the call keyword arguments (line 929)
    kwargs_121887 = {}
    # Getting the type of 'len' (line 929)
    len_121885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 30), 'len', False)
    # Calling len(args, kwargs) (line 929)
    len_call_result_121888 = invoke(stypy.reporting.localization.Localization(__file__, 929, 30), len_121885, *[toadd1_121886], **kwargs_121887)
    
    # Applying the binary operator '+' (line 929)
    result_add_121889 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 17), '+', len_call_result_121884, len_call_result_121888)
    
    # Getting the type of 'wrap' (line 929)
    wrap_121890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 44), 'wrap')
    # Applying the binary operator '>' (line 929)
    result_gt_121891 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 17), '>', result_add_121889, wrap_121890)
    
    # Applying the binary operator 'or' (line 928)
    result_or_keyword_121892 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 12), 'or', result_gt_121880, result_gt_121891)
    
    # Testing the type of an if condition (line 928)
    if_condition_121893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 928, 8), result_or_keyword_121892)
    # Assigning a type to the variable 'if_condition_121893' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), 'if_condition_121893', if_condition_121893)
    # SSA begins for if statement (line 928)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'output' (line 930)
    output_121894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'output')
    # Getting the type of 'line1' (line 930)
    line1_121895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 22), 'line1')
    str_121896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 30), 'str', '\n')
    # Applying the binary operator '+' (line 930)
    result_add_121897 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 22), '+', line1_121895, str_121896)
    
    # Getting the type of 'line2' (line 930)
    line2_121898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 37), 'line2')
    # Applying the binary operator '+' (line 930)
    result_add_121899 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 35), '+', result_add_121897, line2_121898)
    
    str_121900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 45), 'str', '\n ')
    # Applying the binary operator '+' (line 930)
    result_add_121901 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 43), '+', result_add_121899, str_121900)
    
    # Applying the binary operator '+=' (line 930)
    result_iadd_121902 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 12), '+=', output_121894, result_add_121901)
    # Assigning a type to the variable 'output' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'output', result_iadd_121902)
    
    
    # Assigning a Name to a Name (line 931):
    
    # Assigning a Name to a Name (line 931):
    # Getting the type of 'toadd1' (line 931)
    toadd1_121903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 20), 'toadd1')
    # Assigning a type to the variable 'line1' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'line1', toadd1_121903)
    
    # Assigning a Name to a Name (line 932):
    
    # Assigning a Name to a Name (line 932):
    # Getting the type of 'toadd2' (line 932)
    toadd2_121904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 20), 'toadd2')
    # Assigning a type to the variable 'line2' (line 932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 12), 'line2', toadd2_121904)
    # SSA branch for the else part of an if statement (line 928)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'line2' (line 934)
    line2_121905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 12), 'line2')
    # Getting the type of 'partstr' (line 934)
    partstr_121906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 21), 'partstr')
    str_121907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 31), 'str', ' ')
    
    # Call to len(...): (line 934)
    # Processing the call arguments (line 934)
    # Getting the type of 'power' (line 934)
    power_121909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 40), 'power', False)
    # Processing the call keyword arguments (line 934)
    kwargs_121910 = {}
    # Getting the type of 'len' (line 934)
    len_121908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 36), 'len', False)
    # Calling len(args, kwargs) (line 934)
    len_call_result_121911 = invoke(stypy.reporting.localization.Localization(__file__, 934, 36), len_121908, *[power_121909], **kwargs_121910)
    
    int_121912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 47), 'int')
    # Applying the binary operator '-' (line 934)
    result_sub_121913 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 36), '-', len_call_result_121911, int_121912)
    
    # Applying the binary operator '*' (line 934)
    result_mul_121914 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 31), '*', str_121907, result_sub_121913)
    
    # Applying the binary operator '+' (line 934)
    result_add_121915 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 21), '+', partstr_121906, result_mul_121914)
    
    # Applying the binary operator '+=' (line 934)
    result_iadd_121916 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 12), '+=', line2_121905, result_add_121915)
    # Assigning a type to the variable 'line2' (line 934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 12), 'line2', result_iadd_121916)
    
    
    # Getting the type of 'line1' (line 935)
    line1_121917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 12), 'line1')
    str_121918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 21), 'str', ' ')
    
    # Call to len(...): (line 935)
    # Processing the call arguments (line 935)
    # Getting the type of 'partstr' (line 935)
    partstr_121920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 30), 'partstr', False)
    # Processing the call keyword arguments (line 935)
    kwargs_121921 = {}
    # Getting the type of 'len' (line 935)
    len_121919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 26), 'len', False)
    # Calling len(args, kwargs) (line 935)
    len_call_result_121922 = invoke(stypy.reporting.localization.Localization(__file__, 935, 26), len_121919, *[partstr_121920], **kwargs_121921)
    
    int_121923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 39), 'int')
    # Applying the binary operator '-' (line 935)
    result_sub_121924 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 26), '-', len_call_result_121922, int_121923)
    
    # Applying the binary operator '*' (line 935)
    result_mul_121925 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 21), '*', str_121918, result_sub_121924)
    
    # Getting the type of 'power' (line 935)
    power_121926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 44), 'power')
    # Applying the binary operator '+' (line 935)
    result_add_121927 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 21), '+', result_mul_121925, power_121926)
    
    # Applying the binary operator '+=' (line 935)
    result_iadd_121928 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 12), '+=', line1_121917, result_add_121927)
    # Assigning a type to the variable 'line1' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 12), 'line1', result_iadd_121928)
    
    # SSA join for if statement (line 928)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 918)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'output' (line 936)
    output_121929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 4), 'output')
    # Getting the type of 'line1' (line 936)
    line1_121930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 14), 'line1')
    str_121931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 22), 'str', '\n')
    # Applying the binary operator '+' (line 936)
    result_add_121932 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 14), '+', line1_121930, str_121931)
    
    # Getting the type of 'line2' (line 936)
    line2_121933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 29), 'line2')
    # Applying the binary operator '+' (line 936)
    result_add_121934 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 27), '+', result_add_121932, line2_121933)
    
    # Applying the binary operator '+=' (line 936)
    result_iadd_121935 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 4), '+=', output_121929, result_add_121934)
    # Assigning a type to the variable 'output' (line 936)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 4), 'output', result_iadd_121935)
    
    # Getting the type of 'output' (line 937)
    output_121936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 11), 'output')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 937)
    n_121937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 25), 'n')
    slice_121938 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 937, 20), n_121937, None, None)
    # Getting the type of 'astr' (line 937)
    astr_121939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 20), 'astr')
    # Obtaining the member '__getitem__' of a type (line 937)
    getitem___121940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 20), astr_121939, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 937)
    subscript_call_result_121941 = invoke(stypy.reporting.localization.Localization(__file__, 937, 20), getitem___121940, slice_121938)
    
    # Applying the binary operator '+' (line 937)
    result_add_121942 = python_operator(stypy.reporting.localization.Localization(__file__, 937, 11), '+', output_121936, subscript_call_result_121941)
    
    # Assigning a type to the variable 'stypy_return_type' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'stypy_return_type', result_add_121942)
    
    # ################# End of '_raise_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raise_power' in the type store
    # Getting the type of 'stypy_return_type' (line 913)
    stypy_return_type_121943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raise_power'
    return stypy_return_type_121943

# Assigning a type to the variable '_raise_power' (line 913)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 0), '_raise_power', _raise_power)
# Declaration of the 'poly1d' class

class poly1d(object, ):
    str_121944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, (-1)), 'str', '\n    A one-dimensional polynomial class.\n\n    A convenience class, used to encapsulate "natural" operations on\n    polynomials so that said operations may take on their customary\n    form in code (see Examples).\n\n    Parameters\n    ----------\n    c_or_r : array_like\n        The polynomial\'s coefficients, in decreasing powers, or if\n        the value of the second parameter is True, the polynomial\'s\n        roots (values where the polynomial evaluates to 0).  For example,\n        ``poly1d([1, 2, 3])`` returns an object that represents\n        :math:`x^2 + 2x + 3`, whereas ``poly1d([1, 2, 3], True)`` returns\n        one that represents :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6`.\n    r : bool, optional\n        If True, `c_or_r` specifies the polynomial\'s roots; the default\n        is False.\n    variable : str, optional\n        Changes the variable used when printing `p` from `x` to `variable`\n        (see Examples).\n\n    Examples\n    --------\n    Construct the polynomial :math:`x^2 + 2x + 3`:\n\n    >>> p = np.poly1d([1, 2, 3])\n    >>> print(np.poly1d(p))\n       2\n    1 x + 2 x + 3\n\n    Evaluate the polynomial at :math:`x = 0.5`:\n\n    >>> p(0.5)\n    4.25\n\n    Find the roots:\n\n    >>> p.r\n    array([-1.+1.41421356j, -1.-1.41421356j])\n    >>> p(p.r)\n    array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j])\n\n    These numbers in the previous line represent (0, 0) to machine precision\n\n    Show the coefficients:\n\n    >>> p.c\n    array([1, 2, 3])\n\n    Display the order (the leading zero-coefficients are removed):\n\n    >>> p.order\n    2\n\n    Show the coefficient of the k-th power in the polynomial\n    (which is equivalent to ``p.c[-(i+1)]``):\n\n    >>> p[1]\n    2\n\n    Polynomials can be added, subtracted, multiplied, and divided\n    (returns quotient and remainder):\n\n    >>> p * p\n    poly1d([ 1,  4, 10, 12,  9])\n\n    >>> (p**3 + 4) / p\n    (poly1d([  1.,   4.,  10.,  12.,   9.]), poly1d([ 4.]))\n\n    ``asarray(p)`` gives the coefficient array, so polynomials can be\n    used in all functions that accept arrays:\n\n    >>> p**2 # square of polynomial\n    poly1d([ 1,  4, 10, 12,  9])\n\n    >>> np.square(p) # square of individual coefficients\n    array([1, 4, 9])\n\n    The variable used in the string representation of `p` can be modified,\n    using the `variable` parameter:\n\n    >>> p = np.poly1d([1,2,3], variable=\'z\')\n    >>> print(p)\n       2\n    1 z + 2 z + 3\n\n    Construct a polynomial from its roots:\n\n    >>> np.poly1d([1, 2], True)\n    poly1d([ 1, -3,  2])\n\n    This is the same polynomial as obtained by:\n\n    >>> np.poly1d([1, -1]) * np.poly1d([1, -2])\n    poly1d([ 1, -3,  2])\n\n    ')
    
    # Assigning a Name to a Name (line 1040):
    
    # Assigning a Name to a Name (line 1041):
    
    # Assigning a Name to a Name (line 1042):
    
    # Assigning a Name to a Name (line 1043):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_121945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 33), 'int')
        # Getting the type of 'None' (line 1045)
        None_121946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 45), 'None')
        defaults = [int_121945, None_121946]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1045, 4, False)
        # Assigning a type to the variable 'self' (line 1046)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__init__', ['c_or_r', 'r', 'variable'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['c_or_r', 'r', 'variable'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Call to isinstance(...): (line 1046)
        # Processing the call arguments (line 1046)
        # Getting the type of 'c_or_r' (line 1046)
        c_or_r_121948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 22), 'c_or_r', False)
        # Getting the type of 'poly1d' (line 1046)
        poly1d_121949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 30), 'poly1d', False)
        # Processing the call keyword arguments (line 1046)
        kwargs_121950 = {}
        # Getting the type of 'isinstance' (line 1046)
        isinstance_121947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1046)
        isinstance_call_result_121951 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 11), isinstance_121947, *[c_or_r_121948, poly1d_121949], **kwargs_121950)
        
        # Testing the type of an if condition (line 1046)
        if_condition_121952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1046, 8), isinstance_call_result_121951)
        # Assigning a type to the variable 'if_condition_121952' (line 1046)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'if_condition_121952', if_condition_121952)
        # SSA begins for if statement (line 1046)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to keys(...): (line 1047)
        # Processing the call keyword arguments (line 1047)
        kwargs_121956 = {}
        # Getting the type of 'c_or_r' (line 1047)
        c_or_r_121953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 23), 'c_or_r', False)
        # Obtaining the member '__dict__' of a type (line 1047)
        dict___121954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 23), c_or_r_121953, '__dict__')
        # Obtaining the member 'keys' of a type (line 1047)
        keys_121955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 23), dict___121954, 'keys')
        # Calling keys(args, kwargs) (line 1047)
        keys_call_result_121957 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 23), keys_121955, *[], **kwargs_121956)
        
        # Testing the type of a for loop iterable (line 1047)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1047, 12), keys_call_result_121957)
        # Getting the type of the for loop variable (line 1047)
        for_loop_var_121958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1047, 12), keys_call_result_121957)
        # Assigning a type to the variable 'key' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 12), 'key', for_loop_var_121958)
        # SSA begins for a for statement (line 1047)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 1048):
        
        # Assigning a Subscript to a Subscript (line 1048):
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 1048)
        key_121959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 53), 'key')
        # Getting the type of 'c_or_r' (line 1048)
        c_or_r_121960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 37), 'c_or_r')
        # Obtaining the member '__dict__' of a type (line 1048)
        dict___121961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1048, 37), c_or_r_121960, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 1048)
        getitem___121962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1048, 37), dict___121961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1048)
        subscript_call_result_121963 = invoke(stypy.reporting.localization.Localization(__file__, 1048, 37), getitem___121962, key_121959)
        
        # Getting the type of 'self' (line 1048)
        self_121964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 16), 'self')
        # Obtaining the member '__dict__' of a type (line 1048)
        dict___121965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1048, 16), self_121964, '__dict__')
        # Getting the type of 'key' (line 1048)
        key_121966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 30), 'key')
        # Storing an element on a container (line 1048)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1048, 16), dict___121965, (key_121966, subscript_call_result_121963))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 1049)
        # Getting the type of 'variable' (line 1049)
        variable_121967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 12), 'variable')
        # Getting the type of 'None' (line 1049)
        None_121968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 31), 'None')
        
        (may_be_121969, more_types_in_union_121970) = may_not_be_none(variable_121967, None_121968)

        if may_be_121969:

            if more_types_in_union_121970:
                # Runtime conditional SSA (line 1049)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1050):
            
            # Assigning a Name to a Subscript (line 1050):
            # Getting the type of 'variable' (line 1050)
            variable_121971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 44), 'variable')
            # Getting the type of 'self' (line 1050)
            self_121972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 16), 'self')
            # Obtaining the member '__dict__' of a type (line 1050)
            dict___121973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 16), self_121972, '__dict__')
            str_121974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 30), 'str', 'variable')
            # Storing an element on a container (line 1050)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1050, 16), dict___121973, (str_121974, variable_121971))

            if more_types_in_union_121970:
                # SSA join for if statement (line 1049)
                module_type_store = module_type_store.join_ssa_context()


        
        # Assigning a type to the variable 'stypy_return_type' (line 1051)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 1046)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'r' (line 1052)
        r_121975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 11), 'r')
        # Testing the type of an if condition (line 1052)
        if_condition_121976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1052, 8), r_121975)
        # Assigning a type to the variable 'if_condition_121976' (line 1052)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'if_condition_121976', if_condition_121976)
        # SSA begins for if statement (line 1052)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1053):
        
        # Assigning a Call to a Name (line 1053):
        
        # Call to poly(...): (line 1053)
        # Processing the call arguments (line 1053)
        # Getting the type of 'c_or_r' (line 1053)
        c_or_r_121978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 26), 'c_or_r', False)
        # Processing the call keyword arguments (line 1053)
        kwargs_121979 = {}
        # Getting the type of 'poly' (line 1053)
        poly_121977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 21), 'poly', False)
        # Calling poly(args, kwargs) (line 1053)
        poly_call_result_121980 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 21), poly_121977, *[c_or_r_121978], **kwargs_121979)
        
        # Assigning a type to the variable 'c_or_r' (line 1053)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 12), 'c_or_r', poly_call_result_121980)
        # SSA join for if statement (line 1052)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1054):
        
        # Assigning a Call to a Name (line 1054):
        
        # Call to atleast_1d(...): (line 1054)
        # Processing the call arguments (line 1054)
        # Getting the type of 'c_or_r' (line 1054)
        c_or_r_121982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 28), 'c_or_r', False)
        # Processing the call keyword arguments (line 1054)
        kwargs_121983 = {}
        # Getting the type of 'atleast_1d' (line 1054)
        atleast_1d_121981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 17), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 1054)
        atleast_1d_call_result_121984 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 17), atleast_1d_121981, *[c_or_r_121982], **kwargs_121983)
        
        # Assigning a type to the variable 'c_or_r' (line 1054)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'c_or_r', atleast_1d_call_result_121984)
        
        
        
        # Call to len(...): (line 1055)
        # Processing the call arguments (line 1055)
        # Getting the type of 'c_or_r' (line 1055)
        c_or_r_121986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 15), 'c_or_r', False)
        # Obtaining the member 'shape' of a type (line 1055)
        shape_121987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 15), c_or_r_121986, 'shape')
        # Processing the call keyword arguments (line 1055)
        kwargs_121988 = {}
        # Getting the type of 'len' (line 1055)
        len_121985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 11), 'len', False)
        # Calling len(args, kwargs) (line 1055)
        len_call_result_121989 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 11), len_121985, *[shape_121987], **kwargs_121988)
        
        int_121990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 31), 'int')
        # Applying the binary operator '>' (line 1055)
        result_gt_121991 = python_operator(stypy.reporting.localization.Localization(__file__, 1055, 11), '>', len_call_result_121989, int_121990)
        
        # Testing the type of an if condition (line 1055)
        if_condition_121992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1055, 8), result_gt_121991)
        # Assigning a type to the variable 'if_condition_121992' (line 1055)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 8), 'if_condition_121992', if_condition_121992)
        # SSA begins for if statement (line 1055)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1056)
        # Processing the call arguments (line 1056)
        str_121994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 29), 'str', 'Polynomial must be 1d only.')
        # Processing the call keyword arguments (line 1056)
        kwargs_121995 = {}
        # Getting the type of 'ValueError' (line 1056)
        ValueError_121993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1056)
        ValueError_call_result_121996 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 18), ValueError_121993, *[str_121994], **kwargs_121995)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1056, 12), ValueError_call_result_121996, 'raise parameter', BaseException)
        # SSA join for if statement (line 1055)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1057):
        
        # Assigning a Call to a Name (line 1057):
        
        # Call to trim_zeros(...): (line 1057)
        # Processing the call arguments (line 1057)
        # Getting the type of 'c_or_r' (line 1057)
        c_or_r_121998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 28), 'c_or_r', False)
        # Processing the call keyword arguments (line 1057)
        str_121999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 41), 'str', 'f')
        keyword_122000 = str_121999
        kwargs_122001 = {'trim': keyword_122000}
        # Getting the type of 'trim_zeros' (line 1057)
        trim_zeros_121997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 17), 'trim_zeros', False)
        # Calling trim_zeros(args, kwargs) (line 1057)
        trim_zeros_call_result_122002 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 17), trim_zeros_121997, *[c_or_r_121998], **kwargs_122001)
        
        # Assigning a type to the variable 'c_or_r' (line 1057)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 8), 'c_or_r', trim_zeros_call_result_122002)
        
        
        
        # Call to len(...): (line 1058)
        # Processing the call arguments (line 1058)
        # Getting the type of 'c_or_r' (line 1058)
        c_or_r_122004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 15), 'c_or_r', False)
        # Processing the call keyword arguments (line 1058)
        kwargs_122005 = {}
        # Getting the type of 'len' (line 1058)
        len_122003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 11), 'len', False)
        # Calling len(args, kwargs) (line 1058)
        len_call_result_122006 = invoke(stypy.reporting.localization.Localization(__file__, 1058, 11), len_122003, *[c_or_r_122004], **kwargs_122005)
        
        int_122007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 26), 'int')
        # Applying the binary operator '==' (line 1058)
        result_eq_122008 = python_operator(stypy.reporting.localization.Localization(__file__, 1058, 11), '==', len_call_result_122006, int_122007)
        
        # Testing the type of an if condition (line 1058)
        if_condition_122009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1058, 8), result_eq_122008)
        # Assigning a type to the variable 'if_condition_122009' (line 1058)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 8), 'if_condition_122009', if_condition_122009)
        # SSA begins for if statement (line 1058)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1059):
        
        # Assigning a Call to a Name (line 1059):
        
        # Call to array(...): (line 1059)
        # Processing the call arguments (line 1059)
        
        # Obtaining an instance of the builtin type 'list' (line 1059)
        list_122012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1059)
        # Adding element type (line 1059)
        float_122013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1059, 30), list_122012, float_122013)
        
        # Processing the call keyword arguments (line 1059)
        kwargs_122014 = {}
        # Getting the type of 'NX' (line 1059)
        NX_122010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 21), 'NX', False)
        # Obtaining the member 'array' of a type (line 1059)
        array_122011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1059, 21), NX_122010, 'array')
        # Calling array(args, kwargs) (line 1059)
        array_call_result_122015 = invoke(stypy.reporting.localization.Localization(__file__, 1059, 21), array_122011, *[list_122012], **kwargs_122014)
        
        # Assigning a type to the variable 'c_or_r' (line 1059)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 12), 'c_or_r', array_call_result_122015)
        # SSA join for if statement (line 1058)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 1060):
        
        # Assigning a Name to a Subscript (line 1060):
        # Getting the type of 'c_or_r' (line 1060)
        c_or_r_122016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 34), 'c_or_r')
        # Getting the type of 'self' (line 1060)
        self_122017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 1060)
        dict___122018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 8), self_122017, '__dict__')
        str_122019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 22), 'str', 'coeffs')
        # Storing an element on a container (line 1060)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1060, 8), dict___122018, (str_122019, c_or_r_122016))
        
        # Assigning a BinOp to a Subscript (line 1061):
        
        # Assigning a BinOp to a Subscript (line 1061):
        
        # Call to len(...): (line 1061)
        # Processing the call arguments (line 1061)
        # Getting the type of 'c_or_r' (line 1061)
        c_or_r_122021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 37), 'c_or_r', False)
        # Processing the call keyword arguments (line 1061)
        kwargs_122022 = {}
        # Getting the type of 'len' (line 1061)
        len_122020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 33), 'len', False)
        # Calling len(args, kwargs) (line 1061)
        len_call_result_122023 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 33), len_122020, *[c_or_r_122021], **kwargs_122022)
        
        int_122024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 47), 'int')
        # Applying the binary operator '-' (line 1061)
        result_sub_122025 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 33), '-', len_call_result_122023, int_122024)
        
        # Getting the type of 'self' (line 1061)
        self_122026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 1061)
        dict___122027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 8), self_122026, '__dict__')
        str_122028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 22), 'str', 'order')
        # Storing an element on a container (line 1061)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 8), dict___122027, (str_122028, result_sub_122025))
        
        # Type idiom detected: calculating its left and rigth part (line 1062)
        # Getting the type of 'variable' (line 1062)
        variable_122029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 11), 'variable')
        # Getting the type of 'None' (line 1062)
        None_122030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 23), 'None')
        
        (may_be_122031, more_types_in_union_122032) = may_be_none(variable_122029, None_122030)

        if may_be_122031:

            if more_types_in_union_122032:
                # Runtime conditional SSA (line 1062)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 1063):
            
            # Assigning a Str to a Name (line 1063):
            str_122033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 23), 'str', 'x')
            # Assigning a type to the variable 'variable' (line 1063)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 12), 'variable', str_122033)

            if more_types_in_union_122032:
                # SSA join for if statement (line 1062)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Subscript (line 1064):
        
        # Assigning a Name to a Subscript (line 1064):
        # Getting the type of 'variable' (line 1064)
        variable_122034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 36), 'variable')
        # Getting the type of 'self' (line 1064)
        self_122035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 1064)
        dict___122036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 8), self_122035, '__dict__')
        str_122037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 22), 'str', 'variable')
        # Storing an element on a container (line 1064)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1064, 8), dict___122036, (str_122037, variable_122034))
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __array__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1066)
        None_122038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 26), 'None')
        defaults = [None_122038]
        # Create a new context for function '__array__'
        module_type_store = module_type_store.open_function_context('__array__', 1066, 4, False)
        # Assigning a type to the variable 'self' (line 1067)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__array__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__array__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__array__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__array__.__dict__.__setitem__('stypy_function_name', 'poly1d.__array__')
        poly1d.__array__.__dict__.__setitem__('stypy_param_names_list', ['t'])
        poly1d.__array__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__array__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__array__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__array__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__array__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__array__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__array__', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array__', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array__(...)' code ##################

        
        # Getting the type of 't' (line 1067)
        t_122039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 11), 't')
        # Testing the type of an if condition (line 1067)
        if_condition_122040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1067, 8), t_122039)
        # Assigning a type to the variable 'if_condition_122040' (line 1067)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'if_condition_122040', if_condition_122040)
        # SSA begins for if statement (line 1067)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to asarray(...): (line 1068)
        # Processing the call arguments (line 1068)
        # Getting the type of 'self' (line 1068)
        self_122043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1068)
        coeffs_122044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1068, 30), self_122043, 'coeffs')
        # Getting the type of 't' (line 1068)
        t_122045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 43), 't', False)
        # Processing the call keyword arguments (line 1068)
        kwargs_122046 = {}
        # Getting the type of 'NX' (line 1068)
        NX_122041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 19), 'NX', False)
        # Obtaining the member 'asarray' of a type (line 1068)
        asarray_122042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1068, 19), NX_122041, 'asarray')
        # Calling asarray(args, kwargs) (line 1068)
        asarray_call_result_122047 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 19), asarray_122042, *[coeffs_122044, t_122045], **kwargs_122046)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1068)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1068, 12), 'stypy_return_type', asarray_call_result_122047)
        # SSA branch for the else part of an if statement (line 1067)
        module_type_store.open_ssa_branch('else')
        
        # Call to asarray(...): (line 1070)
        # Processing the call arguments (line 1070)
        # Getting the type of 'self' (line 1070)
        self_122050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1070)
        coeffs_122051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 30), self_122050, 'coeffs')
        # Processing the call keyword arguments (line 1070)
        kwargs_122052 = {}
        # Getting the type of 'NX' (line 1070)
        NX_122048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 19), 'NX', False)
        # Obtaining the member 'asarray' of a type (line 1070)
        asarray_122049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 19), NX_122048, 'asarray')
        # Calling asarray(args, kwargs) (line 1070)
        asarray_call_result_122053 = invoke(stypy.reporting.localization.Localization(__file__, 1070, 19), asarray_122049, *[coeffs_122051], **kwargs_122052)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1070)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1070, 12), 'stypy_return_type', asarray_call_result_122053)
        # SSA join for if statement (line 1067)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__array__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array__' in the type store
        # Getting the type of 'stypy_return_type' (line 1066)
        stypy_return_type_122054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array__'
        return stypy_return_type_122054


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 1072, 4, False)
        # Assigning a type to the variable 'self' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'poly1d.__repr__')
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Assigning a Call to a Name (line 1073):
        
        # Assigning a Call to a Name (line 1073):
        
        # Call to repr(...): (line 1073)
        # Processing the call arguments (line 1073)
        # Getting the type of 'self' (line 1073)
        self_122056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 20), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1073)
        coeffs_122057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 20), self_122056, 'coeffs')
        # Processing the call keyword arguments (line 1073)
        kwargs_122058 = {}
        # Getting the type of 'repr' (line 1073)
        repr_122055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 1073)
        repr_call_result_122059 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 15), repr_122055, *[coeffs_122057], **kwargs_122058)
        
        # Assigning a type to the variable 'vals' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'vals', repr_call_result_122059)
        
        # Assigning a Subscript to a Name (line 1074):
        
        # Assigning a Subscript to a Name (line 1074):
        
        # Obtaining the type of the subscript
        int_122060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 20), 'int')
        int_122061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 22), 'int')
        slice_122062 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1074, 15), int_122060, int_122061, None)
        # Getting the type of 'vals' (line 1074)
        vals_122063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 15), 'vals')
        # Obtaining the member '__getitem__' of a type (line 1074)
        getitem___122064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 15), vals_122063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1074)
        subscript_call_result_122065 = invoke(stypy.reporting.localization.Localization(__file__, 1074, 15), getitem___122064, slice_122062)
        
        # Assigning a type to the variable 'vals' (line 1074)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1074, 8), 'vals', subscript_call_result_122065)
        str_122066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 15), 'str', 'poly1d(%s)')
        # Getting the type of 'vals' (line 1075)
        vals_122067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 30), 'vals')
        # Applying the binary operator '%' (line 1075)
        result_mod_122068 = python_operator(stypy.reporting.localization.Localization(__file__, 1075, 15), '%', str_122066, vals_122067)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1075)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'stypy_return_type', result_mod_122068)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 1072)
        stypy_return_type_122069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_122069


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 1077, 4, False)
        # Assigning a type to the variable 'self' (line 1078)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__len__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__len__.__dict__.__setitem__('stypy_function_name', 'poly1d.__len__')
        poly1d.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        poly1d.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        # Getting the type of 'self' (line 1078)
        self_122070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 15), 'self')
        # Obtaining the member 'order' of a type (line 1078)
        order_122071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1078, 15), self_122070, 'order')
        # Assigning a type to the variable 'stypy_return_type' (line 1078)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 8), 'stypy_return_type', order_122071)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 1077)
        stypy_return_type_122072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122072)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_122072


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 1080, 4, False)
        # Assigning a type to the variable 'self' (line 1081)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1081, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_function_name', 'poly1d.__str__')
        poly1d.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        poly1d.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Assigning a Str to a Name (line 1081):
        
        # Assigning a Str to a Name (line 1081):
        str_122073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 17), 'str', '0')
        # Assigning a type to the variable 'thestr' (line 1081)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1081, 8), 'thestr', str_122073)
        
        # Assigning a Attribute to a Name (line 1082):
        
        # Assigning a Attribute to a Name (line 1082):
        # Getting the type of 'self' (line 1082)
        self_122074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 14), 'self')
        # Obtaining the member 'variable' of a type (line 1082)
        variable_122075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1082, 14), self_122074, 'variable')
        # Assigning a type to the variable 'var' (line 1082)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 8), 'var', variable_122075)
        
        # Assigning a Subscript to a Name (line 1085):
        
        # Assigning a Subscript to a Name (line 1085):
        
        # Obtaining the type of the subscript
        
        # Call to accumulate(...): (line 1085)
        # Processing the call arguments (line 1085)
        
        # Getting the type of 'self' (line 1085)
        self_122079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 54), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1085)
        coeffs_122080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 54), self_122079, 'coeffs')
        int_122081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 69), 'int')
        # Applying the binary operator '!=' (line 1085)
        result_ne_122082 = python_operator(stypy.reporting.localization.Localization(__file__, 1085, 54), '!=', coeffs_122080, int_122081)
        
        # Processing the call keyword arguments (line 1085)
        kwargs_122083 = {}
        # Getting the type of 'NX' (line 1085)
        NX_122076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 29), 'NX', False)
        # Obtaining the member 'logical_or' of a type (line 1085)
        logical_or_122077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 29), NX_122076, 'logical_or')
        # Obtaining the member 'accumulate' of a type (line 1085)
        accumulate_122078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 29), logical_or_122077, 'accumulate')
        # Calling accumulate(args, kwargs) (line 1085)
        accumulate_call_result_122084 = invoke(stypy.reporting.localization.Localization(__file__, 1085, 29), accumulate_122078, *[result_ne_122082], **kwargs_122083)
        
        # Getting the type of 'self' (line 1085)
        self_122085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 17), 'self')
        # Obtaining the member 'coeffs' of a type (line 1085)
        coeffs_122086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 17), self_122085, 'coeffs')
        # Obtaining the member '__getitem__' of a type (line 1085)
        getitem___122087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 17), coeffs_122086, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1085)
        subscript_call_result_122088 = invoke(stypy.reporting.localization.Localization(__file__, 1085, 17), getitem___122087, accumulate_call_result_122084)
        
        # Assigning a type to the variable 'coeffs' (line 1085)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'coeffs', subscript_call_result_122088)
        
        # Assigning a BinOp to a Name (line 1086):
        
        # Assigning a BinOp to a Name (line 1086):
        
        # Call to len(...): (line 1086)
        # Processing the call arguments (line 1086)
        # Getting the type of 'coeffs' (line 1086)
        coeffs_122090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 16), 'coeffs', False)
        # Processing the call keyword arguments (line 1086)
        kwargs_122091 = {}
        # Getting the type of 'len' (line 1086)
        len_122089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 12), 'len', False)
        # Calling len(args, kwargs) (line 1086)
        len_call_result_122092 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 12), len_122089, *[coeffs_122090], **kwargs_122091)
        
        int_122093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 24), 'int')
        # Applying the binary operator '-' (line 1086)
        result_sub_122094 = python_operator(stypy.reporting.localization.Localization(__file__, 1086, 12), '-', len_call_result_122092, int_122093)
        
        # Assigning a type to the variable 'N' (line 1086)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 8), 'N', result_sub_122094)

        @norecursion
        def fmt_float(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fmt_float'
            module_type_store = module_type_store.open_function_context('fmt_float', 1088, 8, False)
            
            # Passed parameters checking function
            fmt_float.stypy_localization = localization
            fmt_float.stypy_type_of_self = None
            fmt_float.stypy_type_store = module_type_store
            fmt_float.stypy_function_name = 'fmt_float'
            fmt_float.stypy_param_names_list = ['q']
            fmt_float.stypy_varargs_param_name = None
            fmt_float.stypy_kwargs_param_name = None
            fmt_float.stypy_call_defaults = defaults
            fmt_float.stypy_call_varargs = varargs
            fmt_float.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fmt_float', ['q'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fmt_float', localization, ['q'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fmt_float(...)' code ##################

            
            # Assigning a BinOp to a Name (line 1089):
            
            # Assigning a BinOp to a Name (line 1089):
            str_122095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 16), 'str', '%.4g')
            # Getting the type of 'q' (line 1089)
            q_122096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 25), 'q')
            # Applying the binary operator '%' (line 1089)
            result_mod_122097 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 16), '%', str_122095, q_122096)
            
            # Assigning a type to the variable 's' (line 1089)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 12), 's', result_mod_122097)
            
            
            # Call to endswith(...): (line 1090)
            # Processing the call arguments (line 1090)
            str_122100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 26), 'str', '.0000')
            # Processing the call keyword arguments (line 1090)
            kwargs_122101 = {}
            # Getting the type of 's' (line 1090)
            s_122098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 15), 's', False)
            # Obtaining the member 'endswith' of a type (line 1090)
            endswith_122099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 15), s_122098, 'endswith')
            # Calling endswith(args, kwargs) (line 1090)
            endswith_call_result_122102 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 15), endswith_122099, *[str_122100], **kwargs_122101)
            
            # Testing the type of an if condition (line 1090)
            if_condition_122103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1090, 12), endswith_call_result_122102)
            # Assigning a type to the variable 'if_condition_122103' (line 1090)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1090, 12), 'if_condition_122103', if_condition_122103)
            # SSA begins for if statement (line 1090)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 1091):
            
            # Assigning a Subscript to a Name (line 1091):
            
            # Obtaining the type of the subscript
            int_122104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 23), 'int')
            slice_122105 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1091, 20), None, int_122104, None)
            # Getting the type of 's' (line 1091)
            s_122106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 20), 's')
            # Obtaining the member '__getitem__' of a type (line 1091)
            getitem___122107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 20), s_122106, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1091)
            subscript_call_result_122108 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 20), getitem___122107, slice_122105)
            
            # Assigning a type to the variable 's' (line 1091)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1091, 16), 's', subscript_call_result_122108)
            # SSA join for if statement (line 1090)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 's' (line 1092)
            s_122109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 19), 's')
            # Assigning a type to the variable 'stypy_return_type' (line 1092)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 12), 'stypy_return_type', s_122109)
            
            # ################# End of 'fmt_float(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fmt_float' in the type store
            # Getting the type of 'stypy_return_type' (line 1088)
            stypy_return_type_122110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_122110)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fmt_float'
            return stypy_return_type_122110

        # Assigning a type to the variable 'fmt_float' (line 1088)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'fmt_float', fmt_float)
        
        
        # Call to range(...): (line 1094)
        # Processing the call arguments (line 1094)
        
        # Call to len(...): (line 1094)
        # Processing the call arguments (line 1094)
        # Getting the type of 'coeffs' (line 1094)
        coeffs_122113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 27), 'coeffs', False)
        # Processing the call keyword arguments (line 1094)
        kwargs_122114 = {}
        # Getting the type of 'len' (line 1094)
        len_122112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 23), 'len', False)
        # Calling len(args, kwargs) (line 1094)
        len_call_result_122115 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 23), len_122112, *[coeffs_122113], **kwargs_122114)
        
        # Processing the call keyword arguments (line 1094)
        kwargs_122116 = {}
        # Getting the type of 'range' (line 1094)
        range_122111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 17), 'range', False)
        # Calling range(args, kwargs) (line 1094)
        range_call_result_122117 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 17), range_122111, *[len_call_result_122115], **kwargs_122116)
        
        # Testing the type of a for loop iterable (line 1094)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1094, 8), range_call_result_122117)
        # Getting the type of the for loop variable (line 1094)
        for_loop_var_122118 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1094, 8), range_call_result_122117)
        # Assigning a type to the variable 'k' (line 1094)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'k', for_loop_var_122118)
        # SSA begins for a for statement (line 1094)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to iscomplex(...): (line 1095)
        # Processing the call arguments (line 1095)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1095)
        k_122120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 36), 'k', False)
        # Getting the type of 'coeffs' (line 1095)
        coeffs_122121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 29), 'coeffs', False)
        # Obtaining the member '__getitem__' of a type (line 1095)
        getitem___122122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 29), coeffs_122121, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1095)
        subscript_call_result_122123 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 29), getitem___122122, k_122120)
        
        # Processing the call keyword arguments (line 1095)
        kwargs_122124 = {}
        # Getting the type of 'iscomplex' (line 1095)
        iscomplex_122119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 19), 'iscomplex', False)
        # Calling iscomplex(args, kwargs) (line 1095)
        iscomplex_call_result_122125 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 19), iscomplex_122119, *[subscript_call_result_122123], **kwargs_122124)
        
        # Applying the 'not' unary operator (line 1095)
        result_not__122126 = python_operator(stypy.reporting.localization.Localization(__file__, 1095, 15), 'not', iscomplex_call_result_122125)
        
        # Testing the type of an if condition (line 1095)
        if_condition_122127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1095, 12), result_not__122126)
        # Assigning a type to the variable 'if_condition_122127' (line 1095)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'if_condition_122127', if_condition_122127)
        # SSA begins for if statement (line 1095)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1096):
        
        # Assigning a Call to a Name (line 1096):
        
        # Call to fmt_float(...): (line 1096)
        # Processing the call arguments (line 1096)
        
        # Call to real(...): (line 1096)
        # Processing the call arguments (line 1096)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1096)
        k_122130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 48), 'k', False)
        # Getting the type of 'coeffs' (line 1096)
        coeffs_122131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 41), 'coeffs', False)
        # Obtaining the member '__getitem__' of a type (line 1096)
        getitem___122132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1096, 41), coeffs_122131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1096)
        subscript_call_result_122133 = invoke(stypy.reporting.localization.Localization(__file__, 1096, 41), getitem___122132, k_122130)
        
        # Processing the call keyword arguments (line 1096)
        kwargs_122134 = {}
        # Getting the type of 'real' (line 1096)
        real_122129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 36), 'real', False)
        # Calling real(args, kwargs) (line 1096)
        real_call_result_122135 = invoke(stypy.reporting.localization.Localization(__file__, 1096, 36), real_122129, *[subscript_call_result_122133], **kwargs_122134)
        
        # Processing the call keyword arguments (line 1096)
        kwargs_122136 = {}
        # Getting the type of 'fmt_float' (line 1096)
        fmt_float_122128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 26), 'fmt_float', False)
        # Calling fmt_float(args, kwargs) (line 1096)
        fmt_float_call_result_122137 = invoke(stypy.reporting.localization.Localization(__file__, 1096, 26), fmt_float_122128, *[real_call_result_122135], **kwargs_122136)
        
        # Assigning a type to the variable 'coefstr' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 16), 'coefstr', fmt_float_call_result_122137)
        # SSA branch for the else part of an if statement (line 1095)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to real(...): (line 1097)
        # Processing the call arguments (line 1097)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1097)
        k_122139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 29), 'k', False)
        # Getting the type of 'coeffs' (line 1097)
        coeffs_122140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 22), 'coeffs', False)
        # Obtaining the member '__getitem__' of a type (line 1097)
        getitem___122141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 22), coeffs_122140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1097)
        subscript_call_result_122142 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 22), getitem___122141, k_122139)
        
        # Processing the call keyword arguments (line 1097)
        kwargs_122143 = {}
        # Getting the type of 'real' (line 1097)
        real_122138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 17), 'real', False)
        # Calling real(args, kwargs) (line 1097)
        real_call_result_122144 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 17), real_122138, *[subscript_call_result_122142], **kwargs_122143)
        
        int_122145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 36), 'int')
        # Applying the binary operator '==' (line 1097)
        result_eq_122146 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 17), '==', real_call_result_122144, int_122145)
        
        # Testing the type of an if condition (line 1097)
        if_condition_122147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1097, 17), result_eq_122146)
        # Assigning a type to the variable 'if_condition_122147' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 17), 'if_condition_122147', if_condition_122147)
        # SSA begins for if statement (line 1097)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1098):
        
        # Assigning a BinOp to a Name (line 1098):
        str_122148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 26), 'str', '%sj')
        
        # Call to fmt_float(...): (line 1098)
        # Processing the call arguments (line 1098)
        
        # Call to imag(...): (line 1098)
        # Processing the call arguments (line 1098)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1098)
        k_122151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 56), 'k', False)
        # Getting the type of 'coeffs' (line 1098)
        coeffs_122152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 49), 'coeffs', False)
        # Obtaining the member '__getitem__' of a type (line 1098)
        getitem___122153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1098, 49), coeffs_122152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1098)
        subscript_call_result_122154 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 49), getitem___122153, k_122151)
        
        # Processing the call keyword arguments (line 1098)
        kwargs_122155 = {}
        # Getting the type of 'imag' (line 1098)
        imag_122150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 44), 'imag', False)
        # Calling imag(args, kwargs) (line 1098)
        imag_call_result_122156 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 44), imag_122150, *[subscript_call_result_122154], **kwargs_122155)
        
        # Processing the call keyword arguments (line 1098)
        kwargs_122157 = {}
        # Getting the type of 'fmt_float' (line 1098)
        fmt_float_122149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 34), 'fmt_float', False)
        # Calling fmt_float(args, kwargs) (line 1098)
        fmt_float_call_result_122158 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 34), fmt_float_122149, *[imag_call_result_122156], **kwargs_122157)
        
        # Applying the binary operator '%' (line 1098)
        result_mod_122159 = python_operator(stypy.reporting.localization.Localization(__file__, 1098, 26), '%', str_122148, fmt_float_call_result_122158)
        
        # Assigning a type to the variable 'coefstr' (line 1098)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1098, 16), 'coefstr', result_mod_122159)
        # SSA branch for the else part of an if statement (line 1097)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 1100):
        
        # Assigning a BinOp to a Name (line 1100):
        str_122160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 26), 'str', '(%s + %sj)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1100)
        tuple_122161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1100)
        # Adding element type (line 1100)
        
        # Call to fmt_float(...): (line 1100)
        # Processing the call arguments (line 1100)
        
        # Call to real(...): (line 1100)
        # Processing the call arguments (line 1100)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1100)
        k_122164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 64), 'k', False)
        # Getting the type of 'coeffs' (line 1100)
        coeffs_122165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 57), 'coeffs', False)
        # Obtaining the member '__getitem__' of a type (line 1100)
        getitem___122166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 57), coeffs_122165, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1100)
        subscript_call_result_122167 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 57), getitem___122166, k_122164)
        
        # Processing the call keyword arguments (line 1100)
        kwargs_122168 = {}
        # Getting the type of 'real' (line 1100)
        real_122163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 52), 'real', False)
        # Calling real(args, kwargs) (line 1100)
        real_call_result_122169 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 52), real_122163, *[subscript_call_result_122167], **kwargs_122168)
        
        # Processing the call keyword arguments (line 1100)
        kwargs_122170 = {}
        # Getting the type of 'fmt_float' (line 1100)
        fmt_float_122162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 42), 'fmt_float', False)
        # Calling fmt_float(args, kwargs) (line 1100)
        fmt_float_call_result_122171 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 42), fmt_float_122162, *[real_call_result_122169], **kwargs_122170)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1100, 42), tuple_122161, fmt_float_call_result_122171)
        # Adding element type (line 1100)
        
        # Call to fmt_float(...): (line 1101)
        # Processing the call arguments (line 1101)
        
        # Call to imag(...): (line 1101)
        # Processing the call arguments (line 1101)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1101)
        k_122174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 64), 'k', False)
        # Getting the type of 'coeffs' (line 1101)
        coeffs_122175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 57), 'coeffs', False)
        # Obtaining the member '__getitem__' of a type (line 1101)
        getitem___122176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 57), coeffs_122175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1101)
        subscript_call_result_122177 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 57), getitem___122176, k_122174)
        
        # Processing the call keyword arguments (line 1101)
        kwargs_122178 = {}
        # Getting the type of 'imag' (line 1101)
        imag_122173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 52), 'imag', False)
        # Calling imag(args, kwargs) (line 1101)
        imag_call_result_122179 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 52), imag_122173, *[subscript_call_result_122177], **kwargs_122178)
        
        # Processing the call keyword arguments (line 1101)
        kwargs_122180 = {}
        # Getting the type of 'fmt_float' (line 1101)
        fmt_float_122172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 42), 'fmt_float', False)
        # Calling fmt_float(args, kwargs) (line 1101)
        fmt_float_call_result_122181 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 42), fmt_float_122172, *[imag_call_result_122179], **kwargs_122180)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1100, 42), tuple_122161, fmt_float_call_result_122181)
        
        # Applying the binary operator '%' (line 1100)
        result_mod_122182 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 26), '%', str_122160, tuple_122161)
        
        # Assigning a type to the variable 'coefstr' (line 1100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 16), 'coefstr', result_mod_122182)
        # SSA join for if statement (line 1097)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1095)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1103):
        
        # Assigning a BinOp to a Name (line 1103):
        # Getting the type of 'N' (line 1103)
        N_122183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 21), 'N')
        # Getting the type of 'k' (line 1103)
        k_122184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 23), 'k')
        # Applying the binary operator '-' (line 1103)
        result_sub_122185 = python_operator(stypy.reporting.localization.Localization(__file__, 1103, 21), '-', N_122183, k_122184)
        
        # Assigning a type to the variable 'power' (line 1103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 12), 'power', result_sub_122185)
        
        
        # Getting the type of 'power' (line 1104)
        power_122186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 15), 'power')
        int_122187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 24), 'int')
        # Applying the binary operator '==' (line 1104)
        result_eq_122188 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 15), '==', power_122186, int_122187)
        
        # Testing the type of an if condition (line 1104)
        if_condition_122189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1104, 12), result_eq_122188)
        # Assigning a type to the variable 'if_condition_122189' (line 1104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 12), 'if_condition_122189', if_condition_122189)
        # SSA begins for if statement (line 1104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'coefstr' (line 1105)
        coefstr_122190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 19), 'coefstr')
        str_122191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 30), 'str', '0')
        # Applying the binary operator '!=' (line 1105)
        result_ne_122192 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 19), '!=', coefstr_122190, str_122191)
        
        # Testing the type of an if condition (line 1105)
        if_condition_122193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1105, 16), result_ne_122192)
        # Assigning a type to the variable 'if_condition_122193' (line 1105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1105, 16), 'if_condition_122193', if_condition_122193)
        # SSA begins for if statement (line 1105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1106):
        
        # Assigning a BinOp to a Name (line 1106):
        str_122194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 29), 'str', '%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1106)
        tuple_122195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1106)
        # Adding element type (line 1106)
        # Getting the type of 'coefstr' (line 1106)
        coefstr_122196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 37), 'coefstr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 37), tuple_122195, coefstr_122196)
        
        # Applying the binary operator '%' (line 1106)
        result_mod_122197 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 29), '%', str_122194, tuple_122195)
        
        # Assigning a type to the variable 'newstr' (line 1106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 20), 'newstr', result_mod_122197)
        # SSA branch for the else part of an if statement (line 1105)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'k' (line 1108)
        k_122198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 23), 'k')
        int_122199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 28), 'int')
        # Applying the binary operator '==' (line 1108)
        result_eq_122200 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 23), '==', k_122198, int_122199)
        
        # Testing the type of an if condition (line 1108)
        if_condition_122201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1108, 20), result_eq_122200)
        # Assigning a type to the variable 'if_condition_122201' (line 1108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 20), 'if_condition_122201', if_condition_122201)
        # SSA begins for if statement (line 1108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 1109):
        
        # Assigning a Str to a Name (line 1109):
        str_122202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 33), 'str', '0')
        # Assigning a type to the variable 'newstr' (line 1109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 24), 'newstr', str_122202)
        # SSA branch for the else part of an if statement (line 1108)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 1111):
        
        # Assigning a Str to a Name (line 1111):
        str_122203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1111, 33), 'str', '')
        # Assigning a type to the variable 'newstr' (line 1111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1111, 24), 'newstr', str_122203)
        # SSA join for if statement (line 1108)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1105)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1104)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'power' (line 1112)
        power_122204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 17), 'power')
        int_122205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, 26), 'int')
        # Applying the binary operator '==' (line 1112)
        result_eq_122206 = python_operator(stypy.reporting.localization.Localization(__file__, 1112, 17), '==', power_122204, int_122205)
        
        # Testing the type of an if condition (line 1112)
        if_condition_122207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1112, 17), result_eq_122206)
        # Assigning a type to the variable 'if_condition_122207' (line 1112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 17), 'if_condition_122207', if_condition_122207)
        # SSA begins for if statement (line 1112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'coefstr' (line 1113)
        coefstr_122208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 19), 'coefstr')
        str_122209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 30), 'str', '0')
        # Applying the binary operator '==' (line 1113)
        result_eq_122210 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 19), '==', coefstr_122208, str_122209)
        
        # Testing the type of an if condition (line 1113)
        if_condition_122211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1113, 16), result_eq_122210)
        # Assigning a type to the variable 'if_condition_122211' (line 1113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 16), 'if_condition_122211', if_condition_122211)
        # SSA begins for if statement (line 1113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 1114):
        
        # Assigning a Str to a Name (line 1114):
        str_122212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 29), 'str', '')
        # Assigning a type to the variable 'newstr' (line 1114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 20), 'newstr', str_122212)
        # SSA branch for the else part of an if statement (line 1113)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'coefstr' (line 1115)
        coefstr_122213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 21), 'coefstr')
        str_122214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 32), 'str', 'b')
        # Applying the binary operator '==' (line 1115)
        result_eq_122215 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 21), '==', coefstr_122213, str_122214)
        
        # Testing the type of an if condition (line 1115)
        if_condition_122216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1115, 21), result_eq_122215)
        # Assigning a type to the variable 'if_condition_122216' (line 1115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 21), 'if_condition_122216', if_condition_122216)
        # SSA begins for if statement (line 1115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 1116):
        
        # Assigning a Name to a Name (line 1116):
        # Getting the type of 'var' (line 1116)
        var_122217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 29), 'var')
        # Assigning a type to the variable 'newstr' (line 1116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 20), 'newstr', var_122217)
        # SSA branch for the else part of an if statement (line 1115)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 1118):
        
        # Assigning a BinOp to a Name (line 1118):
        str_122218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 29), 'str', '%s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1118)
        tuple_122219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1118)
        # Adding element type (line 1118)
        # Getting the type of 'coefstr' (line 1118)
        coefstr_122220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 40), 'coefstr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1118, 40), tuple_122219, coefstr_122220)
        # Adding element type (line 1118)
        # Getting the type of 'var' (line 1118)
        var_122221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 49), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1118, 40), tuple_122219, var_122221)
        
        # Applying the binary operator '%' (line 1118)
        result_mod_122222 = python_operator(stypy.reporting.localization.Localization(__file__, 1118, 29), '%', str_122218, tuple_122219)
        
        # Assigning a type to the variable 'newstr' (line 1118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 20), 'newstr', result_mod_122222)
        # SSA join for if statement (line 1115)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1113)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1112)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'coefstr' (line 1120)
        coefstr_122223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 19), 'coefstr')
        str_122224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 30), 'str', '0')
        # Applying the binary operator '==' (line 1120)
        result_eq_122225 = python_operator(stypy.reporting.localization.Localization(__file__, 1120, 19), '==', coefstr_122223, str_122224)
        
        # Testing the type of an if condition (line 1120)
        if_condition_122226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1120, 16), result_eq_122225)
        # Assigning a type to the variable 'if_condition_122226' (line 1120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1120, 16), 'if_condition_122226', if_condition_122226)
        # SSA begins for if statement (line 1120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 1121):
        
        # Assigning a Str to a Name (line 1121):
        str_122227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 29), 'str', '')
        # Assigning a type to the variable 'newstr' (line 1121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 20), 'newstr', str_122227)
        # SSA branch for the else part of an if statement (line 1120)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'coefstr' (line 1122)
        coefstr_122228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 21), 'coefstr')
        str_122229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 32), 'str', 'b')
        # Applying the binary operator '==' (line 1122)
        result_eq_122230 = python_operator(stypy.reporting.localization.Localization(__file__, 1122, 21), '==', coefstr_122228, str_122229)
        
        # Testing the type of an if condition (line 1122)
        if_condition_122231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1122, 21), result_eq_122230)
        # Assigning a type to the variable 'if_condition_122231' (line 1122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 21), 'if_condition_122231', if_condition_122231)
        # SSA begins for if statement (line 1122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1123):
        
        # Assigning a BinOp to a Name (line 1123):
        str_122232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1123, 29), 'str', '%s**%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1123)
        tuple_122233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1123, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1123)
        # Adding element type (line 1123)
        # Getting the type of 'var' (line 1123)
        var_122234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 41), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1123, 41), tuple_122233, var_122234)
        # Adding element type (line 1123)
        # Getting the type of 'power' (line 1123)
        power_122235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 46), 'power')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1123, 41), tuple_122233, power_122235)
        
        # Applying the binary operator '%' (line 1123)
        result_mod_122236 = python_operator(stypy.reporting.localization.Localization(__file__, 1123, 29), '%', str_122232, tuple_122233)
        
        # Assigning a type to the variable 'newstr' (line 1123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1123, 20), 'newstr', result_mod_122236)
        # SSA branch for the else part of an if statement (line 1122)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 1125):
        
        # Assigning a BinOp to a Name (line 1125):
        str_122237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 29), 'str', '%s %s**%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1125)
        tuple_122238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1125)
        # Adding element type (line 1125)
        # Getting the type of 'coefstr' (line 1125)
        coefstr_122239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 44), 'coefstr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 44), tuple_122238, coefstr_122239)
        # Adding element type (line 1125)
        # Getting the type of 'var' (line 1125)
        var_122240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 53), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 44), tuple_122238, var_122240)
        # Adding element type (line 1125)
        # Getting the type of 'power' (line 1125)
        power_122241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 58), 'power')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 44), tuple_122238, power_122241)
        
        # Applying the binary operator '%' (line 1125)
        result_mod_122242 = python_operator(stypy.reporting.localization.Localization(__file__, 1125, 29), '%', str_122237, tuple_122238)
        
        # Assigning a type to the variable 'newstr' (line 1125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1125, 20), 'newstr', result_mod_122242)
        # SSA join for if statement (line 1122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1120)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1112)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'k' (line 1127)
        k_122243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 15), 'k')
        int_122244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 19), 'int')
        # Applying the binary operator '>' (line 1127)
        result_gt_122245 = python_operator(stypy.reporting.localization.Localization(__file__, 1127, 15), '>', k_122243, int_122244)
        
        # Testing the type of an if condition (line 1127)
        if_condition_122246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1127, 12), result_gt_122245)
        # Assigning a type to the variable 'if_condition_122246' (line 1127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1127, 12), 'if_condition_122246', if_condition_122246)
        # SSA begins for if statement (line 1127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'newstr' (line 1128)
        newstr_122247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 19), 'newstr')
        str_122248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 29), 'str', '')
        # Applying the binary operator '!=' (line 1128)
        result_ne_122249 = python_operator(stypy.reporting.localization.Localization(__file__, 1128, 19), '!=', newstr_122247, str_122248)
        
        # Testing the type of an if condition (line 1128)
        if_condition_122250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1128, 16), result_ne_122249)
        # Assigning a type to the variable 'if_condition_122250' (line 1128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1128, 16), 'if_condition_122250', if_condition_122250)
        # SSA begins for if statement (line 1128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to startswith(...): (line 1129)
        # Processing the call arguments (line 1129)
        str_122253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1129, 41), 'str', '-')
        # Processing the call keyword arguments (line 1129)
        kwargs_122254 = {}
        # Getting the type of 'newstr' (line 1129)
        newstr_122251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 23), 'newstr', False)
        # Obtaining the member 'startswith' of a type (line 1129)
        startswith_122252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1129, 23), newstr_122251, 'startswith')
        # Calling startswith(args, kwargs) (line 1129)
        startswith_call_result_122255 = invoke(stypy.reporting.localization.Localization(__file__, 1129, 23), startswith_122252, *[str_122253], **kwargs_122254)
        
        # Testing the type of an if condition (line 1129)
        if_condition_122256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1129, 20), startswith_call_result_122255)
        # Assigning a type to the variable 'if_condition_122256' (line 1129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 20), 'if_condition_122256', if_condition_122256)
        # SSA begins for if statement (line 1129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1130):
        
        # Assigning a BinOp to a Name (line 1130):
        str_122257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 33), 'str', '%s - %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1130)
        tuple_122258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1130)
        # Adding element type (line 1130)
        # Getting the type of 'thestr' (line 1130)
        thestr_122259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 46), 'thestr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 46), tuple_122258, thestr_122259)
        # Adding element type (line 1130)
        
        # Obtaining the type of the subscript
        int_122260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 61), 'int')
        slice_122261 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1130, 54), int_122260, None, None)
        # Getting the type of 'newstr' (line 1130)
        newstr_122262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 54), 'newstr')
        # Obtaining the member '__getitem__' of a type (line 1130)
        getitem___122263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 54), newstr_122262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1130)
        subscript_call_result_122264 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 54), getitem___122263, slice_122261)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 46), tuple_122258, subscript_call_result_122264)
        
        # Applying the binary operator '%' (line 1130)
        result_mod_122265 = python_operator(stypy.reporting.localization.Localization(__file__, 1130, 33), '%', str_122257, tuple_122258)
        
        # Assigning a type to the variable 'thestr' (line 1130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1130, 24), 'thestr', result_mod_122265)
        # SSA branch for the else part of an if statement (line 1129)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 1132):
        
        # Assigning a BinOp to a Name (line 1132):
        str_122266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 33), 'str', '%s + %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1132)
        tuple_122267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1132)
        # Adding element type (line 1132)
        # Getting the type of 'thestr' (line 1132)
        thestr_122268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 46), 'thestr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1132, 46), tuple_122267, thestr_122268)
        # Adding element type (line 1132)
        # Getting the type of 'newstr' (line 1132)
        newstr_122269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 54), 'newstr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1132, 46), tuple_122267, newstr_122269)
        
        # Applying the binary operator '%' (line 1132)
        result_mod_122270 = python_operator(stypy.reporting.localization.Localization(__file__, 1132, 33), '%', str_122266, tuple_122267)
        
        # Assigning a type to the variable 'thestr' (line 1132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 24), 'thestr', result_mod_122270)
        # SSA join for if statement (line 1129)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1128)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1127)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1134):
        
        # Assigning a Name to a Name (line 1134):
        # Getting the type of 'newstr' (line 1134)
        newstr_122271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 25), 'newstr')
        # Assigning a type to the variable 'thestr' (line 1134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 16), 'thestr', newstr_122271)
        # SSA join for if statement (line 1127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _raise_power(...): (line 1135)
        # Processing the call arguments (line 1135)
        # Getting the type of 'thestr' (line 1135)
        thestr_122273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 28), 'thestr', False)
        # Processing the call keyword arguments (line 1135)
        kwargs_122274 = {}
        # Getting the type of '_raise_power' (line 1135)
        _raise_power_122272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 15), '_raise_power', False)
        # Calling _raise_power(args, kwargs) (line 1135)
        _raise_power_call_result_122275 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 15), _raise_power_122272, *[thestr_122273], **kwargs_122274)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 8), 'stypy_return_type', _raise_power_call_result_122275)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 1080)
        stypy_return_type_122276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_122276


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 1137, 4, False)
        # Assigning a type to the variable 'self' (line 1138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__call__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__call__.__dict__.__setitem__('stypy_function_name', 'poly1d.__call__')
        poly1d.__call__.__dict__.__setitem__('stypy_param_names_list', ['val'])
        poly1d.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__call__', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to polyval(...): (line 1138)
        # Processing the call arguments (line 1138)
        # Getting the type of 'self' (line 1138)
        self_122278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 23), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1138)
        coeffs_122279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1138, 23), self_122278, 'coeffs')
        # Getting the type of 'val' (line 1138)
        val_122280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 36), 'val', False)
        # Processing the call keyword arguments (line 1138)
        kwargs_122281 = {}
        # Getting the type of 'polyval' (line 1138)
        polyval_122277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 15), 'polyval', False)
        # Calling polyval(args, kwargs) (line 1138)
        polyval_call_result_122282 = invoke(stypy.reporting.localization.Localization(__file__, 1138, 15), polyval_122277, *[coeffs_122279, val_122280], **kwargs_122281)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 8), 'stypy_return_type', polyval_call_result_122282)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 1137)
        stypy_return_type_122283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_122283


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 1140, 4, False)
        # Assigning a type to the variable 'self' (line 1141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__neg__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__neg__.__dict__.__setitem__('stypy_function_name', 'poly1d.__neg__')
        poly1d.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        poly1d.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        # Call to poly1d(...): (line 1141)
        # Processing the call arguments (line 1141)
        
        # Getting the type of 'self' (line 1141)
        self_122285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 23), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1141)
        coeffs_122286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 23), self_122285, 'coeffs')
        # Applying the 'usub' unary operator (line 1141)
        result___neg___122287 = python_operator(stypy.reporting.localization.Localization(__file__, 1141, 22), 'usub', coeffs_122286)
        
        # Processing the call keyword arguments (line 1141)
        kwargs_122288 = {}
        # Getting the type of 'poly1d' (line 1141)
        poly1d_122284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1141)
        poly1d_call_result_122289 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 15), poly1d_122284, *[result___neg___122287], **kwargs_122288)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 8), 'stypy_return_type', poly1d_call_result_122289)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 1140)
        stypy_return_type_122290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_122290


    @norecursion
    def __pos__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pos__'
        module_type_store = module_type_store.open_function_context('__pos__', 1143, 4, False)
        # Assigning a type to the variable 'self' (line 1144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__pos__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__pos__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__pos__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__pos__.__dict__.__setitem__('stypy_function_name', 'poly1d.__pos__')
        poly1d.__pos__.__dict__.__setitem__('stypy_param_names_list', [])
        poly1d.__pos__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__pos__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__pos__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__pos__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__pos__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__pos__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__pos__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pos__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pos__(...)' code ##################

        # Getting the type of 'self' (line 1144)
        self_122291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 1144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 8), 'stypy_return_type', self_122291)
        
        # ################# End of '__pos__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pos__' in the type store
        # Getting the type of 'stypy_return_type' (line 1143)
        stypy_return_type_122292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pos__'
        return stypy_return_type_122292


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 1146, 4, False)
        # Assigning a type to the variable 'self' (line 1147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__mul__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__mul__.__dict__.__setitem__('stypy_function_name', 'poly1d.__mul__')
        poly1d.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        
        # Call to isscalar(...): (line 1147)
        # Processing the call arguments (line 1147)
        # Getting the type of 'other' (line 1147)
        other_122294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 20), 'other', False)
        # Processing the call keyword arguments (line 1147)
        kwargs_122295 = {}
        # Getting the type of 'isscalar' (line 1147)
        isscalar_122293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 11), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 1147)
        isscalar_call_result_122296 = invoke(stypy.reporting.localization.Localization(__file__, 1147, 11), isscalar_122293, *[other_122294], **kwargs_122295)
        
        # Testing the type of an if condition (line 1147)
        if_condition_122297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1147, 8), isscalar_call_result_122296)
        # Assigning a type to the variable 'if_condition_122297' (line 1147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 8), 'if_condition_122297', if_condition_122297)
        # SSA begins for if statement (line 1147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to poly1d(...): (line 1148)
        # Processing the call arguments (line 1148)
        # Getting the type of 'self' (line 1148)
        self_122299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 26), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1148)
        coeffs_122300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 26), self_122299, 'coeffs')
        # Getting the type of 'other' (line 1148)
        other_122301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 40), 'other', False)
        # Applying the binary operator '*' (line 1148)
        result_mul_122302 = python_operator(stypy.reporting.localization.Localization(__file__, 1148, 26), '*', coeffs_122300, other_122301)
        
        # Processing the call keyword arguments (line 1148)
        kwargs_122303 = {}
        # Getting the type of 'poly1d' (line 1148)
        poly1d_122298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 19), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1148)
        poly1d_call_result_122304 = invoke(stypy.reporting.localization.Localization(__file__, 1148, 19), poly1d_122298, *[result_mul_122302], **kwargs_122303)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'stypy_return_type', poly1d_call_result_122304)
        # SSA branch for the else part of an if statement (line 1147)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1150):
        
        # Assigning a Call to a Name (line 1150):
        
        # Call to poly1d(...): (line 1150)
        # Processing the call arguments (line 1150)
        # Getting the type of 'other' (line 1150)
        other_122306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 27), 'other', False)
        # Processing the call keyword arguments (line 1150)
        kwargs_122307 = {}
        # Getting the type of 'poly1d' (line 1150)
        poly1d_122305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 20), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1150)
        poly1d_call_result_122308 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 20), poly1d_122305, *[other_122306], **kwargs_122307)
        
        # Assigning a type to the variable 'other' (line 1150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 12), 'other', poly1d_call_result_122308)
        
        # Call to poly1d(...): (line 1151)
        # Processing the call arguments (line 1151)
        
        # Call to polymul(...): (line 1151)
        # Processing the call arguments (line 1151)
        # Getting the type of 'self' (line 1151)
        self_122311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 34), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1151)
        coeffs_122312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 34), self_122311, 'coeffs')
        # Getting the type of 'other' (line 1151)
        other_122313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 47), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1151)
        coeffs_122314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 47), other_122313, 'coeffs')
        # Processing the call keyword arguments (line 1151)
        kwargs_122315 = {}
        # Getting the type of 'polymul' (line 1151)
        polymul_122310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 26), 'polymul', False)
        # Calling polymul(args, kwargs) (line 1151)
        polymul_call_result_122316 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 26), polymul_122310, *[coeffs_122312, coeffs_122314], **kwargs_122315)
        
        # Processing the call keyword arguments (line 1151)
        kwargs_122317 = {}
        # Getting the type of 'poly1d' (line 1151)
        poly1d_122309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 19), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1151)
        poly1d_call_result_122318 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 19), poly1d_122309, *[polymul_call_result_122316], **kwargs_122317)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 12), 'stypy_return_type', poly1d_call_result_122318)
        # SSA join for if statement (line 1147)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 1146)
        stypy_return_type_122319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_122319


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 1153, 4, False)
        # Assigning a type to the variable 'self' (line 1154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__rmul__.__dict__.__setitem__('stypy_function_name', 'poly1d.__rmul__')
        poly1d.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__rmul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        
        
        # Call to isscalar(...): (line 1154)
        # Processing the call arguments (line 1154)
        # Getting the type of 'other' (line 1154)
        other_122321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 20), 'other', False)
        # Processing the call keyword arguments (line 1154)
        kwargs_122322 = {}
        # Getting the type of 'isscalar' (line 1154)
        isscalar_122320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 11), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 1154)
        isscalar_call_result_122323 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 11), isscalar_122320, *[other_122321], **kwargs_122322)
        
        # Testing the type of an if condition (line 1154)
        if_condition_122324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1154, 8), isscalar_call_result_122323)
        # Assigning a type to the variable 'if_condition_122324' (line 1154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 8), 'if_condition_122324', if_condition_122324)
        # SSA begins for if statement (line 1154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to poly1d(...): (line 1155)
        # Processing the call arguments (line 1155)
        # Getting the type of 'other' (line 1155)
        other_122326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 26), 'other', False)
        # Getting the type of 'self' (line 1155)
        self_122327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 34), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1155)
        coeffs_122328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 34), self_122327, 'coeffs')
        # Applying the binary operator '*' (line 1155)
        result_mul_122329 = python_operator(stypy.reporting.localization.Localization(__file__, 1155, 26), '*', other_122326, coeffs_122328)
        
        # Processing the call keyword arguments (line 1155)
        kwargs_122330 = {}
        # Getting the type of 'poly1d' (line 1155)
        poly1d_122325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 19), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1155)
        poly1d_call_result_122331 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 19), poly1d_122325, *[result_mul_122329], **kwargs_122330)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 12), 'stypy_return_type', poly1d_call_result_122331)
        # SSA branch for the else part of an if statement (line 1154)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1157):
        
        # Assigning a Call to a Name (line 1157):
        
        # Call to poly1d(...): (line 1157)
        # Processing the call arguments (line 1157)
        # Getting the type of 'other' (line 1157)
        other_122333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 27), 'other', False)
        # Processing the call keyword arguments (line 1157)
        kwargs_122334 = {}
        # Getting the type of 'poly1d' (line 1157)
        poly1d_122332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 20), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1157)
        poly1d_call_result_122335 = invoke(stypy.reporting.localization.Localization(__file__, 1157, 20), poly1d_122332, *[other_122333], **kwargs_122334)
        
        # Assigning a type to the variable 'other' (line 1157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 12), 'other', poly1d_call_result_122335)
        
        # Call to poly1d(...): (line 1158)
        # Processing the call arguments (line 1158)
        
        # Call to polymul(...): (line 1158)
        # Processing the call arguments (line 1158)
        # Getting the type of 'self' (line 1158)
        self_122338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 34), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1158)
        coeffs_122339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 34), self_122338, 'coeffs')
        # Getting the type of 'other' (line 1158)
        other_122340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 47), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1158)
        coeffs_122341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 47), other_122340, 'coeffs')
        # Processing the call keyword arguments (line 1158)
        kwargs_122342 = {}
        # Getting the type of 'polymul' (line 1158)
        polymul_122337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 26), 'polymul', False)
        # Calling polymul(args, kwargs) (line 1158)
        polymul_call_result_122343 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 26), polymul_122337, *[coeffs_122339, coeffs_122341], **kwargs_122342)
        
        # Processing the call keyword arguments (line 1158)
        kwargs_122344 = {}
        # Getting the type of 'poly1d' (line 1158)
        poly1d_122336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 19), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1158)
        poly1d_call_result_122345 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 19), poly1d_122336, *[polymul_call_result_122343], **kwargs_122344)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 12), 'stypy_return_type', poly1d_call_result_122345)
        # SSA join for if statement (line 1154)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 1153)
        stypy_return_type_122346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122346)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_122346


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 1160, 4, False)
        # Assigning a type to the variable 'self' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__add__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__add__.__dict__.__setitem__('stypy_function_name', 'poly1d.__add__')
        poly1d.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        # Assigning a Call to a Name (line 1161):
        
        # Assigning a Call to a Name (line 1161):
        
        # Call to poly1d(...): (line 1161)
        # Processing the call arguments (line 1161)
        # Getting the type of 'other' (line 1161)
        other_122348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 23), 'other', False)
        # Processing the call keyword arguments (line 1161)
        kwargs_122349 = {}
        # Getting the type of 'poly1d' (line 1161)
        poly1d_122347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 16), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1161)
        poly1d_call_result_122350 = invoke(stypy.reporting.localization.Localization(__file__, 1161, 16), poly1d_122347, *[other_122348], **kwargs_122349)
        
        # Assigning a type to the variable 'other' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 8), 'other', poly1d_call_result_122350)
        
        # Call to poly1d(...): (line 1162)
        # Processing the call arguments (line 1162)
        
        # Call to polyadd(...): (line 1162)
        # Processing the call arguments (line 1162)
        # Getting the type of 'self' (line 1162)
        self_122353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1162)
        coeffs_122354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 30), self_122353, 'coeffs')
        # Getting the type of 'other' (line 1162)
        other_122355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 43), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1162)
        coeffs_122356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 43), other_122355, 'coeffs')
        # Processing the call keyword arguments (line 1162)
        kwargs_122357 = {}
        # Getting the type of 'polyadd' (line 1162)
        polyadd_122352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 22), 'polyadd', False)
        # Calling polyadd(args, kwargs) (line 1162)
        polyadd_call_result_122358 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 22), polyadd_122352, *[coeffs_122354, coeffs_122356], **kwargs_122357)
        
        # Processing the call keyword arguments (line 1162)
        kwargs_122359 = {}
        # Getting the type of 'poly1d' (line 1162)
        poly1d_122351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1162)
        poly1d_call_result_122360 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 15), poly1d_122351, *[polyadd_call_result_122358], **kwargs_122359)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 8), 'stypy_return_type', poly1d_call_result_122360)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 1160)
        stypy_return_type_122361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_122361


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 1164, 4, False)
        # Assigning a type to the variable 'self' (line 1165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__radd__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__radd__.__dict__.__setitem__('stypy_function_name', 'poly1d.__radd__')
        poly1d.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__radd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__radd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__radd__(...)' code ##################

        
        # Assigning a Call to a Name (line 1165):
        
        # Assigning a Call to a Name (line 1165):
        
        # Call to poly1d(...): (line 1165)
        # Processing the call arguments (line 1165)
        # Getting the type of 'other' (line 1165)
        other_122363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 23), 'other', False)
        # Processing the call keyword arguments (line 1165)
        kwargs_122364 = {}
        # Getting the type of 'poly1d' (line 1165)
        poly1d_122362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 16), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1165)
        poly1d_call_result_122365 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 16), poly1d_122362, *[other_122363], **kwargs_122364)
        
        # Assigning a type to the variable 'other' (line 1165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 8), 'other', poly1d_call_result_122365)
        
        # Call to poly1d(...): (line 1166)
        # Processing the call arguments (line 1166)
        
        # Call to polyadd(...): (line 1166)
        # Processing the call arguments (line 1166)
        # Getting the type of 'self' (line 1166)
        self_122368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1166)
        coeffs_122369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 30), self_122368, 'coeffs')
        # Getting the type of 'other' (line 1166)
        other_122370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 43), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1166)
        coeffs_122371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 43), other_122370, 'coeffs')
        # Processing the call keyword arguments (line 1166)
        kwargs_122372 = {}
        # Getting the type of 'polyadd' (line 1166)
        polyadd_122367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 22), 'polyadd', False)
        # Calling polyadd(args, kwargs) (line 1166)
        polyadd_call_result_122373 = invoke(stypy.reporting.localization.Localization(__file__, 1166, 22), polyadd_122367, *[coeffs_122369, coeffs_122371], **kwargs_122372)
        
        # Processing the call keyword arguments (line 1166)
        kwargs_122374 = {}
        # Getting the type of 'poly1d' (line 1166)
        poly1d_122366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1166)
        poly1d_call_result_122375 = invoke(stypy.reporting.localization.Localization(__file__, 1166, 15), poly1d_122366, *[polyadd_call_result_122373], **kwargs_122374)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1166, 8), 'stypy_return_type', poly1d_call_result_122375)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 1164)
        stypy_return_type_122376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_122376


    @norecursion
    def __pow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pow__'
        module_type_store = module_type_store.open_function_context('__pow__', 1168, 4, False)
        # Assigning a type to the variable 'self' (line 1169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__pow__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__pow__.__dict__.__setitem__('stypy_function_name', 'poly1d.__pow__')
        poly1d.__pow__.__dict__.__setitem__('stypy_param_names_list', ['val'])
        poly1d.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__pow__', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pow__', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pow__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        
        # Call to isscalar(...): (line 1169)
        # Processing the call arguments (line 1169)
        # Getting the type of 'val' (line 1169)
        val_122378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 24), 'val', False)
        # Processing the call keyword arguments (line 1169)
        kwargs_122379 = {}
        # Getting the type of 'isscalar' (line 1169)
        isscalar_122377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 15), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 1169)
        isscalar_call_result_122380 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 15), isscalar_122377, *[val_122378], **kwargs_122379)
        
        # Applying the 'not' unary operator (line 1169)
        result_not__122381 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 11), 'not', isscalar_call_result_122380)
        
        
        
        # Call to int(...): (line 1169)
        # Processing the call arguments (line 1169)
        # Getting the type of 'val' (line 1169)
        val_122383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 36), 'val', False)
        # Processing the call keyword arguments (line 1169)
        kwargs_122384 = {}
        # Getting the type of 'int' (line 1169)
        int_122382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 32), 'int', False)
        # Calling int(args, kwargs) (line 1169)
        int_call_result_122385 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 32), int_122382, *[val_122383], **kwargs_122384)
        
        # Getting the type of 'val' (line 1169)
        val_122386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 44), 'val')
        # Applying the binary operator '!=' (line 1169)
        result_ne_122387 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 32), '!=', int_call_result_122385, val_122386)
        
        # Applying the binary operator 'or' (line 1169)
        result_or_keyword_122388 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 11), 'or', result_not__122381, result_ne_122387)
        
        # Getting the type of 'val' (line 1169)
        val_122389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 51), 'val')
        int_122390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, 57), 'int')
        # Applying the binary operator '<' (line 1169)
        result_lt_122391 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 51), '<', val_122389, int_122390)
        
        # Applying the binary operator 'or' (line 1169)
        result_or_keyword_122392 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 11), 'or', result_or_keyword_122388, result_lt_122391)
        
        # Testing the type of an if condition (line 1169)
        if_condition_122393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1169, 8), result_or_keyword_122392)
        # Assigning a type to the variable 'if_condition_122393' (line 1169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 8), 'if_condition_122393', if_condition_122393)
        # SSA begins for if statement (line 1169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1170)
        # Processing the call arguments (line 1170)
        str_122395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1170, 29), 'str', 'Power to non-negative integers only.')
        # Processing the call keyword arguments (line 1170)
        kwargs_122396 = {}
        # Getting the type of 'ValueError' (line 1170)
        ValueError_122394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1170)
        ValueError_call_result_122397 = invoke(stypy.reporting.localization.Localization(__file__, 1170, 18), ValueError_122394, *[str_122395], **kwargs_122396)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1170, 12), ValueError_call_result_122397, 'raise parameter', BaseException)
        # SSA join for if statement (line 1169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 1171):
        
        # Assigning a List to a Name (line 1171):
        
        # Obtaining an instance of the builtin type 'list' (line 1171)
        list_122398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1171)
        # Adding element type (line 1171)
        int_122399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1171, 14), list_122398, int_122399)
        
        # Assigning a type to the variable 'res' (line 1171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 8), 'res', list_122398)
        
        
        # Call to range(...): (line 1172)
        # Processing the call arguments (line 1172)
        # Getting the type of 'val' (line 1172)
        val_122401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 23), 'val', False)
        # Processing the call keyword arguments (line 1172)
        kwargs_122402 = {}
        # Getting the type of 'range' (line 1172)
        range_122400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 17), 'range', False)
        # Calling range(args, kwargs) (line 1172)
        range_call_result_122403 = invoke(stypy.reporting.localization.Localization(__file__, 1172, 17), range_122400, *[val_122401], **kwargs_122402)
        
        # Testing the type of a for loop iterable (line 1172)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1172, 8), range_call_result_122403)
        # Getting the type of the for loop variable (line 1172)
        for_loop_var_122404 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1172, 8), range_call_result_122403)
        # Assigning a type to the variable '_' (line 1172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 8), '_', for_loop_var_122404)
        # SSA begins for a for statement (line 1172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 1173):
        
        # Assigning a Call to a Name (line 1173):
        
        # Call to polymul(...): (line 1173)
        # Processing the call arguments (line 1173)
        # Getting the type of 'self' (line 1173)
        self_122406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 26), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1173)
        coeffs_122407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1173, 26), self_122406, 'coeffs')
        # Getting the type of 'res' (line 1173)
        res_122408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 39), 'res', False)
        # Processing the call keyword arguments (line 1173)
        kwargs_122409 = {}
        # Getting the type of 'polymul' (line 1173)
        polymul_122405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 18), 'polymul', False)
        # Calling polymul(args, kwargs) (line 1173)
        polymul_call_result_122410 = invoke(stypy.reporting.localization.Localization(__file__, 1173, 18), polymul_122405, *[coeffs_122407, res_122408], **kwargs_122409)
        
        # Assigning a type to the variable 'res' (line 1173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 12), 'res', polymul_call_result_122410)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to poly1d(...): (line 1174)
        # Processing the call arguments (line 1174)
        # Getting the type of 'res' (line 1174)
        res_122412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 22), 'res', False)
        # Processing the call keyword arguments (line 1174)
        kwargs_122413 = {}
        # Getting the type of 'poly1d' (line 1174)
        poly1d_122411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1174)
        poly1d_call_result_122414 = invoke(stypy.reporting.localization.Localization(__file__, 1174, 15), poly1d_122411, *[res_122412], **kwargs_122413)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 8), 'stypy_return_type', poly1d_call_result_122414)
        
        # ################# End of '__pow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pow__' in the type store
        # Getting the type of 'stypy_return_type' (line 1168)
        stypy_return_type_122415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122415)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pow__'
        return stypy_return_type_122415


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 1176, 4, False)
        # Assigning a type to the variable 'self' (line 1177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__sub__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__sub__.__dict__.__setitem__('stypy_function_name', 'poly1d.__sub__')
        poly1d.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        
        # Assigning a Call to a Name (line 1177):
        
        # Assigning a Call to a Name (line 1177):
        
        # Call to poly1d(...): (line 1177)
        # Processing the call arguments (line 1177)
        # Getting the type of 'other' (line 1177)
        other_122417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 23), 'other', False)
        # Processing the call keyword arguments (line 1177)
        kwargs_122418 = {}
        # Getting the type of 'poly1d' (line 1177)
        poly1d_122416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 16), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1177)
        poly1d_call_result_122419 = invoke(stypy.reporting.localization.Localization(__file__, 1177, 16), poly1d_122416, *[other_122417], **kwargs_122418)
        
        # Assigning a type to the variable 'other' (line 1177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 8), 'other', poly1d_call_result_122419)
        
        # Call to poly1d(...): (line 1178)
        # Processing the call arguments (line 1178)
        
        # Call to polysub(...): (line 1178)
        # Processing the call arguments (line 1178)
        # Getting the type of 'self' (line 1178)
        self_122422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1178)
        coeffs_122423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1178, 30), self_122422, 'coeffs')
        # Getting the type of 'other' (line 1178)
        other_122424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 43), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1178)
        coeffs_122425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1178, 43), other_122424, 'coeffs')
        # Processing the call keyword arguments (line 1178)
        kwargs_122426 = {}
        # Getting the type of 'polysub' (line 1178)
        polysub_122421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 22), 'polysub', False)
        # Calling polysub(args, kwargs) (line 1178)
        polysub_call_result_122427 = invoke(stypy.reporting.localization.Localization(__file__, 1178, 22), polysub_122421, *[coeffs_122423, coeffs_122425], **kwargs_122426)
        
        # Processing the call keyword arguments (line 1178)
        kwargs_122428 = {}
        # Getting the type of 'poly1d' (line 1178)
        poly1d_122420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1178)
        poly1d_call_result_122429 = invoke(stypy.reporting.localization.Localization(__file__, 1178, 15), poly1d_122420, *[polysub_call_result_122427], **kwargs_122428)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1178, 8), 'stypy_return_type', poly1d_call_result_122429)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 1176)
        stypy_return_type_122430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_122430


    @norecursion
    def __rsub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rsub__'
        module_type_store = module_type_store.open_function_context('__rsub__', 1180, 4, False)
        # Assigning a type to the variable 'self' (line 1181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__rsub__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__rsub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__rsub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__rsub__.__dict__.__setitem__('stypy_function_name', 'poly1d.__rsub__')
        poly1d.__rsub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__rsub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__rsub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__rsub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__rsub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__rsub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__rsub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__rsub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rsub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rsub__(...)' code ##################

        
        # Assigning a Call to a Name (line 1181):
        
        # Assigning a Call to a Name (line 1181):
        
        # Call to poly1d(...): (line 1181)
        # Processing the call arguments (line 1181)
        # Getting the type of 'other' (line 1181)
        other_122432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 23), 'other', False)
        # Processing the call keyword arguments (line 1181)
        kwargs_122433 = {}
        # Getting the type of 'poly1d' (line 1181)
        poly1d_122431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 16), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1181)
        poly1d_call_result_122434 = invoke(stypy.reporting.localization.Localization(__file__, 1181, 16), poly1d_122431, *[other_122432], **kwargs_122433)
        
        # Assigning a type to the variable 'other' (line 1181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 8), 'other', poly1d_call_result_122434)
        
        # Call to poly1d(...): (line 1182)
        # Processing the call arguments (line 1182)
        
        # Call to polysub(...): (line 1182)
        # Processing the call arguments (line 1182)
        # Getting the type of 'other' (line 1182)
        other_122437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 30), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1182)
        coeffs_122438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1182, 30), other_122437, 'coeffs')
        # Getting the type of 'self' (line 1182)
        self_122439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 44), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1182)
        coeffs_122440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1182, 44), self_122439, 'coeffs')
        # Processing the call keyword arguments (line 1182)
        kwargs_122441 = {}
        # Getting the type of 'polysub' (line 1182)
        polysub_122436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 22), 'polysub', False)
        # Calling polysub(args, kwargs) (line 1182)
        polysub_call_result_122442 = invoke(stypy.reporting.localization.Localization(__file__, 1182, 22), polysub_122436, *[coeffs_122438, coeffs_122440], **kwargs_122441)
        
        # Processing the call keyword arguments (line 1182)
        kwargs_122443 = {}
        # Getting the type of 'poly1d' (line 1182)
        poly1d_122435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1182)
        poly1d_call_result_122444 = invoke(stypy.reporting.localization.Localization(__file__, 1182, 15), poly1d_122435, *[polysub_call_result_122442], **kwargs_122443)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 8), 'stypy_return_type', poly1d_call_result_122444)
        
        # ################# End of '__rsub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rsub__' in the type store
        # Getting the type of 'stypy_return_type' (line 1180)
        stypy_return_type_122445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122445)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rsub__'
        return stypy_return_type_122445


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 1184, 4, False)
        # Assigning a type to the variable 'self' (line 1185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__div__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__div__.__dict__.__setitem__('stypy_function_name', 'poly1d.__div__')
        poly1d.__div__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__div__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__div__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__div__(...)' code ##################

        
        
        # Call to isscalar(...): (line 1185)
        # Processing the call arguments (line 1185)
        # Getting the type of 'other' (line 1185)
        other_122447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 20), 'other', False)
        # Processing the call keyword arguments (line 1185)
        kwargs_122448 = {}
        # Getting the type of 'isscalar' (line 1185)
        isscalar_122446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 11), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 1185)
        isscalar_call_result_122449 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 11), isscalar_122446, *[other_122447], **kwargs_122448)
        
        # Testing the type of an if condition (line 1185)
        if_condition_122450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1185, 8), isscalar_call_result_122449)
        # Assigning a type to the variable 'if_condition_122450' (line 1185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 8), 'if_condition_122450', if_condition_122450)
        # SSA begins for if statement (line 1185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to poly1d(...): (line 1186)
        # Processing the call arguments (line 1186)
        # Getting the type of 'self' (line 1186)
        self_122452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 26), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1186)
        coeffs_122453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 26), self_122452, 'coeffs')
        # Getting the type of 'other' (line 1186)
        other_122454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 38), 'other', False)
        # Applying the binary operator 'div' (line 1186)
        result_div_122455 = python_operator(stypy.reporting.localization.Localization(__file__, 1186, 26), 'div', coeffs_122453, other_122454)
        
        # Processing the call keyword arguments (line 1186)
        kwargs_122456 = {}
        # Getting the type of 'poly1d' (line 1186)
        poly1d_122451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 19), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1186)
        poly1d_call_result_122457 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 19), poly1d_122451, *[result_div_122455], **kwargs_122456)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 12), 'stypy_return_type', poly1d_call_result_122457)
        # SSA branch for the else part of an if statement (line 1185)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1188):
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to poly1d(...): (line 1188)
        # Processing the call arguments (line 1188)
        # Getting the type of 'other' (line 1188)
        other_122459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 27), 'other', False)
        # Processing the call keyword arguments (line 1188)
        kwargs_122460 = {}
        # Getting the type of 'poly1d' (line 1188)
        poly1d_122458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 20), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1188)
        poly1d_call_result_122461 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 20), poly1d_122458, *[other_122459], **kwargs_122460)
        
        # Assigning a type to the variable 'other' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 12), 'other', poly1d_call_result_122461)
        
        # Call to polydiv(...): (line 1189)
        # Processing the call arguments (line 1189)
        # Getting the type of 'self' (line 1189)
        self_122463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 27), 'self', False)
        # Getting the type of 'other' (line 1189)
        other_122464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 33), 'other', False)
        # Processing the call keyword arguments (line 1189)
        kwargs_122465 = {}
        # Getting the type of 'polydiv' (line 1189)
        polydiv_122462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 19), 'polydiv', False)
        # Calling polydiv(args, kwargs) (line 1189)
        polydiv_call_result_122466 = invoke(stypy.reporting.localization.Localization(__file__, 1189, 19), polydiv_122462, *[self_122463, other_122464], **kwargs_122465)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 12), 'stypy_return_type', polydiv_call_result_122466)
        # SSA join for if statement (line 1185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 1184)
        stypy_return_type_122467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_122467

    
    # Assigning a Name to a Name (line 1191):

    @norecursion
    def __rdiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdiv__'
        module_type_store = module_type_store.open_function_context('__rdiv__', 1193, 4, False)
        # Assigning a type to the variable 'self' (line 1194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__rdiv__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_function_name', 'poly1d.__rdiv__')
        poly1d.__rdiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__rdiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__rdiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__rdiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdiv__(...)' code ##################

        
        
        # Call to isscalar(...): (line 1194)
        # Processing the call arguments (line 1194)
        # Getting the type of 'other' (line 1194)
        other_122469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 20), 'other', False)
        # Processing the call keyword arguments (line 1194)
        kwargs_122470 = {}
        # Getting the type of 'isscalar' (line 1194)
        isscalar_122468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 11), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 1194)
        isscalar_call_result_122471 = invoke(stypy.reporting.localization.Localization(__file__, 1194, 11), isscalar_122468, *[other_122469], **kwargs_122470)
        
        # Testing the type of an if condition (line 1194)
        if_condition_122472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1194, 8), isscalar_call_result_122471)
        # Assigning a type to the variable 'if_condition_122472' (line 1194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 8), 'if_condition_122472', if_condition_122472)
        # SSA begins for if statement (line 1194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to poly1d(...): (line 1195)
        # Processing the call arguments (line 1195)
        # Getting the type of 'other' (line 1195)
        other_122474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 26), 'other', False)
        # Getting the type of 'self' (line 1195)
        self_122475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 32), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1195)
        coeffs_122476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1195, 32), self_122475, 'coeffs')
        # Applying the binary operator 'div' (line 1195)
        result_div_122477 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 26), 'div', other_122474, coeffs_122476)
        
        # Processing the call keyword arguments (line 1195)
        kwargs_122478 = {}
        # Getting the type of 'poly1d' (line 1195)
        poly1d_122473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 19), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1195)
        poly1d_call_result_122479 = invoke(stypy.reporting.localization.Localization(__file__, 1195, 19), poly1d_122473, *[result_div_122477], **kwargs_122478)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 12), 'stypy_return_type', poly1d_call_result_122479)
        # SSA branch for the else part of an if statement (line 1194)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1197):
        
        # Assigning a Call to a Name (line 1197):
        
        # Call to poly1d(...): (line 1197)
        # Processing the call arguments (line 1197)
        # Getting the type of 'other' (line 1197)
        other_122481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 27), 'other', False)
        # Processing the call keyword arguments (line 1197)
        kwargs_122482 = {}
        # Getting the type of 'poly1d' (line 1197)
        poly1d_122480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 20), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1197)
        poly1d_call_result_122483 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 20), poly1d_122480, *[other_122481], **kwargs_122482)
        
        # Assigning a type to the variable 'other' (line 1197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 12), 'other', poly1d_call_result_122483)
        
        # Call to polydiv(...): (line 1198)
        # Processing the call arguments (line 1198)
        # Getting the type of 'other' (line 1198)
        other_122485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 27), 'other', False)
        # Getting the type of 'self' (line 1198)
        self_122486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 34), 'self', False)
        # Processing the call keyword arguments (line 1198)
        kwargs_122487 = {}
        # Getting the type of 'polydiv' (line 1198)
        polydiv_122484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 19), 'polydiv', False)
        # Calling polydiv(args, kwargs) (line 1198)
        polydiv_call_result_122488 = invoke(stypy.reporting.localization.Localization(__file__, 1198, 19), polydiv_122484, *[other_122485, self_122486], **kwargs_122487)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1198, 12), 'stypy_return_type', polydiv_call_result_122488)
        # SSA join for if statement (line 1194)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__rdiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 1193)
        stypy_return_type_122489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdiv__'
        return stypy_return_type_122489

    
    # Assigning a Name to a Name (line 1200):

    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 1202, 4, False)
        # Assigning a type to the variable 'self' (line 1203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'poly1d.__eq__')
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        
        # Getting the type of 'self' (line 1203)
        self_122490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 11), 'self')
        # Obtaining the member 'coeffs' of a type (line 1203)
        coeffs_122491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 11), self_122490, 'coeffs')
        # Obtaining the member 'shape' of a type (line 1203)
        shape_122492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 11), coeffs_122491, 'shape')
        # Getting the type of 'other' (line 1203)
        other_122493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 32), 'other')
        # Obtaining the member 'coeffs' of a type (line 1203)
        coeffs_122494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 32), other_122493, 'coeffs')
        # Obtaining the member 'shape' of a type (line 1203)
        shape_122495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 32), coeffs_122494, 'shape')
        # Applying the binary operator '!=' (line 1203)
        result_ne_122496 = python_operator(stypy.reporting.localization.Localization(__file__, 1203, 11), '!=', shape_122492, shape_122495)
        
        # Testing the type of an if condition (line 1203)
        if_condition_122497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1203, 8), result_ne_122496)
        # Assigning a type to the variable 'if_condition_122497' (line 1203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1203, 8), 'if_condition_122497', if_condition_122497)
        # SSA begins for if statement (line 1203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 1204)
        False_122498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 1204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 12), 'stypy_return_type', False_122498)
        # SSA join for if statement (line 1203)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to all(...): (line 1205)
        # Processing the call keyword arguments (line 1205)
        kwargs_122505 = {}
        
        # Getting the type of 'self' (line 1205)
        self_122499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 16), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1205)
        coeffs_122500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 16), self_122499, 'coeffs')
        # Getting the type of 'other' (line 1205)
        other_122501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 31), 'other', False)
        # Obtaining the member 'coeffs' of a type (line 1205)
        coeffs_122502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 31), other_122501, 'coeffs')
        # Applying the binary operator '==' (line 1205)
        result_eq_122503 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 16), '==', coeffs_122500, coeffs_122502)
        
        # Obtaining the member 'all' of a type (line 1205)
        all_122504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 16), result_eq_122503, 'all')
        # Calling all(args, kwargs) (line 1205)
        all_call_result_122506 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 16), all_122504, *[], **kwargs_122505)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 8), 'stypy_return_type', all_call_result_122506)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 1202)
        stypy_return_type_122507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_122507


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 1207, 4, False)
        # Assigning a type to the variable 'self' (line 1208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__ne__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__ne__.__dict__.__setitem__('stypy_function_name', 'poly1d.__ne__')
        poly1d.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        poly1d.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        
        # Call to __eq__(...): (line 1208)
        # Processing the call arguments (line 1208)
        # Getting the type of 'other' (line 1208)
        other_122510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 31), 'other', False)
        # Processing the call keyword arguments (line 1208)
        kwargs_122511 = {}
        # Getting the type of 'self' (line 1208)
        self_122508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 19), 'self', False)
        # Obtaining the member '__eq__' of a type (line 1208)
        eq___122509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1208, 19), self_122508, '__eq__')
        # Calling __eq__(args, kwargs) (line 1208)
        eq___call_result_122512 = invoke(stypy.reporting.localization.Localization(__file__, 1208, 19), eq___122509, *[other_122510], **kwargs_122511)
        
        # Applying the 'not' unary operator (line 1208)
        result_not__122513 = python_operator(stypy.reporting.localization.Localization(__file__, 1208, 15), 'not', eq___call_result_122512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 8), 'stypy_return_type', result_not__122513)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 1207)
        stypy_return_type_122514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_122514


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 1210, 4, False)
        # Assigning a type to the variable 'self' (line 1211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__setattr__.__dict__.__setitem__('stypy_function_name', 'poly1d.__setattr__')
        poly1d.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['key', 'val'])
        poly1d.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__setattr__', ['key', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setattr__', localization, ['key', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setattr__(...)' code ##################

        
        # Call to ValueError(...): (line 1211)
        # Processing the call arguments (line 1211)
        str_122516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1211, 25), 'str', 'Attributes cannot be changed this way.')
        # Processing the call keyword arguments (line 1211)
        kwargs_122517 = {}
        # Getting the type of 'ValueError' (line 1211)
        ValueError_122515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1211)
        ValueError_call_result_122518 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 14), ValueError_122515, *[str_122516], **kwargs_122517)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1211, 8), ValueError_call_result_122518, 'raise parameter', BaseException)
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 1210)
        stypy_return_type_122519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_122519


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 1213, 4, False)
        # Assigning a type to the variable 'self' (line 1214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__getattr__.__dict__.__setitem__('stypy_function_name', 'poly1d.__getattr__')
        poly1d.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        poly1d.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__getattr__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        
        # Getting the type of 'key' (line 1214)
        key_122520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 11), 'key')
        
        # Obtaining an instance of the builtin type 'list' (line 1214)
        list_122521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1214)
        # Adding element type (line 1214)
        str_122522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 19), 'str', 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1214, 18), list_122521, str_122522)
        # Adding element type (line 1214)
        str_122523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 24), 'str', 'roots')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1214, 18), list_122521, str_122523)
        
        # Applying the binary operator 'in' (line 1214)
        result_contains_122524 = python_operator(stypy.reporting.localization.Localization(__file__, 1214, 11), 'in', key_122520, list_122521)
        
        # Testing the type of an if condition (line 1214)
        if_condition_122525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1214, 8), result_contains_122524)
        # Assigning a type to the variable 'if_condition_122525' (line 1214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 8), 'if_condition_122525', if_condition_122525)
        # SSA begins for if statement (line 1214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to roots(...): (line 1215)
        # Processing the call arguments (line 1215)
        # Getting the type of 'self' (line 1215)
        self_122527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 25), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1215)
        coeffs_122528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1215, 25), self_122527, 'coeffs')
        # Processing the call keyword arguments (line 1215)
        kwargs_122529 = {}
        # Getting the type of 'roots' (line 1215)
        roots_122526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 19), 'roots', False)
        # Calling roots(args, kwargs) (line 1215)
        roots_call_result_122530 = invoke(stypy.reporting.localization.Localization(__file__, 1215, 19), roots_122526, *[coeffs_122528], **kwargs_122529)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 12), 'stypy_return_type', roots_call_result_122530)
        # SSA branch for the else part of an if statement (line 1214)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'key' (line 1216)
        key_122531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 13), 'key')
        
        # Obtaining an instance of the builtin type 'list' (line 1216)
        list_122532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1216)
        # Adding element type (line 1216)
        str_122533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 21), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1216, 20), list_122532, str_122533)
        # Adding element type (line 1216)
        str_122534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 26), 'str', 'coef')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1216, 20), list_122532, str_122534)
        # Adding element type (line 1216)
        str_122535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 34), 'str', 'coefficients')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1216, 20), list_122532, str_122535)
        
        # Applying the binary operator 'in' (line 1216)
        result_contains_122536 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 13), 'in', key_122531, list_122532)
        
        # Testing the type of an if condition (line 1216)
        if_condition_122537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1216, 13), result_contains_122536)
        # Assigning a type to the variable 'if_condition_122537' (line 1216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 13), 'if_condition_122537', if_condition_122537)
        # SSA begins for if statement (line 1216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 1217)
        self_122538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 19), 'self')
        # Obtaining the member 'coeffs' of a type (line 1217)
        coeffs_122539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 19), self_122538, 'coeffs')
        # Assigning a type to the variable 'stypy_return_type' (line 1217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 12), 'stypy_return_type', coeffs_122539)
        # SSA branch for the else part of an if statement (line 1216)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'key' (line 1218)
        key_122540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 13), 'key')
        
        # Obtaining an instance of the builtin type 'list' (line 1218)
        list_122541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1218)
        # Adding element type (line 1218)
        str_122542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 21), 'str', 'o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1218, 20), list_122541, str_122542)
        
        # Applying the binary operator 'in' (line 1218)
        result_contains_122543 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 13), 'in', key_122540, list_122541)
        
        # Testing the type of an if condition (line 1218)
        if_condition_122544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1218, 13), result_contains_122543)
        # Assigning a type to the variable 'if_condition_122544' (line 1218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 13), 'if_condition_122544', if_condition_122544)
        # SSA begins for if statement (line 1218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 1219)
        self_122545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 19), 'self')
        # Obtaining the member 'order' of a type (line 1219)
        order_122546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 19), self_122545, 'order')
        # Assigning a type to the variable 'stypy_return_type' (line 1219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1219, 12), 'stypy_return_type', order_122546)
        # SSA branch for the else part of an if statement (line 1218)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 1221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 1222)
        key_122547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 37), 'key')
        # Getting the type of 'self' (line 1222)
        self_122548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 23), 'self')
        # Obtaining the member '__dict__' of a type (line 1222)
        dict___122549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1222, 23), self_122548, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 1222)
        getitem___122550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1222, 23), dict___122549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1222)
        subscript_call_result_122551 = invoke(stypy.reporting.localization.Localization(__file__, 1222, 23), getitem___122550, key_122547)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 16), 'stypy_return_type', subscript_call_result_122551)
        # SSA branch for the except part of a try statement (line 1221)
        # SSA branch for the except 'KeyError' branch of a try statement (line 1221)
        module_type_store.open_ssa_branch('except')
        
        # Call to AttributeError(...): (line 1224)
        # Processing the call arguments (line 1224)
        str_122553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 20), 'str', "'%s' has no attribute '%s'")
        
        # Obtaining an instance of the builtin type 'tuple' (line 1225)
        tuple_122554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1225)
        # Adding element type (line 1225)
        # Getting the type of 'self' (line 1225)
        self_122555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 52), 'self', False)
        # Obtaining the member '__class__' of a type (line 1225)
        class___122556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1225, 52), self_122555, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1225, 52), tuple_122554, class___122556)
        # Adding element type (line 1225)
        # Getting the type of 'key' (line 1225)
        key_122557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 68), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1225, 52), tuple_122554, key_122557)
        
        # Applying the binary operator '%' (line 1225)
        result_mod_122558 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 20), '%', str_122553, tuple_122554)
        
        # Processing the call keyword arguments (line 1224)
        kwargs_122559 = {}
        # Getting the type of 'AttributeError' (line 1224)
        AttributeError_122552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 22), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 1224)
        AttributeError_call_result_122560 = invoke(stypy.reporting.localization.Localization(__file__, 1224, 22), AttributeError_122552, *[result_mod_122558], **kwargs_122559)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1224, 16), AttributeError_call_result_122560, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 1221)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1218)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1216)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1214)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 1213)
        stypy_return_type_122561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122561)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_122561


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 1227, 4, False)
        # Assigning a type to the variable 'self' (line 1228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__getitem__.__dict__.__setitem__('stypy_function_name', 'poly1d.__getitem__')
        poly1d.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['val'])
        poly1d.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__getitem__', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a BinOp to a Name (line 1228):
        
        # Assigning a BinOp to a Name (line 1228):
        # Getting the type of 'self' (line 1228)
        self_122562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 14), 'self')
        # Obtaining the member 'order' of a type (line 1228)
        order_122563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1228, 14), self_122562, 'order')
        # Getting the type of 'val' (line 1228)
        val_122564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 27), 'val')
        # Applying the binary operator '-' (line 1228)
        result_sub_122565 = python_operator(stypy.reporting.localization.Localization(__file__, 1228, 14), '-', order_122563, val_122564)
        
        # Assigning a type to the variable 'ind' (line 1228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 8), 'ind', result_sub_122565)
        
        
        # Getting the type of 'val' (line 1229)
        val_122566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 11), 'val')
        # Getting the type of 'self' (line 1229)
        self_122567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 17), 'self')
        # Obtaining the member 'order' of a type (line 1229)
        order_122568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1229, 17), self_122567, 'order')
        # Applying the binary operator '>' (line 1229)
        result_gt_122569 = python_operator(stypy.reporting.localization.Localization(__file__, 1229, 11), '>', val_122566, order_122568)
        
        # Testing the type of an if condition (line 1229)
        if_condition_122570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1229, 8), result_gt_122569)
        # Assigning a type to the variable 'if_condition_122570' (line 1229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 8), 'if_condition_122570', if_condition_122570)
        # SSA begins for if statement (line 1229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_122571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 1230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 12), 'stypy_return_type', int_122571)
        # SSA join for if statement (line 1229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'val' (line 1231)
        val_122572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 11), 'val')
        int_122573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 17), 'int')
        # Applying the binary operator '<' (line 1231)
        result_lt_122574 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 11), '<', val_122572, int_122573)
        
        # Testing the type of an if condition (line 1231)
        if_condition_122575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1231, 8), result_lt_122574)
        # Assigning a type to the variable 'if_condition_122575' (line 1231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 8), 'if_condition_122575', if_condition_122575)
        # SSA begins for if statement (line 1231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_122576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 1232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 12), 'stypy_return_type', int_122576)
        # SSA join for if statement (line 1231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 1233)
        ind_122577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 27), 'ind')
        # Getting the type of 'self' (line 1233)
        self_122578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 15), 'self')
        # Obtaining the member 'coeffs' of a type (line 1233)
        coeffs_122579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1233, 15), self_122578, 'coeffs')
        # Obtaining the member '__getitem__' of a type (line 1233)
        getitem___122580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1233, 15), coeffs_122579, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1233)
        subscript_call_result_122581 = invoke(stypy.reporting.localization.Localization(__file__, 1233, 15), getitem___122580, ind_122577)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1233, 8), 'stypy_return_type', subscript_call_result_122581)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 1227)
        stypy_return_type_122582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_122582


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 1235, 4, False)
        # Assigning a type to the variable 'self' (line 1236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__setitem__.__dict__.__setitem__('stypy_function_name', 'poly1d.__setitem__')
        poly1d.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['key', 'val'])
        poly1d.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__setitem__', ['key', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['key', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        # Assigning a BinOp to a Name (line 1236):
        
        # Assigning a BinOp to a Name (line 1236):
        # Getting the type of 'self' (line 1236)
        self_122583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 14), 'self')
        # Obtaining the member 'order' of a type (line 1236)
        order_122584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 14), self_122583, 'order')
        # Getting the type of 'key' (line 1236)
        key_122585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 27), 'key')
        # Applying the binary operator '-' (line 1236)
        result_sub_122586 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 14), '-', order_122584, key_122585)
        
        # Assigning a type to the variable 'ind' (line 1236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 8), 'ind', result_sub_122586)
        
        
        # Getting the type of 'key' (line 1237)
        key_122587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 11), 'key')
        int_122588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1237, 17), 'int')
        # Applying the binary operator '<' (line 1237)
        result_lt_122589 = python_operator(stypy.reporting.localization.Localization(__file__, 1237, 11), '<', key_122587, int_122588)
        
        # Testing the type of an if condition (line 1237)
        if_condition_122590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1237, 8), result_lt_122589)
        # Assigning a type to the variable 'if_condition_122590' (line 1237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 8), 'if_condition_122590', if_condition_122590)
        # SSA begins for if statement (line 1237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1238)
        # Processing the call arguments (line 1238)
        str_122592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 29), 'str', 'Does not support negative powers.')
        # Processing the call keyword arguments (line 1238)
        kwargs_122593 = {}
        # Getting the type of 'ValueError' (line 1238)
        ValueError_122591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1238)
        ValueError_call_result_122594 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 18), ValueError_122591, *[str_122592], **kwargs_122593)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1238, 12), ValueError_call_result_122594, 'raise parameter', BaseException)
        # SSA join for if statement (line 1237)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'key' (line 1239)
        key_122595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 11), 'key')
        # Getting the type of 'self' (line 1239)
        self_122596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 17), 'self')
        # Obtaining the member 'order' of a type (line 1239)
        order_122597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 17), self_122596, 'order')
        # Applying the binary operator '>' (line 1239)
        result_gt_122598 = python_operator(stypy.reporting.localization.Localization(__file__, 1239, 11), '>', key_122595, order_122597)
        
        # Testing the type of an if condition (line 1239)
        if_condition_122599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1239, 8), result_gt_122598)
        # Assigning a type to the variable 'if_condition_122599' (line 1239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 8), 'if_condition_122599', if_condition_122599)
        # SSA begins for if statement (line 1239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1240):
        
        # Assigning a Call to a Name (line 1240):
        
        # Call to zeros(...): (line 1240)
        # Processing the call arguments (line 1240)
        # Getting the type of 'key' (line 1240)
        key_122602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 26), 'key', False)
        # Getting the type of 'self' (line 1240)
        self_122603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 30), 'self', False)
        # Obtaining the member 'order' of a type (line 1240)
        order_122604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 30), self_122603, 'order')
        # Applying the binary operator '-' (line 1240)
        result_sub_122605 = python_operator(stypy.reporting.localization.Localization(__file__, 1240, 26), '-', key_122602, order_122604)
        
        # Getting the type of 'self' (line 1240)
        self_122606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 42), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1240)
        coeffs_122607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 42), self_122606, 'coeffs')
        # Obtaining the member 'dtype' of a type (line 1240)
        dtype_122608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 42), coeffs_122607, 'dtype')
        # Processing the call keyword arguments (line 1240)
        kwargs_122609 = {}
        # Getting the type of 'NX' (line 1240)
        NX_122600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 17), 'NX', False)
        # Obtaining the member 'zeros' of a type (line 1240)
        zeros_122601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 17), NX_122600, 'zeros')
        # Calling zeros(args, kwargs) (line 1240)
        zeros_call_result_122610 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 17), zeros_122601, *[result_sub_122605, dtype_122608], **kwargs_122609)
        
        # Assigning a type to the variable 'zr' (line 1240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 12), 'zr', zeros_call_result_122610)
        
        # Assigning a Call to a Subscript (line 1241):
        
        # Assigning a Call to a Subscript (line 1241):
        
        # Call to concatenate(...): (line 1241)
        # Processing the call arguments (line 1241)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1241)
        tuple_122613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1241)
        # Adding element type (line 1241)
        # Getting the type of 'zr' (line 1241)
        zr_122614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 54), 'zr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1241, 54), tuple_122613, zr_122614)
        # Adding element type (line 1241)
        # Getting the type of 'self' (line 1241)
        self_122615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 58), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1241)
        coeffs_122616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 58), self_122615, 'coeffs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1241, 54), tuple_122613, coeffs_122616)
        
        # Processing the call keyword arguments (line 1241)
        kwargs_122617 = {}
        # Getting the type of 'NX' (line 1241)
        NX_122611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 38), 'NX', False)
        # Obtaining the member 'concatenate' of a type (line 1241)
        concatenate_122612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 38), NX_122611, 'concatenate')
        # Calling concatenate(args, kwargs) (line 1241)
        concatenate_call_result_122618 = invoke(stypy.reporting.localization.Localization(__file__, 1241, 38), concatenate_122612, *[tuple_122613], **kwargs_122617)
        
        # Getting the type of 'self' (line 1241)
        self_122619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 12), 'self')
        # Obtaining the member '__dict__' of a type (line 1241)
        dict___122620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 12), self_122619, '__dict__')
        str_122621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, 26), 'str', 'coeffs')
        # Storing an element on a container (line 1241)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1241, 12), dict___122620, (str_122621, concatenate_call_result_122618))
        
        # Assigning a Name to a Subscript (line 1242):
        
        # Assigning a Name to a Subscript (line 1242):
        # Getting the type of 'key' (line 1242)
        key_122622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 37), 'key')
        # Getting the type of 'self' (line 1242)
        self_122623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 12), 'self')
        # Obtaining the member '__dict__' of a type (line 1242)
        dict___122624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1242, 12), self_122623, '__dict__')
        str_122625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1242, 26), 'str', 'order')
        # Storing an element on a container (line 1242)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1242, 12), dict___122624, (str_122625, key_122622))
        
        # Assigning a Num to a Name (line 1243):
        
        # Assigning a Num to a Name (line 1243):
        int_122626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1243, 18), 'int')
        # Assigning a type to the variable 'ind' (line 1243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1243, 12), 'ind', int_122626)
        # SSA join for if statement (line 1239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 1244):
        
        # Assigning a Name to a Subscript (line 1244):
        # Getting the type of 'val' (line 1244)
        val_122627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 39), 'val')
        
        # Obtaining the type of the subscript
        str_122628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1244, 22), 'str', 'coeffs')
        # Getting the type of 'self' (line 1244)
        self_122629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 1244)
        dict___122630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 8), self_122629, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 1244)
        getitem___122631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 8), dict___122630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1244)
        subscript_call_result_122632 = invoke(stypy.reporting.localization.Localization(__file__, 1244, 8), getitem___122631, str_122628)
        
        # Getting the type of 'ind' (line 1244)
        ind_122633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 32), 'ind')
        # Storing an element on a container (line 1244)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1244, 8), subscript_call_result_122632, (ind_122633, val_122627))
        # Assigning a type to the variable 'stypy_return_type' (line 1245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1245, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 1235)
        stypy_return_type_122634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122634)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_122634


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 1247, 4, False)
        # Assigning a type to the variable 'self' (line 1248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.__iter__.__dict__.__setitem__('stypy_localization', localization)
        poly1d.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.__iter__.__dict__.__setitem__('stypy_function_name', 'poly1d.__iter__')
        poly1d.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        poly1d.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        # Call to iter(...): (line 1248)
        # Processing the call arguments (line 1248)
        # Getting the type of 'self' (line 1248)
        self_122636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 20), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1248)
        coeffs_122637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 20), self_122636, 'coeffs')
        # Processing the call keyword arguments (line 1248)
        kwargs_122638 = {}
        # Getting the type of 'iter' (line 1248)
        iter_122635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 15), 'iter', False)
        # Calling iter(args, kwargs) (line 1248)
        iter_call_result_122639 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 15), iter_122635, *[coeffs_122637], **kwargs_122638)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'stypy_return_type', iter_call_result_122639)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 1247)
        stypy_return_type_122640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_122640


    @norecursion
    def integ(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_122641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1250, 22), 'int')
        int_122642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1250, 27), 'int')
        defaults = [int_122641, int_122642]
        # Create a new context for function 'integ'
        module_type_store = module_type_store.open_function_context('integ', 1250, 4, False)
        # Assigning a type to the variable 'self' (line 1251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.integ.__dict__.__setitem__('stypy_localization', localization)
        poly1d.integ.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.integ.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.integ.__dict__.__setitem__('stypy_function_name', 'poly1d.integ')
        poly1d.integ.__dict__.__setitem__('stypy_param_names_list', ['m', 'k'])
        poly1d.integ.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.integ.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.integ.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.integ.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.integ.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.integ.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.integ', ['m', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integ', localization, ['m', 'k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integ(...)' code ##################

        str_122643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1260, (-1)), 'str', '\n        Return an antiderivative (indefinite integral) of this polynomial.\n\n        Refer to `polyint` for full documentation.\n\n        See Also\n        --------\n        polyint : equivalent function\n\n        ')
        
        # Call to poly1d(...): (line 1261)
        # Processing the call arguments (line 1261)
        
        # Call to polyint(...): (line 1261)
        # Processing the call arguments (line 1261)
        # Getting the type of 'self' (line 1261)
        self_122646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1261)
        coeffs_122647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 30), self_122646, 'coeffs')
        # Processing the call keyword arguments (line 1261)
        # Getting the type of 'm' (line 1261)
        m_122648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 45), 'm', False)
        keyword_122649 = m_122648
        # Getting the type of 'k' (line 1261)
        k_122650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 50), 'k', False)
        keyword_122651 = k_122650
        kwargs_122652 = {'k': keyword_122651, 'm': keyword_122649}
        # Getting the type of 'polyint' (line 1261)
        polyint_122645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 22), 'polyint', False)
        # Calling polyint(args, kwargs) (line 1261)
        polyint_call_result_122653 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 22), polyint_122645, *[coeffs_122647], **kwargs_122652)
        
        # Processing the call keyword arguments (line 1261)
        kwargs_122654 = {}
        # Getting the type of 'poly1d' (line 1261)
        poly1d_122644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1261)
        poly1d_call_result_122655 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 15), poly1d_122644, *[polyint_call_result_122653], **kwargs_122654)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 8), 'stypy_return_type', poly1d_call_result_122655)
        
        # ################# End of 'integ(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integ' in the type store
        # Getting the type of 'stypy_return_type' (line 1250)
        stypy_return_type_122656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122656)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integ'
        return stypy_return_type_122656


    @norecursion
    def deriv(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_122657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1263, 22), 'int')
        defaults = [int_122657]
        # Create a new context for function 'deriv'
        module_type_store = module_type_store.open_function_context('deriv', 1263, 4, False)
        # Assigning a type to the variable 'self' (line 1264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poly1d.deriv.__dict__.__setitem__('stypy_localization', localization)
        poly1d.deriv.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poly1d.deriv.__dict__.__setitem__('stypy_type_store', module_type_store)
        poly1d.deriv.__dict__.__setitem__('stypy_function_name', 'poly1d.deriv')
        poly1d.deriv.__dict__.__setitem__('stypy_param_names_list', ['m'])
        poly1d.deriv.__dict__.__setitem__('stypy_varargs_param_name', None)
        poly1d.deriv.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poly1d.deriv.__dict__.__setitem__('stypy_call_defaults', defaults)
        poly1d.deriv.__dict__.__setitem__('stypy_call_varargs', varargs)
        poly1d.deriv.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poly1d.deriv.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poly1d.deriv', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deriv', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deriv(...)' code ##################

        str_122658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1273, (-1)), 'str', '\n        Return a derivative of this polynomial.\n\n        Refer to `polyder` for full documentation.\n\n        See Also\n        --------\n        polyder : equivalent function\n\n        ')
        
        # Call to poly1d(...): (line 1274)
        # Processing the call arguments (line 1274)
        
        # Call to polyder(...): (line 1274)
        # Processing the call arguments (line 1274)
        # Getting the type of 'self' (line 1274)
        self_122661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 30), 'self', False)
        # Obtaining the member 'coeffs' of a type (line 1274)
        coeffs_122662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1274, 30), self_122661, 'coeffs')
        # Processing the call keyword arguments (line 1274)
        # Getting the type of 'm' (line 1274)
        m_122663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 45), 'm', False)
        keyword_122664 = m_122663
        kwargs_122665 = {'m': keyword_122664}
        # Getting the type of 'polyder' (line 1274)
        polyder_122660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 22), 'polyder', False)
        # Calling polyder(args, kwargs) (line 1274)
        polyder_call_result_122666 = invoke(stypy.reporting.localization.Localization(__file__, 1274, 22), polyder_122660, *[coeffs_122662], **kwargs_122665)
        
        # Processing the call keyword arguments (line 1274)
        kwargs_122667 = {}
        # Getting the type of 'poly1d' (line 1274)
        poly1d_122659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 15), 'poly1d', False)
        # Calling poly1d(args, kwargs) (line 1274)
        poly1d_call_result_122668 = invoke(stypy.reporting.localization.Localization(__file__, 1274, 15), poly1d_122659, *[polyder_call_result_122666], **kwargs_122667)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1274, 8), 'stypy_return_type', poly1d_call_result_122668)
        
        # ################# End of 'deriv(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deriv' in the type store
        # Getting the type of 'stypy_return_type' (line 1263)
        stypy_return_type_122669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deriv'
        return stypy_return_type_122669


# Assigning a type to the variable 'poly1d' (line 940)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 0), 'poly1d', poly1d)

# Assigning a Name to a Name (line 1040):
# Getting the type of 'None' (line 1040)
None_122670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 13), 'None')
# Getting the type of 'poly1d'
poly1d_122671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Setting the type of the member 'coeffs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122671, 'coeffs', None_122670)

# Assigning a Name to a Name (line 1041):
# Getting the type of 'None' (line 1041)
None_122672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 12), 'None')
# Getting the type of 'poly1d'
poly1d_122673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Setting the type of the member 'order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122673, 'order', None_122672)

# Assigning a Name to a Name (line 1042):
# Getting the type of 'None' (line 1042)
None_122674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 15), 'None')
# Getting the type of 'poly1d'
poly1d_122675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Setting the type of the member 'variable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122675, 'variable', None_122674)

# Assigning a Name to a Name (line 1043):
# Getting the type of 'None' (line 1043)
None_122676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 15), 'None')
# Getting the type of 'poly1d'
poly1d_122677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Setting the type of the member '__hash__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122677, '__hash__', None_122676)

# Assigning a Name to a Name (line 1191):
# Getting the type of 'poly1d'
poly1d_122678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Obtaining the member '__div__' of a type
div___122679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122678, '__div__')
# Getting the type of 'poly1d'
poly1d_122680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Setting the type of the member '__truediv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122680, '__truediv__', div___122679)

# Assigning a Name to a Name (line 1200):
# Getting the type of 'poly1d'
poly1d_122681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Obtaining the member '__rdiv__' of a type
rdiv___122682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122681, '__rdiv__')
# Getting the type of 'poly1d'
poly1d_122683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'poly1d')
# Setting the type of the member '__rtruediv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), poly1d_122683, '__rtruediv__', rdiv___122682)

# Call to simplefilter(...): (line 1278)
# Processing the call arguments (line 1278)
str_122686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1278, 22), 'str', 'always')
# Getting the type of 'RankWarning' (line 1278)
RankWarning_122687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1278, 32), 'RankWarning', False)
# Processing the call keyword arguments (line 1278)
kwargs_122688 = {}
# Getting the type of 'warnings' (line 1278)
warnings_122684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1278, 0), 'warnings', False)
# Obtaining the member 'simplefilter' of a type (line 1278)
simplefilter_122685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1278, 0), warnings_122684, 'simplefilter')
# Calling simplefilter(args, kwargs) (line 1278)
simplefilter_call_result_122689 = invoke(stypy.reporting.localization.Localization(__file__, 1278, 0), simplefilter_122685, *[str_122686, RankWarning_122687], **kwargs_122688)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
