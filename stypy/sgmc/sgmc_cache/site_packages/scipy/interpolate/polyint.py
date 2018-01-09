
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import warnings
4: 
5: import numpy as np
6: from scipy.special import factorial
7: 
8: from scipy._lib.six import xrange
9: from scipy._lib._util import _asarray_validated
10: 
11: 
12: __all__ = ["KroghInterpolator", "krogh_interpolate", "BarycentricInterpolator",
13:            "barycentric_interpolate", "approximate_taylor_polynomial"]
14: 
15: 
16: def _isscalar(x):
17:     '''Check whether x is if a scalar type, or 0-dim'''
18:     return np.isscalar(x) or hasattr(x, 'shape') and x.shape == ()
19: 
20: 
21: class _Interpolator1D(object):
22:     '''
23:     Common features in univariate interpolation
24: 
25:     Deal with input data type and interpolation axis rolling.  The
26:     actual interpolator can assume the y-data is of shape (n, r) where
27:     `n` is the number of x-points, and `r` the number of variables,
28:     and use self.dtype as the y-data type.
29: 
30:     Attributes
31:     ----------
32:     _y_axis
33:         Axis along which the interpolation goes in the original array
34:     _y_extra_shape
35:         Additional trailing shape of the input arrays, excluding
36:         the interpolation axis.
37:     dtype
38:         Dtype of the y-data arrays. Can be set via set_dtype, which
39:         forces it to be float or complex.
40: 
41:     Methods
42:     -------
43:     __call__
44:     _prepare_x
45:     _finish_y
46:     _reshape_yi
47:     _set_yi
48:     _set_dtype
49:     _evaluate
50: 
51:     '''
52: 
53:     __slots__ = ('_y_axis', '_y_extra_shape', 'dtype')
54: 
55:     def __init__(self, xi=None, yi=None, axis=None):
56:         self._y_axis = axis
57:         self._y_extra_shape = None
58:         self.dtype = None
59:         if yi is not None:
60:             self._set_yi(yi, xi=xi, axis=axis)
61: 
62:     def __call__(self, x):
63:         '''
64:         Evaluate the interpolant
65: 
66:         Parameters
67:         ----------
68:         x : array_like
69:             Points to evaluate the interpolant at.
70: 
71:         Returns
72:         -------
73:         y : array_like
74:             Interpolated values. Shape is determined by replacing
75:             the interpolation axis in the original array with the shape of x.
76: 
77:         '''
78:         x, x_shape = self._prepare_x(x)
79:         y = self._evaluate(x)
80:         return self._finish_y(y, x_shape)
81: 
82:     def _evaluate(self, x):
83:         '''
84:         Actually evaluate the value of the interpolator.
85:         '''
86:         raise NotImplementedError()
87: 
88:     def _prepare_x(self, x):
89:         '''Reshape input x array to 1-D'''
90:         x = _asarray_validated(x, check_finite=False, as_inexact=True)
91:         x_shape = x.shape
92:         return x.ravel(), x_shape
93: 
94:     def _finish_y(self, y, x_shape):
95:         '''Reshape interpolated y back to n-d array similar to initial y'''
96:         y = y.reshape(x_shape + self._y_extra_shape)
97:         if self._y_axis != 0 and x_shape != ():
98:             nx = len(x_shape)
99:             ny = len(self._y_extra_shape)
100:             s = (list(range(nx, nx + self._y_axis))
101:                  + list(range(nx)) + list(range(nx+self._y_axis, nx+ny)))
102:             y = y.transpose(s)
103:         return y
104: 
105:     def _reshape_yi(self, yi, check=False):
106:         yi = np.rollaxis(np.asarray(yi), self._y_axis)
107:         if check and yi.shape[1:] != self._y_extra_shape:
108:             ok_shape = "%r + (N,) + %r" % (self._y_extra_shape[-self._y_axis:],
109:                                            self._y_extra_shape[:-self._y_axis])
110:             raise ValueError("Data must be of shape %s" % ok_shape)
111:         return yi.reshape((yi.shape[0], -1))
112: 
113:     def _set_yi(self, yi, xi=None, axis=None):
114:         if axis is None:
115:             axis = self._y_axis
116:         if axis is None:
117:             raise ValueError("no interpolation axis specified")
118: 
119:         yi = np.asarray(yi)
120: 
121:         shape = yi.shape
122:         if shape == ():
123:             shape = (1,)
124:         if xi is not None and shape[axis] != len(xi):
125:             raise ValueError("x and y arrays must be equal in length along "
126:                              "interpolation axis.")
127: 
128:         self._y_axis = (axis % yi.ndim)
129:         self._y_extra_shape = yi.shape[:self._y_axis]+yi.shape[self._y_axis+1:]
130:         self.dtype = None
131:         self._set_dtype(yi.dtype)
132: 
133:     def _set_dtype(self, dtype, union=False):
134:         if np.issubdtype(dtype, np.complexfloating) \
135:                or np.issubdtype(self.dtype, np.complexfloating):
136:             self.dtype = np.complex_
137:         else:
138:             if not union or self.dtype != np.complex_:
139:                 self.dtype = np.float_
140: 
141: 
142: class _Interpolator1DWithDerivatives(_Interpolator1D):
143:     def derivatives(self, x, der=None):
144:         '''
145:         Evaluate many derivatives of the polynomial at the point x
146: 
147:         Produce an array of all derivative values at the point x.
148: 
149:         Parameters
150:         ----------
151:         x : array_like
152:             Point or points at which to evaluate the derivatives
153:         der : int or None, optional
154:             How many derivatives to extract; None for all potentially
155:             nonzero derivatives (that is a number equal to the number
156:             of points). This number includes the function value as 0th
157:             derivative.
158: 
159:         Returns
160:         -------
161:         d : ndarray
162:             Array with derivatives; d[j] contains the j-th derivative.
163:             Shape of d[j] is determined by replacing the interpolation
164:             axis in the original array with the shape of x.
165: 
166:         Examples
167:         --------
168:         >>> from scipy.interpolate import KroghInterpolator
169:         >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)
170:         array([1.0,2.0,3.0])
171:         >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])
172:         array([[1.0,1.0],
173:                [2.0,2.0],
174:                [3.0,3.0]])
175: 
176:         '''
177:         x, x_shape = self._prepare_x(x)
178:         y = self._evaluate_derivatives(x, der)
179: 
180:         y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
181:         if self._y_axis != 0 and x_shape != ():
182:             nx = len(x_shape)
183:             ny = len(self._y_extra_shape)
184:             s = ([0] + list(range(nx+1, nx + self._y_axis+1))
185:                  + list(range(1,nx+1)) +
186:                  list(range(nx+1+self._y_axis, nx+ny+1)))
187:             y = y.transpose(s)
188:         return y
189: 
190:     def derivative(self, x, der=1):
191:         '''
192:         Evaluate one derivative of the polynomial at the point x
193: 
194:         Parameters
195:         ----------
196:         x : array_like
197:             Point or points at which to evaluate the derivatives
198: 
199:         der : integer, optional
200:             Which derivative to extract. This number includes the
201:             function value as 0th derivative.
202: 
203:         Returns
204:         -------
205:         d : ndarray
206:             Derivative interpolated at the x-points.  Shape of d is
207:             determined by replacing the interpolation axis in the
208:             original array with the shape of x.
209: 
210:         Notes
211:         -----
212:         This is computed by evaluating all derivatives up to the desired
213:         one (using self.derivatives()) and then discarding the rest.
214: 
215:         '''
216:         x, x_shape = self._prepare_x(x)
217:         y = self._evaluate_derivatives(x, der+1)
218:         return self._finish_y(y[der], x_shape)
219: 
220: 
221: class KroghInterpolator(_Interpolator1DWithDerivatives):
222:     '''
223:     Interpolating polynomial for a set of points.
224: 
225:     The polynomial passes through all the pairs (xi,yi). One may
226:     additionally specify a number of derivatives at each point xi;
227:     this is done by repeating the value xi and specifying the
228:     derivatives as successive yi values.
229: 
230:     Allows evaluation of the polynomial and all its derivatives.
231:     For reasons of numerical stability, this function does not compute
232:     the coefficients of the polynomial, although they can be obtained
233:     by evaluating all the derivatives.
234: 
235:     Parameters
236:     ----------
237:     xi : array_like, length N
238:         Known x-coordinates. Must be sorted in increasing order.
239:     yi : array_like
240:         Known y-coordinates. When an xi occurs two or more times in
241:         a row, the corresponding yi's represent derivative values.
242:     axis : int, optional
243:         Axis in the yi array corresponding to the x-coordinate values.
244: 
245:     Notes
246:     -----
247:     Be aware that the algorithms implemented here are not necessarily
248:     the most numerically stable known. Moreover, even in a world of
249:     exact computation, unless the x coordinates are chosen very
250:     carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
251:     polynomial interpolation itself is a very ill-conditioned process
252:     due to the Runge phenomenon. In general, even with well-chosen
253:     x values, degrees higher than about thirty cause problems with
254:     numerical instability in this code.
255: 
256:     Based on [1]_.
257: 
258:     References
259:     ----------
260:     .. [1] Krogh, "Efficient Algorithms for Polynomial Interpolation
261:         and Numerical Differentiation", 1970.
262: 
263:     Examples
264:     --------
265:     To produce a polynomial that is zero at 0 and 1 and has
266:     derivative 2 at 0, call
267: 
268:     >>> from scipy.interpolate import KroghInterpolator
269:     >>> KroghInterpolator([0,0,1],[0,2,0])
270: 
271:     This constructs the quadratic 2*X**2-2*X. The derivative condition
272:     is indicated by the repeated zero in the xi array; the corresponding
273:     yi values are 0, the function value, and 2, the derivative value.
274: 
275:     For another example, given xi, yi, and a derivative ypi for each
276:     point, appropriate arrays can be constructed as:
277: 
278:     >>> xi = np.linspace(0, 1, 5)
279:     >>> yi, ypi = np.random.rand(2, 5)
280:     >>> xi_k, yi_k = np.repeat(xi, 2), np.ravel(np.dstack((yi,ypi)))
281:     >>> KroghInterpolator(xi_k, yi_k)
282: 
283:     To produce a vector-valued polynomial, supply a higher-dimensional
284:     array for yi:
285: 
286:     >>> KroghInterpolator([0,1],[[2,3],[4,5]])
287: 
288:     This constructs a linear polynomial giving (2,3) at 0 and (4,5) at 1.
289: 
290:     '''
291: 
292:     def __init__(self, xi, yi, axis=0):
293:         _Interpolator1DWithDerivatives.__init__(self, xi, yi, axis)
294: 
295:         self.xi = np.asarray(xi)
296:         self.yi = self._reshape_yi(yi)
297:         self.n, self.r = self.yi.shape
298: 
299:         c = np.zeros((self.n+1, self.r), dtype=self.dtype)
300:         c[0] = self.yi[0]
301:         Vk = np.zeros((self.n, self.r), dtype=self.dtype)
302:         for k in xrange(1,self.n):
303:             s = 0
304:             while s <= k and xi[k-s] == xi[k]:
305:                 s += 1
306:             s -= 1
307:             Vk[0] = self.yi[k]/float(factorial(s))
308:             for i in xrange(k-s):
309:                 if xi[i] == xi[k]:
310:                     raise ValueError("Elements if `xi` can't be equal.")
311:                 if s == 0:
312:                     Vk[i+1] = (c[i]-Vk[i])/(xi[i]-xi[k])
313:                 else:
314:                     Vk[i+1] = (Vk[i+1]-Vk[i])/(xi[i]-xi[k])
315:             c[k] = Vk[k-s]
316:         self.c = c
317: 
318:     def _evaluate(self, x):
319:         pi = 1
320:         p = np.zeros((len(x), self.r), dtype=self.dtype)
321:         p += self.c[0,np.newaxis,:]
322:         for k in range(1, self.n):
323:             w = x - self.xi[k-1]
324:             pi = w*pi
325:             p += pi[:,np.newaxis] * self.c[k]
326:         return p
327: 
328:     def _evaluate_derivatives(self, x, der=None):
329:         n = self.n
330:         r = self.r
331: 
332:         if der is None:
333:             der = self.n
334:         pi = np.zeros((n, len(x)))
335:         w = np.zeros((n, len(x)))
336:         pi[0] = 1
337:         p = np.zeros((len(x), self.r), dtype=self.dtype)
338:         p += self.c[0, np.newaxis, :]
339: 
340:         for k in xrange(1, n):
341:             w[k-1] = x - self.xi[k-1]
342:             pi[k] = w[k-1] * pi[k-1]
343:             p += pi[k, :, np.newaxis] * self.c[k]
344: 
345:         cn = np.zeros((max(der, n+1), len(x), r), dtype=self.dtype)
346:         cn[:n+1, :, :] += self.c[:n+1, np.newaxis, :]
347:         cn[0] = p
348:         for k in xrange(1, n):
349:             for i in xrange(1, n-k+1):
350:                 pi[i] = w[k+i-1]*pi[i-1] + pi[i]
351:                 cn[k] = cn[k] + pi[i, :, np.newaxis]*cn[k+i]
352:             cn[k] *= factorial(k)
353: 
354:         cn[n, :, :] = 0
355:         return cn[:der]
356: 
357: 
358: def krogh_interpolate(xi, yi, x, der=0, axis=0):
359:     '''
360:     Convenience function for polynomial interpolation.
361: 
362:     See `KroghInterpolator` for more details.
363: 
364:     Parameters
365:     ----------
366:     xi : array_like
367:         Known x-coordinates.
368:     yi : array_like
369:         Known y-coordinates, of shape ``(xi.size, R)``.  Interpreted as
370:         vectors of length R, or scalars if R=1.
371:     x : array_like
372:         Point or points at which to evaluate the derivatives.
373:     der : int or list, optional
374:         How many derivatives to extract; None for all potentially
375:         nonzero derivatives (that is a number equal to the number
376:         of points), or a list of derivatives to extract. This number
377:         includes the function value as 0th derivative.
378:     axis : int, optional
379:         Axis in the yi array corresponding to the x-coordinate values.
380: 
381:     Returns
382:     -------
383:     d : ndarray
384:         If the interpolator's values are R-dimensional then the
385:         returned array will be the number of derivatives by N by R.
386:         If `x` is a scalar, the middle dimension will be dropped; if
387:         the `yi` are scalars then the last dimension will be dropped.
388: 
389:     See Also
390:     --------
391:     KroghInterpolator
392: 
393:     Notes
394:     -----
395:     Construction of the interpolating polynomial is a relatively expensive
396:     process. If you want to evaluate it repeatedly consider using the class
397:     KroghInterpolator (which is what this function uses).
398: 
399:     '''
400:     P = KroghInterpolator(xi, yi, axis=axis)
401:     if der == 0:
402:         return P(x)
403:     elif _isscalar(der):
404:         return P.derivative(x,der=der)
405:     else:
406:         return P.derivatives(x,der=np.amax(der)+1)[der]
407: 
408: 
409: def approximate_taylor_polynomial(f,x,degree,scale,order=None):
410:     '''
411:     Estimate the Taylor polynomial of f at x by polynomial fitting.
412: 
413:     Parameters
414:     ----------
415:     f : callable
416:         The function whose Taylor polynomial is sought. Should accept
417:         a vector of `x` values.
418:     x : scalar
419:         The point at which the polynomial is to be evaluated.
420:     degree : int
421:         The degree of the Taylor polynomial
422:     scale : scalar
423:         The width of the interval to use to evaluate the Taylor polynomial.
424:         Function values spread over a range this wide are used to fit the
425:         polynomial. Must be chosen carefully.
426:     order : int or None, optional
427:         The order of the polynomial to be used in the fitting; `f` will be
428:         evaluated ``order+1`` times. If None, use `degree`.
429: 
430:     Returns
431:     -------
432:     p : poly1d instance
433:         The Taylor polynomial (translated to the origin, so that
434:         for example p(0)=f(x)).
435: 
436:     Notes
437:     -----
438:     The appropriate choice of "scale" is a trade-off; too large and the
439:     function differs from its Taylor polynomial too much to get a good
440:     answer, too small and round-off errors overwhelm the higher-order terms.
441:     The algorithm used becomes numerically unstable around order 30 even
442:     under ideal circumstances.
443: 
444:     Choosing order somewhat larger than degree may improve the higher-order
445:     terms.
446: 
447:     '''
448:     if order is None:
449:         order = degree
450: 
451:     n = order+1
452:     # Choose n points that cluster near the endpoints of the interval in
453:     # a way that avoids the Runge phenomenon. Ensure, by including the
454:     # endpoint or not as appropriate, that one point always falls at x
455:     # exactly.
456:     xs = scale*np.cos(np.linspace(0,np.pi,n,endpoint=n % 1)) + x
457: 
458:     P = KroghInterpolator(xs, f(xs))
459:     d = P.derivatives(x,der=degree+1)
460: 
461:     return np.poly1d((d/factorial(np.arange(degree+1)))[::-1])
462: 
463: 
464: class BarycentricInterpolator(_Interpolator1D):
465:     '''The interpolating polynomial for a set of points
466: 
467:     Constructs a polynomial that passes through a given set of points.
468:     Allows evaluation of the polynomial, efficient changing of the y
469:     values to be interpolated, and updating by adding more x values.
470:     For reasons of numerical stability, this function does not compute
471:     the coefficients of the polynomial.
472: 
473:     The values yi need to be provided before the function is
474:     evaluated, but none of the preprocessing depends on them, so rapid
475:     updates are possible.
476: 
477:     Parameters
478:     ----------
479:     xi : array_like
480:         1-d array of x coordinates of the points the polynomial
481:         should pass through
482:     yi : array_like, optional
483:         The y coordinates of the points the polynomial should pass through.
484:         If None, the y values will be supplied later via the `set_y` method.
485:     axis : int, optional
486:         Axis in the yi array corresponding to the x-coordinate values.
487: 
488:     Notes
489:     -----
490:     This class uses a "barycentric interpolation" method that treats
491:     the problem as a special case of rational function interpolation.
492:     This algorithm is quite stable, numerically, but even in a world of
493:     exact computation, unless the x coordinates are chosen very
494:     carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
495:     polynomial interpolation itself is a very ill-conditioned process
496:     due to the Runge phenomenon.
497: 
498:     Based on Berrut and Trefethen 2004, "Barycentric Lagrange Interpolation".
499: 
500:     '''
501:     def __init__(self, xi, yi=None, axis=0):
502:         _Interpolator1D.__init__(self, xi, yi, axis)
503: 
504:         self.xi = np.asarray(xi)
505:         self.set_yi(yi)
506:         self.n = len(self.xi)
507: 
508:         self.wi = np.zeros(self.n)
509:         self.wi[0] = 1
510:         for j in xrange(1,self.n):
511:             self.wi[:j] *= (self.xi[j]-self.xi[:j])
512:             self.wi[j] = np.multiply.reduce(self.xi[:j]-self.xi[j])
513:         self.wi **= -1
514: 
515:     def set_yi(self, yi, axis=None):
516:         '''
517:         Update the y values to be interpolated
518: 
519:         The barycentric interpolation algorithm requires the calculation
520:         of weights, but these depend only on the xi. The yi can be changed
521:         at any time.
522: 
523:         Parameters
524:         ----------
525:         yi : array_like
526:             The y coordinates of the points the polynomial should pass through.
527:             If None, the y values will be supplied later.
528:         axis : int, optional
529:             Axis in the yi array corresponding to the x-coordinate values.
530: 
531:         '''
532:         if yi is None:
533:             self.yi = None
534:             return
535:         self._set_yi(yi, xi=self.xi, axis=axis)
536:         self.yi = self._reshape_yi(yi)
537:         self.n, self.r = self.yi.shape
538: 
539:     def add_xi(self, xi, yi=None):
540:         '''
541:         Add more x values to the set to be interpolated
542: 
543:         The barycentric interpolation algorithm allows easy updating by
544:         adding more points for the polynomial to pass through.
545: 
546:         Parameters
547:         ----------
548:         xi : array_like
549:             The x coordinates of the points that the polynomial should pass
550:             through.
551:         yi : array_like, optional
552:             The y coordinates of the points the polynomial should pass through.
553:             Should have shape ``(xi.size, R)``; if R > 1 then the polynomial is
554:             vector-valued.
555:             If `yi` is not given, the y values will be supplied later. `yi` should
556:             be given if and only if the interpolator has y values specified.
557: 
558:         '''
559:         if yi is not None:
560:             if self.yi is None:
561:                 raise ValueError("No previous yi value to update!")
562:             yi = self._reshape_yi(yi, check=True)
563:             self.yi = np.vstack((self.yi,yi))
564:         else:
565:             if self.yi is not None:
566:                 raise ValueError("No update to yi provided!")
567:         old_n = self.n
568:         self.xi = np.concatenate((self.xi,xi))
569:         self.n = len(self.xi)
570:         self.wi **= -1
571:         old_wi = self.wi
572:         self.wi = np.zeros(self.n)
573:         self.wi[:old_n] = old_wi
574:         for j in xrange(old_n,self.n):
575:             self.wi[:j] *= (self.xi[j]-self.xi[:j])
576:             self.wi[j] = np.multiply.reduce(self.xi[:j]-self.xi[j])
577:         self.wi **= -1
578: 
579:     def __call__(self, x):
580:         '''Evaluate the interpolating polynomial at the points x
581: 
582:         Parameters
583:         ----------
584:         x : array_like
585:             Points to evaluate the interpolant at.
586: 
587:         Returns
588:         -------
589:         y : array_like
590:             Interpolated values. Shape is determined by replacing
591:             the interpolation axis in the original array with the shape of x.
592: 
593:         Notes
594:         -----
595:         Currently the code computes an outer product between x and the
596:         weights, that is, it constructs an intermediate array of size
597:         N by len(x), where N is the degree of the polynomial.
598:         '''
599:         return _Interpolator1D.__call__(self, x)
600: 
601:     def _evaluate(self, x):
602:         if x.size == 0:
603:             p = np.zeros((0, self.r), dtype=self.dtype)
604:         else:
605:             c = x[...,np.newaxis]-self.xi
606:             z = c == 0
607:             c[z] = 1
608:             c = self.wi/c
609:             p = np.dot(c,self.yi)/np.sum(c,axis=-1)[...,np.newaxis]
610:             # Now fix where x==some xi
611:             r = np.nonzero(z)
612:             if len(r) == 1:  # evaluation at a scalar
613:                 if len(r[0]) > 0:  # equals one of the points
614:                     p = self.yi[r[0][0]]
615:             else:
616:                 p[r[:-1]] = self.yi[r[-1]]
617:         return p
618: 
619: 
620: def barycentric_interpolate(xi, yi, x, axis=0):
621:     '''
622:     Convenience function for polynomial interpolation.
623: 
624:     Constructs a polynomial that passes through a given set of points,
625:     then evaluates the polynomial. For reasons of numerical stability,
626:     this function does not compute the coefficients of the polynomial.
627: 
628:     This function uses a "barycentric interpolation" method that treats
629:     the problem as a special case of rational function interpolation.
630:     This algorithm is quite stable, numerically, but even in a world of
631:     exact computation, unless the `x` coordinates are chosen very
632:     carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
633:     polynomial interpolation itself is a very ill-conditioned process
634:     due to the Runge phenomenon.
635: 
636:     Parameters
637:     ----------
638:     xi : array_like
639:         1-d array of x coordinates of the points the polynomial should
640:         pass through
641:     yi : array_like
642:         The y coordinates of the points the polynomial should pass through.
643:     x : scalar or array_like
644:         Points to evaluate the interpolator at.
645:     axis : int, optional
646:         Axis in the yi array corresponding to the x-coordinate values.
647: 
648:     Returns
649:     -------
650:     y : scalar or array_like
651:         Interpolated values. Shape is determined by replacing
652:         the interpolation axis in the original array with the shape of x.
653: 
654:     See Also
655:     --------
656:     BarycentricInterpolator
657: 
658:     Notes
659:     -----
660:     Construction of the interpolation weights is a relatively slow process.
661:     If you want to call this many times with the same xi (but possibly
662:     varying yi or x) you should use the class `BarycentricInterpolator`.
663:     This is what this function uses internally.
664: 
665:     '''
666:     return BarycentricInterpolator(xi, yi, axis=axis)(x)
667: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import warnings' statement (line 3)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71419 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_71419) is not StypyTypeError):

    if (import_71419 != 'pyd_module'):
        __import__(import_71419)
        sys_modules_71420 = sys.modules[import_71419]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_71420.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_71419)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special import factorial' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special')

if (type(import_71421) is not StypyTypeError):

    if (import_71421 != 'pyd_module'):
        __import__(import_71421)
        sys_modules_71422 = sys.modules[import_71421]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', sys_modules_71422.module_type_store, module_type_store, ['factorial'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_71422, sys_modules_71422.module_type_store, module_type_store)
    else:
        from scipy.special import factorial

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', None, module_type_store, ['factorial'], [factorial])

else:
    # Assigning a type to the variable 'scipy.special' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', import_71421)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib.six import xrange' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71423 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six')

if (type(import_71423) is not StypyTypeError):

    if (import_71423 != 'pyd_module'):
        __import__(import_71423)
        sys_modules_71424 = sys.modules[import_71423]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', sys_modules_71424.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_71424, sys_modules_71424.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', import_71423)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy._lib._util import _asarray_validated' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71425 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util')

if (type(import_71425) is not StypyTypeError):

    if (import_71425 != 'pyd_module'):
        __import__(import_71425)
        sys_modules_71426 = sys.modules[import_71425]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util', sys_modules_71426.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_71426, sys_modules_71426.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util', import_71425)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['KroghInterpolator', 'krogh_interpolate', 'BarycentricInterpolator', 'barycentric_interpolate', 'approximate_taylor_polynomial']
module_type_store.set_exportable_members(['KroghInterpolator', 'krogh_interpolate', 'BarycentricInterpolator', 'barycentric_interpolate', 'approximate_taylor_polynomial'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_71427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_71428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'KroghInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_71427, str_71428)
# Adding element type (line 12)
str_71429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'str', 'krogh_interpolate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_71427, str_71429)
# Adding element type (line 12)
str_71430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 53), 'str', 'BarycentricInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_71427, str_71430)
# Adding element type (line 12)
str_71431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'barycentric_interpolate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_71427, str_71431)
# Adding element type (line 12)
str_71432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 38), 'str', 'approximate_taylor_polynomial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_71427, str_71432)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_71427)

@norecursion
def _isscalar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_isscalar'
    module_type_store = module_type_store.open_function_context('_isscalar', 16, 0, False)
    
    # Passed parameters checking function
    _isscalar.stypy_localization = localization
    _isscalar.stypy_type_of_self = None
    _isscalar.stypy_type_store = module_type_store
    _isscalar.stypy_function_name = '_isscalar'
    _isscalar.stypy_param_names_list = ['x']
    _isscalar.stypy_varargs_param_name = None
    _isscalar.stypy_kwargs_param_name = None
    _isscalar.stypy_call_defaults = defaults
    _isscalar.stypy_call_varargs = varargs
    _isscalar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_isscalar', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_isscalar', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_isscalar(...)' code ##################

    str_71433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Check whether x is if a scalar type, or 0-dim')
    
    # Evaluating a boolean operation
    
    # Call to isscalar(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'x' (line 18)
    x_71436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'x', False)
    # Processing the call keyword arguments (line 18)
    kwargs_71437 = {}
    # Getting the type of 'np' (line 18)
    np_71434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 18)
    isscalar_71435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), np_71434, 'isscalar')
    # Calling isscalar(args, kwargs) (line 18)
    isscalar_call_result_71438 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), isscalar_71435, *[x_71436], **kwargs_71437)
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'x' (line 18)
    x_71440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 37), 'x', False)
    str_71441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 40), 'str', 'shape')
    # Processing the call keyword arguments (line 18)
    kwargs_71442 = {}
    # Getting the type of 'hasattr' (line 18)
    hasattr_71439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 18)
    hasattr_call_result_71443 = invoke(stypy.reporting.localization.Localization(__file__, 18, 29), hasattr_71439, *[x_71440, str_71441], **kwargs_71442)
    
    
    # Getting the type of 'x' (line 18)
    x_71444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 53), 'x')
    # Obtaining the member 'shape' of a type (line 18)
    shape_71445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 53), x_71444, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 18)
    tuple_71446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 18)
    
    # Applying the binary operator '==' (line 18)
    result_eq_71447 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 53), '==', shape_71445, tuple_71446)
    
    # Applying the binary operator 'and' (line 18)
    result_and_keyword_71448 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 29), 'and', hasattr_call_result_71443, result_eq_71447)
    
    # Applying the binary operator 'or' (line 18)
    result_or_keyword_71449 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), 'or', isscalar_call_result_71438, result_and_keyword_71448)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', result_or_keyword_71449)
    
    # ################# End of '_isscalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_isscalar' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_71450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_71450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_isscalar'
    return stypy_return_type_71450

# Assigning a type to the variable '_isscalar' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_isscalar', _isscalar)
# Declaration of the '_Interpolator1D' class

class _Interpolator1D(object, ):
    str_71451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '\n    Common features in univariate interpolation\n\n    Deal with input data type and interpolation axis rolling.  The\n    actual interpolator can assume the y-data is of shape (n, r) where\n    `n` is the number of x-points, and `r` the number of variables,\n    and use self.dtype as the y-data type.\n\n    Attributes\n    ----------\n    _y_axis\n        Axis along which the interpolation goes in the original array\n    _y_extra_shape\n        Additional trailing shape of the input arrays, excluding\n        the interpolation axis.\n    dtype\n        Dtype of the y-data arrays. Can be set via set_dtype, which\n        forces it to be float or complex.\n\n    Methods\n    -------\n    __call__\n    _prepare_x\n    _finish_y\n    _reshape_yi\n    _set_yi\n    _set_dtype\n    _evaluate\n\n    ')
    
    # Assigning a Tuple to a Name (line 53):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 55)
        None_71452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'None')
        # Getting the type of 'None' (line 55)
        None_71453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'None')
        # Getting the type of 'None' (line 55)
        None_71454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'None')
        defaults = [None_71452, None_71453, None_71454]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D.__init__', ['xi', 'yi', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xi', 'yi', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'axis' (line 56)
        axis_71455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'axis')
        # Getting the type of 'self' (line 56)
        self_71456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member '_y_axis' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_71456, '_y_axis', axis_71455)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'None' (line 57)
        None_71457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'None')
        # Getting the type of 'self' (line 57)
        self_71458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member '_y_extra_shape' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_71458, '_y_extra_shape', None_71457)
        
        # Assigning a Name to a Attribute (line 58):
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'None' (line 58)
        None_71459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'None')
        # Getting the type of 'self' (line 58)
        self_71460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_71460, 'dtype', None_71459)
        
        # Type idiom detected: calculating its left and rigth part (line 59)
        # Getting the type of 'yi' (line 59)
        yi_71461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'yi')
        # Getting the type of 'None' (line 59)
        None_71462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'None')
        
        (may_be_71463, more_types_in_union_71464) = may_not_be_none(yi_71461, None_71462)

        if may_be_71463:

            if more_types_in_union_71464:
                # Runtime conditional SSA (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _set_yi(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'yi' (line 60)
            yi_71467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'yi', False)
            # Processing the call keyword arguments (line 60)
            # Getting the type of 'xi' (line 60)
            xi_71468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'xi', False)
            keyword_71469 = xi_71468
            # Getting the type of 'axis' (line 60)
            axis_71470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'axis', False)
            keyword_71471 = axis_71470
            kwargs_71472 = {'xi': keyword_71469, 'axis': keyword_71471}
            # Getting the type of 'self' (line 60)
            self_71465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self', False)
            # Obtaining the member '_set_yi' of a type (line 60)
            _set_yi_71466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), self_71465, '_set_yi')
            # Calling _set_yi(args, kwargs) (line 60)
            _set_yi_call_result_71473 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), _set_yi_71466, *[yi_71467], **kwargs_71472)
            

            if more_types_in_union_71464:
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_function_name', '_Interpolator1D.__call__')
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_71474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '\n        Evaluate the interpolant\n\n        Parameters\n        ----------\n        x : array_like\n            Points to evaluate the interpolant at.\n\n        Returns\n        -------\n        y : array_like\n            Interpolated values. Shape is determined by replacing\n            the interpolation axis in the original array with the shape of x.\n\n        ')
        
        # Assigning a Call to a Tuple (line 78):
        
        # Assigning a Subscript to a Name (line 78):
        
        # Obtaining the type of the subscript
        int_71475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'int')
        
        # Call to _prepare_x(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'x' (line 78)
        x_71478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'x', False)
        # Processing the call keyword arguments (line 78)
        kwargs_71479 = {}
        # Getting the type of 'self' (line 78)
        self_71476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'self', False)
        # Obtaining the member '_prepare_x' of a type (line 78)
        _prepare_x_71477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), self_71476, '_prepare_x')
        # Calling _prepare_x(args, kwargs) (line 78)
        _prepare_x_call_result_71480 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), _prepare_x_71477, *[x_71478], **kwargs_71479)
        
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___71481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), _prepare_x_call_result_71480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_71482 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), getitem___71481, int_71475)
        
        # Assigning a type to the variable 'tuple_var_assignment_71409' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_71409', subscript_call_result_71482)
        
        # Assigning a Subscript to a Name (line 78):
        
        # Obtaining the type of the subscript
        int_71483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'int')
        
        # Call to _prepare_x(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'x' (line 78)
        x_71486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'x', False)
        # Processing the call keyword arguments (line 78)
        kwargs_71487 = {}
        # Getting the type of 'self' (line 78)
        self_71484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'self', False)
        # Obtaining the member '_prepare_x' of a type (line 78)
        _prepare_x_71485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), self_71484, '_prepare_x')
        # Calling _prepare_x(args, kwargs) (line 78)
        _prepare_x_call_result_71488 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), _prepare_x_71485, *[x_71486], **kwargs_71487)
        
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___71489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), _prepare_x_call_result_71488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_71490 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), getitem___71489, int_71483)
        
        # Assigning a type to the variable 'tuple_var_assignment_71410' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_71410', subscript_call_result_71490)
        
        # Assigning a Name to a Name (line 78):
        # Getting the type of 'tuple_var_assignment_71409' (line 78)
        tuple_var_assignment_71409_71491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_71409')
        # Assigning a type to the variable 'x' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'x', tuple_var_assignment_71409_71491)
        
        # Assigning a Name to a Name (line 78):
        # Getting the type of 'tuple_var_assignment_71410' (line 78)
        tuple_var_assignment_71410_71492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_71410')
        # Assigning a type to the variable 'x_shape' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'x_shape', tuple_var_assignment_71410_71492)
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to _evaluate(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'x' (line 79)
        x_71495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'x', False)
        # Processing the call keyword arguments (line 79)
        kwargs_71496 = {}
        # Getting the type of 'self' (line 79)
        self_71493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self', False)
        # Obtaining the member '_evaluate' of a type (line 79)
        _evaluate_71494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_71493, '_evaluate')
        # Calling _evaluate(args, kwargs) (line 79)
        _evaluate_call_result_71497 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), _evaluate_71494, *[x_71495], **kwargs_71496)
        
        # Assigning a type to the variable 'y' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'y', _evaluate_call_result_71497)
        
        # Call to _finish_y(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'y' (line 80)
        y_71500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'y', False)
        # Getting the type of 'x_shape' (line 80)
        x_shape_71501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'x_shape', False)
        # Processing the call keyword arguments (line 80)
        kwargs_71502 = {}
        # Getting the type of 'self' (line 80)
        self_71498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'self', False)
        # Obtaining the member '_finish_y' of a type (line 80)
        _finish_y_71499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), self_71498, '_finish_y')
        # Calling _finish_y(args, kwargs) (line 80)
        _finish_y_call_result_71503 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), _finish_y_71499, *[y_71500, x_shape_71501], **kwargs_71502)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', _finish_y_call_result_71503)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_71504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_71504


    @norecursion
    def _evaluate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_evaluate'
        module_type_store = module_type_store.open_function_context('_evaluate', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_function_name', '_Interpolator1D._evaluate')
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D._evaluate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D._evaluate', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_evaluate', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_evaluate(...)' code ##################

        str_71505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', '\n        Actually evaluate the value of the interpolator.\n        ')
        
        # Call to NotImplementedError(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_71507 = {}
        # Getting the type of 'NotImplementedError' (line 86)
        NotImplementedError_71506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 86)
        NotImplementedError_call_result_71508 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), NotImplementedError_71506, *[], **kwargs_71507)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 86, 8), NotImplementedError_call_result_71508, 'raise parameter', BaseException)
        
        # ################# End of '_evaluate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_evaluate' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_71509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71509)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_evaluate'
        return stypy_return_type_71509


    @norecursion
    def _prepare_x(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_prepare_x'
        module_type_store = module_type_store.open_function_context('_prepare_x', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_function_name', '_Interpolator1D._prepare_x')
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D._prepare_x.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D._prepare_x', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_prepare_x', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_prepare_x(...)' code ##################

        str_71510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'str', 'Reshape input x array to 1-D')
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to _asarray_validated(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'x' (line 90)
        x_71512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'x', False)
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'False' (line 90)
        False_71513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 47), 'False', False)
        keyword_71514 = False_71513
        # Getting the type of 'True' (line 90)
        True_71515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 65), 'True', False)
        keyword_71516 = True_71515
        kwargs_71517 = {'as_inexact': keyword_71516, 'check_finite': keyword_71514}
        # Getting the type of '_asarray_validated' (line 90)
        _asarray_validated_71511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 90)
        _asarray_validated_call_result_71518 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), _asarray_validated_71511, *[x_71512], **kwargs_71517)
        
        # Assigning a type to the variable 'x' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'x', _asarray_validated_call_result_71518)
        
        # Assigning a Attribute to a Name (line 91):
        
        # Assigning a Attribute to a Name (line 91):
        # Getting the type of 'x' (line 91)
        x_71519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'x')
        # Obtaining the member 'shape' of a type (line 91)
        shape_71520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 18), x_71519, 'shape')
        # Assigning a type to the variable 'x_shape' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'x_shape', shape_71520)
        
        # Obtaining an instance of the builtin type 'tuple' (line 92)
        tuple_71521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 92)
        # Adding element type (line 92)
        
        # Call to ravel(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_71524 = {}
        # Getting the type of 'x' (line 92)
        x_71522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'x', False)
        # Obtaining the member 'ravel' of a type (line 92)
        ravel_71523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), x_71522, 'ravel')
        # Calling ravel(args, kwargs) (line 92)
        ravel_call_result_71525 = invoke(stypy.reporting.localization.Localization(__file__, 92, 15), ravel_71523, *[], **kwargs_71524)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 15), tuple_71521, ravel_call_result_71525)
        # Adding element type (line 92)
        # Getting the type of 'x_shape' (line 92)
        x_shape_71526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'x_shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 15), tuple_71521, x_shape_71526)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', tuple_71521)
        
        # ################# End of '_prepare_x(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_prepare_x' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_71527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71527)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_prepare_x'
        return stypy_return_type_71527


    @norecursion
    def _finish_y(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_finish_y'
        module_type_store = module_type_store.open_function_context('_finish_y', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_function_name', '_Interpolator1D._finish_y')
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_param_names_list', ['y', 'x_shape'])
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D._finish_y.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D._finish_y', ['y', 'x_shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_finish_y', localization, ['y', 'x_shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_finish_y(...)' code ##################

        str_71528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'str', 'Reshape interpolated y back to n-d array similar to initial y')
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to reshape(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'x_shape' (line 96)
        x_shape_71531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'x_shape', False)
        # Getting the type of 'self' (line 96)
        self_71532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'self', False)
        # Obtaining the member '_y_extra_shape' of a type (line 96)
        _y_extra_shape_71533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 32), self_71532, '_y_extra_shape')
        # Applying the binary operator '+' (line 96)
        result_add_71534 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), '+', x_shape_71531, _y_extra_shape_71533)
        
        # Processing the call keyword arguments (line 96)
        kwargs_71535 = {}
        # Getting the type of 'y' (line 96)
        y_71529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'y', False)
        # Obtaining the member 'reshape' of a type (line 96)
        reshape_71530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), y_71529, 'reshape')
        # Calling reshape(args, kwargs) (line 96)
        reshape_call_result_71536 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), reshape_71530, *[result_add_71534], **kwargs_71535)
        
        # Assigning a type to the variable 'y' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'y', reshape_call_result_71536)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 97)
        self_71537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'self')
        # Obtaining the member '_y_axis' of a type (line 97)
        _y_axis_71538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), self_71537, '_y_axis')
        int_71539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 27), 'int')
        # Applying the binary operator '!=' (line 97)
        result_ne_71540 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '!=', _y_axis_71538, int_71539)
        
        
        # Getting the type of 'x_shape' (line 97)
        x_shape_71541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'x_shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_71542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        
        # Applying the binary operator '!=' (line 97)
        result_ne_71543 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 33), '!=', x_shape_71541, tuple_71542)
        
        # Applying the binary operator 'and' (line 97)
        result_and_keyword_71544 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), 'and', result_ne_71540, result_ne_71543)
        
        # Testing the type of an if condition (line 97)
        if_condition_71545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_and_keyword_71544)
        # Assigning a type to the variable 'if_condition_71545' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_71545', if_condition_71545)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x_shape' (line 98)
        x_shape_71547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'x_shape', False)
        # Processing the call keyword arguments (line 98)
        kwargs_71548 = {}
        # Getting the type of 'len' (line 98)
        len_71546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_71549 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), len_71546, *[x_shape_71547], **kwargs_71548)
        
        # Assigning a type to the variable 'nx' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'nx', len_call_result_71549)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to len(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_71551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'self', False)
        # Obtaining the member '_y_extra_shape' of a type (line 99)
        _y_extra_shape_71552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 21), self_71551, '_y_extra_shape')
        # Processing the call keyword arguments (line 99)
        kwargs_71553 = {}
        # Getting the type of 'len' (line 99)
        len_71550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'len', False)
        # Calling len(args, kwargs) (line 99)
        len_call_result_71554 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), len_71550, *[_y_extra_shape_71552], **kwargs_71553)
        
        # Assigning a type to the variable 'ny' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'ny', len_call_result_71554)
        
        # Assigning a BinOp to a Name (line 100):
        
        # Assigning a BinOp to a Name (line 100):
        
        # Call to list(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to range(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'nx' (line 100)
        nx_71557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'nx', False)
        # Getting the type of 'nx' (line 100)
        nx_71558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'nx', False)
        # Getting the type of 'self' (line 100)
        self_71559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 37), 'self', False)
        # Obtaining the member '_y_axis' of a type (line 100)
        _y_axis_71560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 37), self_71559, '_y_axis')
        # Applying the binary operator '+' (line 100)
        result_add_71561 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 32), '+', nx_71558, _y_axis_71560)
        
        # Processing the call keyword arguments (line 100)
        kwargs_71562 = {}
        # Getting the type of 'range' (line 100)
        range_71556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'range', False)
        # Calling range(args, kwargs) (line 100)
        range_call_result_71563 = invoke(stypy.reporting.localization.Localization(__file__, 100, 22), range_71556, *[nx_71557, result_add_71561], **kwargs_71562)
        
        # Processing the call keyword arguments (line 100)
        kwargs_71564 = {}
        # Getting the type of 'list' (line 100)
        list_71555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'list', False)
        # Calling list(args, kwargs) (line 100)
        list_call_result_71565 = invoke(stypy.reporting.localization.Localization(__file__, 100, 17), list_71555, *[range_call_result_71563], **kwargs_71564)
        
        
        # Call to list(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to range(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'nx' (line 101)
        nx_71568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'nx', False)
        # Processing the call keyword arguments (line 101)
        kwargs_71569 = {}
        # Getting the type of 'range' (line 101)
        range_71567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'range', False)
        # Calling range(args, kwargs) (line 101)
        range_call_result_71570 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), range_71567, *[nx_71568], **kwargs_71569)
        
        # Processing the call keyword arguments (line 101)
        kwargs_71571 = {}
        # Getting the type of 'list' (line 101)
        list_71566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'list', False)
        # Calling list(args, kwargs) (line 101)
        list_call_result_71572 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), list_71566, *[range_call_result_71570], **kwargs_71571)
        
        # Applying the binary operator '+' (line 100)
        result_add_71573 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 17), '+', list_call_result_71565, list_call_result_71572)
        
        
        # Call to list(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to range(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'nx' (line 101)
        nx_71576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 48), 'nx', False)
        # Getting the type of 'self' (line 101)
        self_71577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'self', False)
        # Obtaining the member '_y_axis' of a type (line 101)
        _y_axis_71578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 51), self_71577, '_y_axis')
        # Applying the binary operator '+' (line 101)
        result_add_71579 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 48), '+', nx_71576, _y_axis_71578)
        
        # Getting the type of 'nx' (line 101)
        nx_71580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 65), 'nx', False)
        # Getting the type of 'ny' (line 101)
        ny_71581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 68), 'ny', False)
        # Applying the binary operator '+' (line 101)
        result_add_71582 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 65), '+', nx_71580, ny_71581)
        
        # Processing the call keyword arguments (line 101)
        kwargs_71583 = {}
        # Getting the type of 'range' (line 101)
        range_71575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'range', False)
        # Calling range(args, kwargs) (line 101)
        range_call_result_71584 = invoke(stypy.reporting.localization.Localization(__file__, 101, 42), range_71575, *[result_add_71579, result_add_71582], **kwargs_71583)
        
        # Processing the call keyword arguments (line 101)
        kwargs_71585 = {}
        # Getting the type of 'list' (line 101)
        list_71574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'list', False)
        # Calling list(args, kwargs) (line 101)
        list_call_result_71586 = invoke(stypy.reporting.localization.Localization(__file__, 101, 37), list_71574, *[range_call_result_71584], **kwargs_71585)
        
        # Applying the binary operator '+' (line 101)
        result_add_71587 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 35), '+', result_add_71573, list_call_result_71586)
        
        # Assigning a type to the variable 's' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 's', result_add_71587)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to transpose(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 's' (line 102)
        s_71590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 's', False)
        # Processing the call keyword arguments (line 102)
        kwargs_71591 = {}
        # Getting the type of 'y' (line 102)
        y_71588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'y', False)
        # Obtaining the member 'transpose' of a type (line 102)
        transpose_71589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), y_71588, 'transpose')
        # Calling transpose(args, kwargs) (line 102)
        transpose_call_result_71592 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), transpose_71589, *[s_71590], **kwargs_71591)
        
        # Assigning a type to the variable 'y' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'y', transpose_call_result_71592)
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 103)
        y_71593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', y_71593)
        
        # ################# End of '_finish_y(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_finish_y' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_71594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_finish_y'
        return stypy_return_type_71594


    @norecursion
    def _reshape_yi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 105)
        False_71595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 36), 'False')
        defaults = [False_71595]
        # Create a new context for function '_reshape_yi'
        module_type_store = module_type_store.open_function_context('_reshape_yi', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_function_name', '_Interpolator1D._reshape_yi')
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_param_names_list', ['yi', 'check'])
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D._reshape_yi.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D._reshape_yi', ['yi', 'check'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_reshape_yi', localization, ['yi', 'check'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_reshape_yi(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to rollaxis(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Call to asarray(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'yi' (line 106)
        yi_71600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'yi', False)
        # Processing the call keyword arguments (line 106)
        kwargs_71601 = {}
        # Getting the type of 'np' (line 106)
        np_71598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'np', False)
        # Obtaining the member 'asarray' of a type (line 106)
        asarray_71599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 25), np_71598, 'asarray')
        # Calling asarray(args, kwargs) (line 106)
        asarray_call_result_71602 = invoke(stypy.reporting.localization.Localization(__file__, 106, 25), asarray_71599, *[yi_71600], **kwargs_71601)
        
        # Getting the type of 'self' (line 106)
        self_71603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'self', False)
        # Obtaining the member '_y_axis' of a type (line 106)
        _y_axis_71604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 41), self_71603, '_y_axis')
        # Processing the call keyword arguments (line 106)
        kwargs_71605 = {}
        # Getting the type of 'np' (line 106)
        np_71596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'np', False)
        # Obtaining the member 'rollaxis' of a type (line 106)
        rollaxis_71597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 13), np_71596, 'rollaxis')
        # Calling rollaxis(args, kwargs) (line 106)
        rollaxis_call_result_71606 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), rollaxis_71597, *[asarray_call_result_71602, _y_axis_71604], **kwargs_71605)
        
        # Assigning a type to the variable 'yi' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'yi', rollaxis_call_result_71606)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'check' (line 107)
        check_71607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'check')
        
        
        # Obtaining the type of the subscript
        int_71608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'int')
        slice_71609 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 107, 21), int_71608, None, None)
        # Getting the type of 'yi' (line 107)
        yi_71610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'yi')
        # Obtaining the member 'shape' of a type (line 107)
        shape_71611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 21), yi_71610, 'shape')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___71612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 21), shape_71611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_71613 = invoke(stypy.reporting.localization.Localization(__file__, 107, 21), getitem___71612, slice_71609)
        
        # Getting the type of 'self' (line 107)
        self_71614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'self')
        # Obtaining the member '_y_extra_shape' of a type (line 107)
        _y_extra_shape_71615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 37), self_71614, '_y_extra_shape')
        # Applying the binary operator '!=' (line 107)
        result_ne_71616 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 21), '!=', subscript_call_result_71613, _y_extra_shape_71615)
        
        # Applying the binary operator 'and' (line 107)
        result_and_keyword_71617 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), 'and', check_71607, result_ne_71616)
        
        # Testing the type of an if condition (line 107)
        if_condition_71618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_and_keyword_71617)
        # Assigning a type to the variable 'if_condition_71618' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_71618', if_condition_71618)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 108):
        
        # Assigning a BinOp to a Name (line 108):
        str_71619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'str', '%r + (N,) + %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_71620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'self' (line 108)
        self_71621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 64), 'self')
        # Obtaining the member '_y_axis' of a type (line 108)
        _y_axis_71622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 64), self_71621, '_y_axis')
        # Applying the 'usub' unary operator (line 108)
        result___neg___71623 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 63), 'usub', _y_axis_71622)
        
        slice_71624 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 43), result___neg___71623, None, None)
        # Getting the type of 'self' (line 108)
        self_71625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 43), 'self')
        # Obtaining the member '_y_extra_shape' of a type (line 108)
        _y_extra_shape_71626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 43), self_71625, '_y_extra_shape')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___71627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 43), _y_extra_shape_71626, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_71628 = invoke(stypy.reporting.localization.Localization(__file__, 108, 43), getitem___71627, slice_71624)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 43), tuple_71620, subscript_call_result_71628)
        # Adding element type (line 108)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'self' (line 109)
        self_71629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 65), 'self')
        # Obtaining the member '_y_axis' of a type (line 109)
        _y_axis_71630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 65), self_71629, '_y_axis')
        # Applying the 'usub' unary operator (line 109)
        result___neg___71631 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 64), 'usub', _y_axis_71630)
        
        slice_71632 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 43), None, result___neg___71631, None)
        # Getting the type of 'self' (line 109)
        self_71633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 43), 'self')
        # Obtaining the member '_y_extra_shape' of a type (line 109)
        _y_extra_shape_71634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 43), self_71633, '_y_extra_shape')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___71635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 43), _y_extra_shape_71634, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_71636 = invoke(stypy.reporting.localization.Localization(__file__, 109, 43), getitem___71635, slice_71632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 43), tuple_71620, subscript_call_result_71636)
        
        # Applying the binary operator '%' (line 108)
        result_mod_71637 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), '%', str_71619, tuple_71620)
        
        # Assigning a type to the variable 'ok_shape' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'ok_shape', result_mod_71637)
        
        # Call to ValueError(...): (line 110)
        # Processing the call arguments (line 110)
        str_71639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'str', 'Data must be of shape %s')
        # Getting the type of 'ok_shape' (line 110)
        ok_shape_71640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 58), 'ok_shape', False)
        # Applying the binary operator '%' (line 110)
        result_mod_71641 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 29), '%', str_71639, ok_shape_71640)
        
        # Processing the call keyword arguments (line 110)
        kwargs_71642 = {}
        # Getting the type of 'ValueError' (line 110)
        ValueError_71638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 110)
        ValueError_call_result_71643 = invoke(stypy.reporting.localization.Localization(__file__, 110, 18), ValueError_71638, *[result_mod_71641], **kwargs_71642)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 110, 12), ValueError_call_result_71643, 'raise parameter', BaseException)
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to reshape(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'tuple' (line 111)
        tuple_71646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 111)
        # Adding element type (line 111)
        
        # Obtaining the type of the subscript
        int_71647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 36), 'int')
        # Getting the type of 'yi' (line 111)
        yi_71648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'yi', False)
        # Obtaining the member 'shape' of a type (line 111)
        shape_71649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 27), yi_71648, 'shape')
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___71650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 27), shape_71649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_71651 = invoke(stypy.reporting.localization.Localization(__file__, 111, 27), getitem___71650, int_71647)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 27), tuple_71646, subscript_call_result_71651)
        # Adding element type (line 111)
        int_71652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 27), tuple_71646, int_71652)
        
        # Processing the call keyword arguments (line 111)
        kwargs_71653 = {}
        # Getting the type of 'yi' (line 111)
        yi_71644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'yi', False)
        # Obtaining the member 'reshape' of a type (line 111)
        reshape_71645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), yi_71644, 'reshape')
        # Calling reshape(args, kwargs) (line 111)
        reshape_call_result_71654 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), reshape_71645, *[tuple_71646], **kwargs_71653)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', reshape_call_result_71654)
        
        # ################# End of '_reshape_yi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_reshape_yi' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_71655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71655)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_reshape_yi'
        return stypy_return_type_71655


    @norecursion
    def _set_yi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 113)
        None_71656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'None')
        # Getting the type of 'None' (line 113)
        None_71657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'None')
        defaults = [None_71656, None_71657]
        # Create a new context for function '_set_yi'
        module_type_store = module_type_store.open_function_context('_set_yi', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_function_name', '_Interpolator1D._set_yi')
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_param_names_list', ['yi', 'xi', 'axis'])
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D._set_yi.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D._set_yi', ['yi', 'xi', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_yi', localization, ['yi', 'xi', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_yi(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 114)
        # Getting the type of 'axis' (line 114)
        axis_71658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'axis')
        # Getting the type of 'None' (line 114)
        None_71659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'None')
        
        (may_be_71660, more_types_in_union_71661) = may_be_none(axis_71658, None_71659)

        if may_be_71660:

            if more_types_in_union_71661:
                # Runtime conditional SSA (line 114)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 115):
            
            # Assigning a Attribute to a Name (line 115):
            # Getting the type of 'self' (line 115)
            self_71662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'self')
            # Obtaining the member '_y_axis' of a type (line 115)
            _y_axis_71663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), self_71662, '_y_axis')
            # Assigning a type to the variable 'axis' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'axis', _y_axis_71663)

            if more_types_in_union_71661:
                # SSA join for if statement (line 114)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 116)
        # Getting the type of 'axis' (line 116)
        axis_71664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'axis')
        # Getting the type of 'None' (line 116)
        None_71665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'None')
        
        (may_be_71666, more_types_in_union_71667) = may_be_none(axis_71664, None_71665)

        if may_be_71666:

            if more_types_in_union_71667:
                # Runtime conditional SSA (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 117)
            # Processing the call arguments (line 117)
            str_71669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'str', 'no interpolation axis specified')
            # Processing the call keyword arguments (line 117)
            kwargs_71670 = {}
            # Getting the type of 'ValueError' (line 117)
            ValueError_71668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 117)
            ValueError_call_result_71671 = invoke(stypy.reporting.localization.Localization(__file__, 117, 18), ValueError_71668, *[str_71669], **kwargs_71670)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 117, 12), ValueError_call_result_71671, 'raise parameter', BaseException)

            if more_types_in_union_71667:
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to asarray(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'yi' (line 119)
        yi_71674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'yi', False)
        # Processing the call keyword arguments (line 119)
        kwargs_71675 = {}
        # Getting the type of 'np' (line 119)
        np_71672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'np', False)
        # Obtaining the member 'asarray' of a type (line 119)
        asarray_71673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), np_71672, 'asarray')
        # Calling asarray(args, kwargs) (line 119)
        asarray_call_result_71676 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), asarray_71673, *[yi_71674], **kwargs_71675)
        
        # Assigning a type to the variable 'yi' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'yi', asarray_call_result_71676)
        
        # Assigning a Attribute to a Name (line 121):
        
        # Assigning a Attribute to a Name (line 121):
        # Getting the type of 'yi' (line 121)
        yi_71677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'yi')
        # Obtaining the member 'shape' of a type (line 121)
        shape_71678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), yi_71677, 'shape')
        # Assigning a type to the variable 'shape' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'shape', shape_71678)
        
        
        # Getting the type of 'shape' (line 122)
        shape_71679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_71680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        
        # Applying the binary operator '==' (line 122)
        result_eq_71681 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 11), '==', shape_71679, tuple_71680)
        
        # Testing the type of an if condition (line 122)
        if_condition_71682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 8), result_eq_71681)
        # Assigning a type to the variable 'if_condition_71682' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'if_condition_71682', if_condition_71682)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 123):
        
        # Assigning a Tuple to a Name (line 123):
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_71683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_71684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 21), tuple_71683, int_71684)
        
        # Assigning a type to the variable 'shape' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'shape', tuple_71683)
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'xi' (line 124)
        xi_71685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'xi')
        # Getting the type of 'None' (line 124)
        None_71686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'None')
        # Applying the binary operator 'isnot' (line 124)
        result_is_not_71687 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), 'isnot', xi_71685, None_71686)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 124)
        axis_71688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'axis')
        # Getting the type of 'shape' (line 124)
        shape_71689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'shape')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___71690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 30), shape_71689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_71691 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), getitem___71690, axis_71688)
        
        
        # Call to len(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'xi' (line 124)
        xi_71693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 49), 'xi', False)
        # Processing the call keyword arguments (line 124)
        kwargs_71694 = {}
        # Getting the type of 'len' (line 124)
        len_71692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'len', False)
        # Calling len(args, kwargs) (line 124)
        len_call_result_71695 = invoke(stypy.reporting.localization.Localization(__file__, 124, 45), len_71692, *[xi_71693], **kwargs_71694)
        
        # Applying the binary operator '!=' (line 124)
        result_ne_71696 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 30), '!=', subscript_call_result_71691, len_call_result_71695)
        
        # Applying the binary operator 'and' (line 124)
        result_and_keyword_71697 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), 'and', result_is_not_71687, result_ne_71696)
        
        # Testing the type of an if condition (line 124)
        if_condition_71698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_and_keyword_71697)
        # Assigning a type to the variable 'if_condition_71698' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_71698', if_condition_71698)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 125)
        # Processing the call arguments (line 125)
        str_71700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'str', 'x and y arrays must be equal in length along interpolation axis.')
        # Processing the call keyword arguments (line 125)
        kwargs_71701 = {}
        # Getting the type of 'ValueError' (line 125)
        ValueError_71699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 125)
        ValueError_call_result_71702 = invoke(stypy.reporting.localization.Localization(__file__, 125, 18), ValueError_71699, *[str_71700], **kwargs_71701)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 125, 12), ValueError_call_result_71702, 'raise parameter', BaseException)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 128):
        
        # Assigning a BinOp to a Attribute (line 128):
        # Getting the type of 'axis' (line 128)
        axis_71703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'axis')
        # Getting the type of 'yi' (line 128)
        yi_71704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'yi')
        # Obtaining the member 'ndim' of a type (line 128)
        ndim_71705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), yi_71704, 'ndim')
        # Applying the binary operator '%' (line 128)
        result_mod_71706 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 24), '%', axis_71703, ndim_71705)
        
        # Getting the type of 'self' (line 128)
        self_71707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member '_y_axis' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_71707, '_y_axis', result_mod_71706)
        
        # Assigning a BinOp to a Attribute (line 129):
        
        # Assigning a BinOp to a Attribute (line 129):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 129)
        self_71708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'self')
        # Obtaining the member '_y_axis' of a type (line 129)
        _y_axis_71709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 40), self_71708, '_y_axis')
        slice_71710 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 129, 30), None, _y_axis_71709, None)
        # Getting the type of 'yi' (line 129)
        yi_71711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'yi')
        # Obtaining the member 'shape' of a type (line 129)
        shape_71712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 30), yi_71711, 'shape')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___71713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 30), shape_71712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_71714 = invoke(stypy.reporting.localization.Localization(__file__, 129, 30), getitem___71713, slice_71710)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 129)
        self_71715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'self')
        # Obtaining the member '_y_axis' of a type (line 129)
        _y_axis_71716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 63), self_71715, '_y_axis')
        int_71717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 76), 'int')
        # Applying the binary operator '+' (line 129)
        result_add_71718 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 63), '+', _y_axis_71716, int_71717)
        
        slice_71719 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 129, 54), result_add_71718, None, None)
        # Getting the type of 'yi' (line 129)
        yi_71720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 54), 'yi')
        # Obtaining the member 'shape' of a type (line 129)
        shape_71721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 54), yi_71720, 'shape')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___71722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 54), shape_71721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_71723 = invoke(stypy.reporting.localization.Localization(__file__, 129, 54), getitem___71722, slice_71719)
        
        # Applying the binary operator '+' (line 129)
        result_add_71724 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 30), '+', subscript_call_result_71714, subscript_call_result_71723)
        
        # Getting the type of 'self' (line 129)
        self_71725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member '_y_extra_shape' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_71725, '_y_extra_shape', result_add_71724)
        
        # Assigning a Name to a Attribute (line 130):
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'None' (line 130)
        None_71726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'None')
        # Getting the type of 'self' (line 130)
        self_71727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_71727, 'dtype', None_71726)
        
        # Call to _set_dtype(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'yi' (line 131)
        yi_71730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'yi', False)
        # Obtaining the member 'dtype' of a type (line 131)
        dtype_71731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), yi_71730, 'dtype')
        # Processing the call keyword arguments (line 131)
        kwargs_71732 = {}
        # Getting the type of 'self' (line 131)
        self_71728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member '_set_dtype' of a type (line 131)
        _set_dtype_71729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_71728, '_set_dtype')
        # Calling _set_dtype(args, kwargs) (line 131)
        _set_dtype_call_result_71733 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), _set_dtype_71729, *[dtype_71731], **kwargs_71732)
        
        
        # ################# End of '_set_yi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_yi' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_71734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_yi'
        return stypy_return_type_71734


    @norecursion
    def _set_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 133)
        False_71735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'False')
        defaults = [False_71735]
        # Create a new context for function '_set_dtype'
        module_type_store = module_type_store.open_function_context('_set_dtype', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_function_name', '_Interpolator1D._set_dtype')
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_param_names_list', ['dtype', 'union'])
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1D._set_dtype.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1D._set_dtype', ['dtype', 'union'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_dtype', localization, ['dtype', 'union'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_dtype(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Call to issubdtype(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'dtype' (line 134)
        dtype_71738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'dtype', False)
        # Getting the type of 'np' (line 134)
        np_71739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 134)
        complexfloating_71740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), np_71739, 'complexfloating')
        # Processing the call keyword arguments (line 134)
        kwargs_71741 = {}
        # Getting the type of 'np' (line 134)
        np_71736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 134)
        issubdtype_71737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), np_71736, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 134)
        issubdtype_call_result_71742 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), issubdtype_71737, *[dtype_71738, complexfloating_71740], **kwargs_71741)
        
        
        # Call to issubdtype(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_71745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 135)
        dtype_71746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 32), self_71745, 'dtype')
        # Getting the type of 'np' (line 135)
        np_71747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 44), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 135)
        complexfloating_71748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 44), np_71747, 'complexfloating')
        # Processing the call keyword arguments (line 135)
        kwargs_71749 = {}
        # Getting the type of 'np' (line 135)
        np_71743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 135)
        issubdtype_71744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), np_71743, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 135)
        issubdtype_call_result_71750 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), issubdtype_71744, *[dtype_71746, complexfloating_71748], **kwargs_71749)
        
        # Applying the binary operator 'or' (line 134)
        result_or_keyword_71751 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), 'or', issubdtype_call_result_71742, issubdtype_call_result_71750)
        
        # Testing the type of an if condition (line 134)
        if_condition_71752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_or_keyword_71751)
        # Assigning a type to the variable 'if_condition_71752' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_71752', if_condition_71752)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 136):
        
        # Assigning a Attribute to a Attribute (line 136):
        # Getting the type of 'np' (line 136)
        np_71753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'np')
        # Obtaining the member 'complex_' of a type (line 136)
        complex__71754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 25), np_71753, 'complex_')
        # Getting the type of 'self' (line 136)
        self_71755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), self_71755, 'dtype', complex__71754)
        # SSA branch for the else part of an if statement (line 134)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'union' (line 138)
        union_71756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'union')
        # Applying the 'not' unary operator (line 138)
        result_not__71757 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), 'not', union_71756)
        
        
        # Getting the type of 'self' (line 138)
        self_71758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'self')
        # Obtaining the member 'dtype' of a type (line 138)
        dtype_71759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 28), self_71758, 'dtype')
        # Getting the type of 'np' (line 138)
        np_71760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'np')
        # Obtaining the member 'complex_' of a type (line 138)
        complex__71761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), np_71760, 'complex_')
        # Applying the binary operator '!=' (line 138)
        result_ne_71762 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 28), '!=', dtype_71759, complex__71761)
        
        # Applying the binary operator 'or' (line 138)
        result_or_keyword_71763 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), 'or', result_not__71757, result_ne_71762)
        
        # Testing the type of an if condition (line 138)
        if_condition_71764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 12), result_or_keyword_71763)
        # Assigning a type to the variable 'if_condition_71764' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'if_condition_71764', if_condition_71764)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 139):
        
        # Assigning a Attribute to a Attribute (line 139):
        # Getting the type of 'np' (line 139)
        np_71765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'np')
        # Obtaining the member 'float_' of a type (line 139)
        float__71766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 29), np_71765, 'float_')
        # Getting the type of 'self' (line 139)
        self_71767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'self')
        # Setting the type of the member 'dtype' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), self_71767, 'dtype', float__71766)
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_71768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_dtype'
        return stypy_return_type_71768


# Assigning a type to the variable '_Interpolator1D' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '_Interpolator1D', _Interpolator1D)

# Assigning a Tuple to a Name (line 53):

# Obtaining an instance of the builtin type 'tuple' (line 53)
tuple_71769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 53)
# Adding element type (line 53)
str_71770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'str', '_y_axis')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), tuple_71769, str_71770)
# Adding element type (line 53)
str_71771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'str', '_y_extra_shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), tuple_71769, str_71771)
# Adding element type (line 53)
str_71772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 46), 'str', 'dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), tuple_71769, str_71772)

# Getting the type of '_Interpolator1D'
_Interpolator1D_71773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_Interpolator1D')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _Interpolator1D_71773, '__slots__', tuple_71769)
# Declaration of the '_Interpolator1DWithDerivatives' class
# Getting the type of '_Interpolator1D' (line 142)
_Interpolator1D_71774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), '_Interpolator1D')

class _Interpolator1DWithDerivatives(_Interpolator1D_71774, ):

    @norecursion
    def derivatives(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 143)
        None_71775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'None')
        defaults = [None_71775]
        # Create a new context for function 'derivatives'
        module_type_store = module_type_store.open_function_context('derivatives', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_function_name', '_Interpolator1DWithDerivatives.derivatives')
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_param_names_list', ['x', 'der'])
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1DWithDerivatives.derivatives.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1DWithDerivatives.derivatives', ['x', 'der'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derivatives', localization, ['x', 'der'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derivatives(...)' code ##################

        str_71776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n        Evaluate many derivatives of the polynomial at the point x\n\n        Produce an array of all derivative values at the point x.\n\n        Parameters\n        ----------\n        x : array_like\n            Point or points at which to evaluate the derivatives\n        der : int or None, optional\n            How many derivatives to extract; None for all potentially\n            nonzero derivatives (that is a number equal to the number\n            of points). This number includes the function value as 0th\n            derivative.\n\n        Returns\n        -------\n        d : ndarray\n            Array with derivatives; d[j] contains the j-th derivative.\n            Shape of d[j] is determined by replacing the interpolation\n            axis in the original array with the shape of x.\n\n        Examples\n        --------\n        >>> from scipy.interpolate import KroghInterpolator\n        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)\n        array([1.0,2.0,3.0])\n        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])\n        array([[1.0,1.0],\n               [2.0,2.0],\n               [3.0,3.0]])\n\n        ')
        
        # Assigning a Call to a Tuple (line 177):
        
        # Assigning a Subscript to a Name (line 177):
        
        # Obtaining the type of the subscript
        int_71777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        
        # Call to _prepare_x(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'x' (line 177)
        x_71780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 37), 'x', False)
        # Processing the call keyword arguments (line 177)
        kwargs_71781 = {}
        # Getting the type of 'self' (line 177)
        self_71778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'self', False)
        # Obtaining the member '_prepare_x' of a type (line 177)
        _prepare_x_71779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), self_71778, '_prepare_x')
        # Calling _prepare_x(args, kwargs) (line 177)
        _prepare_x_call_result_71782 = invoke(stypy.reporting.localization.Localization(__file__, 177, 21), _prepare_x_71779, *[x_71780], **kwargs_71781)
        
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___71783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), _prepare_x_call_result_71782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_71784 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), getitem___71783, int_71777)
        
        # Assigning a type to the variable 'tuple_var_assignment_71411' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_71411', subscript_call_result_71784)
        
        # Assigning a Subscript to a Name (line 177):
        
        # Obtaining the type of the subscript
        int_71785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        
        # Call to _prepare_x(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'x' (line 177)
        x_71788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 37), 'x', False)
        # Processing the call keyword arguments (line 177)
        kwargs_71789 = {}
        # Getting the type of 'self' (line 177)
        self_71786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'self', False)
        # Obtaining the member '_prepare_x' of a type (line 177)
        _prepare_x_71787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), self_71786, '_prepare_x')
        # Calling _prepare_x(args, kwargs) (line 177)
        _prepare_x_call_result_71790 = invoke(stypy.reporting.localization.Localization(__file__, 177, 21), _prepare_x_71787, *[x_71788], **kwargs_71789)
        
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___71791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), _prepare_x_call_result_71790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_71792 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), getitem___71791, int_71785)
        
        # Assigning a type to the variable 'tuple_var_assignment_71412' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_71412', subscript_call_result_71792)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'tuple_var_assignment_71411' (line 177)
        tuple_var_assignment_71411_71793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_71411')
        # Assigning a type to the variable 'x' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'x', tuple_var_assignment_71411_71793)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'tuple_var_assignment_71412' (line 177)
        tuple_var_assignment_71412_71794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_71412')
        # Assigning a type to the variable 'x_shape' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'x_shape', tuple_var_assignment_71412_71794)
        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to _evaluate_derivatives(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'x' (line 178)
        x_71797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 39), 'x', False)
        # Getting the type of 'der' (line 178)
        der_71798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'der', False)
        # Processing the call keyword arguments (line 178)
        kwargs_71799 = {}
        # Getting the type of 'self' (line 178)
        self_71795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self', False)
        # Obtaining the member '_evaluate_derivatives' of a type (line 178)
        _evaluate_derivatives_71796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_71795, '_evaluate_derivatives')
        # Calling _evaluate_derivatives(args, kwargs) (line 178)
        _evaluate_derivatives_call_result_71800 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), _evaluate_derivatives_71796, *[x_71797, der_71798], **kwargs_71799)
        
        # Assigning a type to the variable 'y' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'y', _evaluate_derivatives_call_result_71800)
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to reshape(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_71803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        
        # Obtaining the type of the subscript
        int_71804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'int')
        # Getting the type of 'y' (line 180)
        y_71805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'y', False)
        # Obtaining the member 'shape' of a type (line 180)
        shape_71806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 23), y_71805, 'shape')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___71807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 23), shape_71806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_71808 = invoke(stypy.reporting.localization.Localization(__file__, 180, 23), getitem___71807, int_71804)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 23), tuple_71803, subscript_call_result_71808)
        
        # Getting the type of 'x_shape' (line 180)
        x_shape_71809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 38), 'x_shape', False)
        # Applying the binary operator '+' (line 180)
        result_add_71810 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 22), '+', tuple_71803, x_shape_71809)
        
        # Getting the type of 'self' (line 180)
        self_71811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 48), 'self', False)
        # Obtaining the member '_y_extra_shape' of a type (line 180)
        _y_extra_shape_71812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 48), self_71811, '_y_extra_shape')
        # Applying the binary operator '+' (line 180)
        result_add_71813 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 46), '+', result_add_71810, _y_extra_shape_71812)
        
        # Processing the call keyword arguments (line 180)
        kwargs_71814 = {}
        # Getting the type of 'y' (line 180)
        y_71801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'y', False)
        # Obtaining the member 'reshape' of a type (line 180)
        reshape_71802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), y_71801, 'reshape')
        # Calling reshape(args, kwargs) (line 180)
        reshape_call_result_71815 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), reshape_71802, *[result_add_71813], **kwargs_71814)
        
        # Assigning a type to the variable 'y' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'y', reshape_call_result_71815)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 181)
        self_71816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'self')
        # Obtaining the member '_y_axis' of a type (line 181)
        _y_axis_71817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 11), self_71816, '_y_axis')
        int_71818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 27), 'int')
        # Applying the binary operator '!=' (line 181)
        result_ne_71819 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), '!=', _y_axis_71817, int_71818)
        
        
        # Getting the type of 'x_shape' (line 181)
        x_shape_71820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'x_shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_71821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        
        # Applying the binary operator '!=' (line 181)
        result_ne_71822 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 33), '!=', x_shape_71820, tuple_71821)
        
        # Applying the binary operator 'and' (line 181)
        result_and_keyword_71823 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'and', result_ne_71819, result_ne_71822)
        
        # Testing the type of an if condition (line 181)
        if_condition_71824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_and_keyword_71823)
        # Assigning a type to the variable 'if_condition_71824' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_71824', if_condition_71824)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to len(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'x_shape' (line 182)
        x_shape_71826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'x_shape', False)
        # Processing the call keyword arguments (line 182)
        kwargs_71827 = {}
        # Getting the type of 'len' (line 182)
        len_71825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'len', False)
        # Calling len(args, kwargs) (line 182)
        len_call_result_71828 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), len_71825, *[x_shape_71826], **kwargs_71827)
        
        # Assigning a type to the variable 'nx' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'nx', len_call_result_71828)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to len(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'self' (line 183)
        self_71830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'self', False)
        # Obtaining the member '_y_extra_shape' of a type (line 183)
        _y_extra_shape_71831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 21), self_71830, '_y_extra_shape')
        # Processing the call keyword arguments (line 183)
        kwargs_71832 = {}
        # Getting the type of 'len' (line 183)
        len_71829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'len', False)
        # Calling len(args, kwargs) (line 183)
        len_call_result_71833 = invoke(stypy.reporting.localization.Localization(__file__, 183, 17), len_71829, *[_y_extra_shape_71831], **kwargs_71832)
        
        # Assigning a type to the variable 'ny' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'ny', len_call_result_71833)
        
        # Assigning a BinOp to a Name (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        
        # Obtaining an instance of the builtin type 'list' (line 184)
        list_71834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 184)
        # Adding element type (line 184)
        int_71835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 17), list_71834, int_71835)
        
        
        # Call to list(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to range(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'nx' (line 184)
        nx_71838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'nx', False)
        int_71839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 37), 'int')
        # Applying the binary operator '+' (line 184)
        result_add_71840 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 34), '+', nx_71838, int_71839)
        
        # Getting the type of 'nx' (line 184)
        nx_71841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 40), 'nx', False)
        # Getting the type of 'self' (line 184)
        self_71842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 45), 'self', False)
        # Obtaining the member '_y_axis' of a type (line 184)
        _y_axis_71843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 45), self_71842, '_y_axis')
        # Applying the binary operator '+' (line 184)
        result_add_71844 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 40), '+', nx_71841, _y_axis_71843)
        
        int_71845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 58), 'int')
        # Applying the binary operator '+' (line 184)
        result_add_71846 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 57), '+', result_add_71844, int_71845)
        
        # Processing the call keyword arguments (line 184)
        kwargs_71847 = {}
        # Getting the type of 'range' (line 184)
        range_71837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'range', False)
        # Calling range(args, kwargs) (line 184)
        range_call_result_71848 = invoke(stypy.reporting.localization.Localization(__file__, 184, 28), range_71837, *[result_add_71840, result_add_71846], **kwargs_71847)
        
        # Processing the call keyword arguments (line 184)
        kwargs_71849 = {}
        # Getting the type of 'list' (line 184)
        list_71836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'list', False)
        # Calling list(args, kwargs) (line 184)
        list_call_result_71850 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), list_71836, *[range_call_result_71848], **kwargs_71849)
        
        # Applying the binary operator '+' (line 184)
        result_add_71851 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 17), '+', list_71834, list_call_result_71850)
        
        
        # Call to list(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Call to range(...): (line 185)
        # Processing the call arguments (line 185)
        int_71854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'int')
        # Getting the type of 'nx' (line 185)
        nx_71855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'nx', False)
        int_71856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 35), 'int')
        # Applying the binary operator '+' (line 185)
        result_add_71857 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 32), '+', nx_71855, int_71856)
        
        # Processing the call keyword arguments (line 185)
        kwargs_71858 = {}
        # Getting the type of 'range' (line 185)
        range_71853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'range', False)
        # Calling range(args, kwargs) (line 185)
        range_call_result_71859 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), range_71853, *[int_71854, result_add_71857], **kwargs_71858)
        
        # Processing the call keyword arguments (line 185)
        kwargs_71860 = {}
        # Getting the type of 'list' (line 185)
        list_71852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'list', False)
        # Calling list(args, kwargs) (line 185)
        list_call_result_71861 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), list_71852, *[range_call_result_71859], **kwargs_71860)
        
        # Applying the binary operator '+' (line 185)
        result_add_71862 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 17), '+', result_add_71851, list_call_result_71861)
        
        
        # Call to list(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to range(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'nx' (line 186)
        nx_71865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'nx', False)
        int_71866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'int')
        # Applying the binary operator '+' (line 186)
        result_add_71867 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 28), '+', nx_71865, int_71866)
        
        # Getting the type of 'self' (line 186)
        self_71868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 'self', False)
        # Obtaining the member '_y_axis' of a type (line 186)
        _y_axis_71869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 33), self_71868, '_y_axis')
        # Applying the binary operator '+' (line 186)
        result_add_71870 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 32), '+', result_add_71867, _y_axis_71869)
        
        # Getting the type of 'nx' (line 186)
        nx_71871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 47), 'nx', False)
        # Getting the type of 'ny' (line 186)
        ny_71872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 50), 'ny', False)
        # Applying the binary operator '+' (line 186)
        result_add_71873 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 47), '+', nx_71871, ny_71872)
        
        int_71874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 53), 'int')
        # Applying the binary operator '+' (line 186)
        result_add_71875 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 52), '+', result_add_71873, int_71874)
        
        # Processing the call keyword arguments (line 186)
        kwargs_71876 = {}
        # Getting the type of 'range' (line 186)
        range_71864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'range', False)
        # Calling range(args, kwargs) (line 186)
        range_call_result_71877 = invoke(stypy.reporting.localization.Localization(__file__, 186, 22), range_71864, *[result_add_71870, result_add_71875], **kwargs_71876)
        
        # Processing the call keyword arguments (line 186)
        kwargs_71878 = {}
        # Getting the type of 'list' (line 186)
        list_71863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'list', False)
        # Calling list(args, kwargs) (line 186)
        list_call_result_71879 = invoke(stypy.reporting.localization.Localization(__file__, 186, 17), list_71863, *[range_call_result_71877], **kwargs_71878)
        
        # Applying the binary operator '+' (line 185)
        result_add_71880 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 39), '+', result_add_71862, list_call_result_71879)
        
        # Assigning a type to the variable 's' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 's', result_add_71880)
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to transpose(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 's' (line 187)
        s_71883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 's', False)
        # Processing the call keyword arguments (line 187)
        kwargs_71884 = {}
        # Getting the type of 'y' (line 187)
        y_71881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'y', False)
        # Obtaining the member 'transpose' of a type (line 187)
        transpose_71882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), y_71881, 'transpose')
        # Calling transpose(args, kwargs) (line 187)
        transpose_call_result_71885 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), transpose_71882, *[s_71883], **kwargs_71884)
        
        # Assigning a type to the variable 'y' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'y', transpose_call_result_71885)
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 188)
        y_71886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', y_71886)
        
        # ################# End of 'derivatives(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derivatives' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_71887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derivatives'
        return stypy_return_type_71887


    @norecursion
    def derivative(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_71888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 32), 'int')
        defaults = [int_71888]
        # Create a new context for function 'derivative'
        module_type_store = module_type_store.open_function_context('derivative', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_localization', localization)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_function_name', '_Interpolator1DWithDerivatives.derivative')
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_param_names_list', ['x', 'der'])
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Interpolator1DWithDerivatives.derivative.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1DWithDerivatives.derivative', ['x', 'der'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derivative', localization, ['x', 'der'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derivative(...)' code ##################

        str_71889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, (-1)), 'str', '\n        Evaluate one derivative of the polynomial at the point x\n\n        Parameters\n        ----------\n        x : array_like\n            Point or points at which to evaluate the derivatives\n\n        der : integer, optional\n            Which derivative to extract. This number includes the\n            function value as 0th derivative.\n\n        Returns\n        -------\n        d : ndarray\n            Derivative interpolated at the x-points.  Shape of d is\n            determined by replacing the interpolation axis in the\n            original array with the shape of x.\n\n        Notes\n        -----\n        This is computed by evaluating all derivatives up to the desired\n        one (using self.derivatives()) and then discarding the rest.\n\n        ')
        
        # Assigning a Call to a Tuple (line 216):
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_71890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
        
        # Call to _prepare_x(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'x' (line 216)
        x_71893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'x', False)
        # Processing the call keyword arguments (line 216)
        kwargs_71894 = {}
        # Getting the type of 'self' (line 216)
        self_71891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'self', False)
        # Obtaining the member '_prepare_x' of a type (line 216)
        _prepare_x_71892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), self_71891, '_prepare_x')
        # Calling _prepare_x(args, kwargs) (line 216)
        _prepare_x_call_result_71895 = invoke(stypy.reporting.localization.Localization(__file__, 216, 21), _prepare_x_71892, *[x_71893], **kwargs_71894)
        
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___71896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), _prepare_x_call_result_71895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_71897 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___71896, int_71890)
        
        # Assigning a type to the variable 'tuple_var_assignment_71413' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_71413', subscript_call_result_71897)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_71898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
        
        # Call to _prepare_x(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'x' (line 216)
        x_71901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'x', False)
        # Processing the call keyword arguments (line 216)
        kwargs_71902 = {}
        # Getting the type of 'self' (line 216)
        self_71899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'self', False)
        # Obtaining the member '_prepare_x' of a type (line 216)
        _prepare_x_71900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), self_71899, '_prepare_x')
        # Calling _prepare_x(args, kwargs) (line 216)
        _prepare_x_call_result_71903 = invoke(stypy.reporting.localization.Localization(__file__, 216, 21), _prepare_x_71900, *[x_71901], **kwargs_71902)
        
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___71904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), _prepare_x_call_result_71903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_71905 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___71904, int_71898)
        
        # Assigning a type to the variable 'tuple_var_assignment_71414' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_71414', subscript_call_result_71905)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_71413' (line 216)
        tuple_var_assignment_71413_71906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_71413')
        # Assigning a type to the variable 'x' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'x', tuple_var_assignment_71413_71906)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_71414' (line 216)
        tuple_var_assignment_71414_71907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_71414')
        # Assigning a type to the variable 'x_shape' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'x_shape', tuple_var_assignment_71414_71907)
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to _evaluate_derivatives(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'x' (line 217)
        x_71910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'x', False)
        # Getting the type of 'der' (line 217)
        der_71911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 42), 'der', False)
        int_71912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 46), 'int')
        # Applying the binary operator '+' (line 217)
        result_add_71913 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 42), '+', der_71911, int_71912)
        
        # Processing the call keyword arguments (line 217)
        kwargs_71914 = {}
        # Getting the type of 'self' (line 217)
        self_71908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'self', False)
        # Obtaining the member '_evaluate_derivatives' of a type (line 217)
        _evaluate_derivatives_71909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), self_71908, '_evaluate_derivatives')
        # Calling _evaluate_derivatives(args, kwargs) (line 217)
        _evaluate_derivatives_call_result_71915 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), _evaluate_derivatives_71909, *[x_71910, result_add_71913], **kwargs_71914)
        
        # Assigning a type to the variable 'y' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'y', _evaluate_derivatives_call_result_71915)
        
        # Call to _finish_y(...): (line 218)
        # Processing the call arguments (line 218)
        
        # Obtaining the type of the subscript
        # Getting the type of 'der' (line 218)
        der_71918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'der', False)
        # Getting the type of 'y' (line 218)
        y_71919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 218)
        getitem___71920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 30), y_71919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 218)
        subscript_call_result_71921 = invoke(stypy.reporting.localization.Localization(__file__, 218, 30), getitem___71920, der_71918)
        
        # Getting the type of 'x_shape' (line 218)
        x_shape_71922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 38), 'x_shape', False)
        # Processing the call keyword arguments (line 218)
        kwargs_71923 = {}
        # Getting the type of 'self' (line 218)
        self_71916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'self', False)
        # Obtaining the member '_finish_y' of a type (line 218)
        _finish_y_71917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), self_71916, '_finish_y')
        # Calling _finish_y(args, kwargs) (line 218)
        _finish_y_call_result_71924 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), _finish_y_71917, *[subscript_call_result_71921, x_shape_71922], **kwargs_71923)
        
        # Assigning a type to the variable 'stypy_return_type' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', _finish_y_call_result_71924)
        
        # ################# End of 'derivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derivative' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_71925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derivative'
        return stypy_return_type_71925


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 0, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Interpolator1DWithDerivatives.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_Interpolator1DWithDerivatives' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), '_Interpolator1DWithDerivatives', _Interpolator1DWithDerivatives)
# Declaration of the 'KroghInterpolator' class
# Getting the type of '_Interpolator1DWithDerivatives' (line 221)
_Interpolator1DWithDerivatives_71926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), '_Interpolator1DWithDerivatives')

class KroghInterpolator(_Interpolator1DWithDerivatives_71926, ):
    str_71927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'str', '\n    Interpolating polynomial for a set of points.\n\n    The polynomial passes through all the pairs (xi,yi). One may\n    additionally specify a number of derivatives at each point xi;\n    this is done by repeating the value xi and specifying the\n    derivatives as successive yi values.\n\n    Allows evaluation of the polynomial and all its derivatives.\n    For reasons of numerical stability, this function does not compute\n    the coefficients of the polynomial, although they can be obtained\n    by evaluating all the derivatives.\n\n    Parameters\n    ----------\n    xi : array_like, length N\n        Known x-coordinates. Must be sorted in increasing order.\n    yi : array_like\n        Known y-coordinates. When an xi occurs two or more times in\n        a row, the corresponding yi\'s represent derivative values.\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values.\n\n    Notes\n    -----\n    Be aware that the algorithms implemented here are not necessarily\n    the most numerically stable known. Moreover, even in a world of\n    exact computation, unless the x coordinates are chosen very\n    carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -\n    polynomial interpolation itself is a very ill-conditioned process\n    due to the Runge phenomenon. In general, even with well-chosen\n    x values, degrees higher than about thirty cause problems with\n    numerical instability in this code.\n\n    Based on [1]_.\n\n    References\n    ----------\n    .. [1] Krogh, "Efficient Algorithms for Polynomial Interpolation\n        and Numerical Differentiation", 1970.\n\n    Examples\n    --------\n    To produce a polynomial that is zero at 0 and 1 and has\n    derivative 2 at 0, call\n\n    >>> from scipy.interpolate import KroghInterpolator\n    >>> KroghInterpolator([0,0,1],[0,2,0])\n\n    This constructs the quadratic 2*X**2-2*X. The derivative condition\n    is indicated by the repeated zero in the xi array; the corresponding\n    yi values are 0, the function value, and 2, the derivative value.\n\n    For another example, given xi, yi, and a derivative ypi for each\n    point, appropriate arrays can be constructed as:\n\n    >>> xi = np.linspace(0, 1, 5)\n    >>> yi, ypi = np.random.rand(2, 5)\n    >>> xi_k, yi_k = np.repeat(xi, 2), np.ravel(np.dstack((yi,ypi)))\n    >>> KroghInterpolator(xi_k, yi_k)\n\n    To produce a vector-valued polynomial, supply a higher-dimensional\n    array for yi:\n\n    >>> KroghInterpolator([0,1],[[2,3],[4,5]])\n\n    This constructs a linear polynomial giving (2,3) at 0 and (4,5) at 1.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_71928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 36), 'int')
        defaults = [int_71928]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KroghInterpolator.__init__', ['xi', 'yi', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xi', 'yi', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'self' (line 293)
        self_71931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 48), 'self', False)
        # Getting the type of 'xi' (line 293)
        xi_71932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 54), 'xi', False)
        # Getting the type of 'yi' (line 293)
        yi_71933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 58), 'yi', False)
        # Getting the type of 'axis' (line 293)
        axis_71934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 62), 'axis', False)
        # Processing the call keyword arguments (line 293)
        kwargs_71935 = {}
        # Getting the type of '_Interpolator1DWithDerivatives' (line 293)
        _Interpolator1DWithDerivatives_71929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), '_Interpolator1DWithDerivatives', False)
        # Obtaining the member '__init__' of a type (line 293)
        init___71930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), _Interpolator1DWithDerivatives_71929, '__init__')
        # Calling __init__(args, kwargs) (line 293)
        init___call_result_71936 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), init___71930, *[self_71931, xi_71932, yi_71933, axis_71934], **kwargs_71935)
        
        
        # Assigning a Call to a Attribute (line 295):
        
        # Assigning a Call to a Attribute (line 295):
        
        # Call to asarray(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'xi' (line 295)
        xi_71939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'xi', False)
        # Processing the call keyword arguments (line 295)
        kwargs_71940 = {}
        # Getting the type of 'np' (line 295)
        np_71937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 295)
        asarray_71938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), np_71937, 'asarray')
        # Calling asarray(args, kwargs) (line 295)
        asarray_call_result_71941 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), asarray_71938, *[xi_71939], **kwargs_71940)
        
        # Getting the type of 'self' (line 295)
        self_71942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self')
        # Setting the type of the member 'xi' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_71942, 'xi', asarray_call_result_71941)
        
        # Assigning a Call to a Attribute (line 296):
        
        # Assigning a Call to a Attribute (line 296):
        
        # Call to _reshape_yi(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'yi' (line 296)
        yi_71945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 35), 'yi', False)
        # Processing the call keyword arguments (line 296)
        kwargs_71946 = {}
        # Getting the type of 'self' (line 296)
        self_71943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 18), 'self', False)
        # Obtaining the member '_reshape_yi' of a type (line 296)
        _reshape_yi_71944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 18), self_71943, '_reshape_yi')
        # Calling _reshape_yi(args, kwargs) (line 296)
        _reshape_yi_call_result_71947 = invoke(stypy.reporting.localization.Localization(__file__, 296, 18), _reshape_yi_71944, *[yi_71945], **kwargs_71946)
        
        # Getting the type of 'self' (line 296)
        self_71948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self')
        # Setting the type of the member 'yi' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_71948, 'yi', _reshape_yi_call_result_71947)
        
        # Assigning a Attribute to a Tuple (line 297):
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_71949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        # Getting the type of 'self' (line 297)
        self_71950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'self')
        # Obtaining the member 'yi' of a type (line 297)
        yi_71951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), self_71950, 'yi')
        # Obtaining the member 'shape' of a type (line 297)
        shape_71952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), yi_71951, 'shape')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___71953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), shape_71952, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_71954 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), getitem___71953, int_71949)
        
        # Assigning a type to the variable 'tuple_var_assignment_71415' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_71415', subscript_call_result_71954)
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_71955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        # Getting the type of 'self' (line 297)
        self_71956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'self')
        # Obtaining the member 'yi' of a type (line 297)
        yi_71957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), self_71956, 'yi')
        # Obtaining the member 'shape' of a type (line 297)
        shape_71958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), yi_71957, 'shape')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___71959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), shape_71958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_71960 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), getitem___71959, int_71955)
        
        # Assigning a type to the variable 'tuple_var_assignment_71416' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_71416', subscript_call_result_71960)
        
        # Assigning a Name to a Attribute (line 297):
        # Getting the type of 'tuple_var_assignment_71415' (line 297)
        tuple_var_assignment_71415_71961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_71415')
        # Getting the type of 'self' (line 297)
        self_71962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self')
        # Setting the type of the member 'n' of a type (line 297)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_71962, 'n', tuple_var_assignment_71415_71961)
        
        # Assigning a Name to a Attribute (line 297):
        # Getting the type of 'tuple_var_assignment_71416' (line 297)
        tuple_var_assignment_71416_71963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_71416')
        # Getting the type of 'self' (line 297)
        self_71964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'self')
        # Setting the type of the member 'r' of a type (line 297)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), self_71964, 'r', tuple_var_assignment_71416_71963)
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to zeros(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_71967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'self' (line 299)
        self_71968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'self', False)
        # Obtaining the member 'n' of a type (line 299)
        n_71969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 22), self_71968, 'n')
        int_71970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 29), 'int')
        # Applying the binary operator '+' (line 299)
        result_add_71971 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 22), '+', n_71969, int_71970)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 22), tuple_71967, result_add_71971)
        # Adding element type (line 299)
        # Getting the type of 'self' (line 299)
        self_71972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'self', False)
        # Obtaining the member 'r' of a type (line 299)
        r_71973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 32), self_71972, 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 22), tuple_71967, r_71973)
        
        # Processing the call keyword arguments (line 299)
        # Getting the type of 'self' (line 299)
        self_71974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 299)
        dtype_71975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 47), self_71974, 'dtype')
        keyword_71976 = dtype_71975
        kwargs_71977 = {'dtype': keyword_71976}
        # Getting the type of 'np' (line 299)
        np_71965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 299)
        zeros_71966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), np_71965, 'zeros')
        # Calling zeros(args, kwargs) (line 299)
        zeros_call_result_71978 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), zeros_71966, *[tuple_71967], **kwargs_71977)
        
        # Assigning a type to the variable 'c' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'c', zeros_call_result_71978)
        
        # Assigning a Subscript to a Subscript (line 300):
        
        # Assigning a Subscript to a Subscript (line 300):
        
        # Obtaining the type of the subscript
        int_71979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 23), 'int')
        # Getting the type of 'self' (line 300)
        self_71980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'self')
        # Obtaining the member 'yi' of a type (line 300)
        yi_71981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), self_71980, 'yi')
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___71982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), yi_71981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_71983 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), getitem___71982, int_71979)
        
        # Getting the type of 'c' (line 300)
        c_71984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'c')
        int_71985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 10), 'int')
        # Storing an element on a container (line 300)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 8), c_71984, (int_71985, subscript_call_result_71983))
        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to zeros(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'tuple' (line 301)
        tuple_71988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 301)
        # Adding element type (line 301)
        # Getting the type of 'self' (line 301)
        self_71989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'self', False)
        # Obtaining the member 'n' of a type (line 301)
        n_71990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 23), self_71989, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 23), tuple_71988, n_71990)
        # Adding element type (line 301)
        # Getting the type of 'self' (line 301)
        self_71991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 31), 'self', False)
        # Obtaining the member 'r' of a type (line 301)
        r_71992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 31), self_71991, 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 23), tuple_71988, r_71992)
        
        # Processing the call keyword arguments (line 301)
        # Getting the type of 'self' (line 301)
        self_71993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 46), 'self', False)
        # Obtaining the member 'dtype' of a type (line 301)
        dtype_71994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 46), self_71993, 'dtype')
        keyword_71995 = dtype_71994
        kwargs_71996 = {'dtype': keyword_71995}
        # Getting the type of 'np' (line 301)
        np_71986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 301)
        zeros_71987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 13), np_71986, 'zeros')
        # Calling zeros(args, kwargs) (line 301)
        zeros_call_result_71997 = invoke(stypy.reporting.localization.Localization(__file__, 301, 13), zeros_71987, *[tuple_71988], **kwargs_71996)
        
        # Assigning a type to the variable 'Vk' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'Vk', zeros_call_result_71997)
        
        
        # Call to xrange(...): (line 302)
        # Processing the call arguments (line 302)
        int_71999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 24), 'int')
        # Getting the type of 'self' (line 302)
        self_72000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 26), 'self', False)
        # Obtaining the member 'n' of a type (line 302)
        n_72001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 26), self_72000, 'n')
        # Processing the call keyword arguments (line 302)
        kwargs_72002 = {}
        # Getting the type of 'xrange' (line 302)
        xrange_71998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 302)
        xrange_call_result_72003 = invoke(stypy.reporting.localization.Localization(__file__, 302, 17), xrange_71998, *[int_71999, n_72001], **kwargs_72002)
        
        # Testing the type of a for loop iterable (line 302)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 302, 8), xrange_call_result_72003)
        # Getting the type of the for loop variable (line 302)
        for_loop_var_72004 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 302, 8), xrange_call_result_72003)
        # Assigning a type to the variable 'k' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'k', for_loop_var_72004)
        # SSA begins for a for statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 303):
        
        # Assigning a Num to a Name (line 303):
        int_72005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 16), 'int')
        # Assigning a type to the variable 's' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 's', int_72005)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 's' (line 304)
        s_72006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 's')
        # Getting the type of 'k' (line 304)
        k_72007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'k')
        # Applying the binary operator '<=' (line 304)
        result_le_72008 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 18), '<=', s_72006, k_72007)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 304)
        k_72009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'k')
        # Getting the type of 's' (line 304)
        s_72010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 's')
        # Applying the binary operator '-' (line 304)
        result_sub_72011 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 32), '-', k_72009, s_72010)
        
        # Getting the type of 'xi' (line 304)
        xi_72012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'xi')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___72013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 29), xi_72012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_72014 = invoke(stypy.reporting.localization.Localization(__file__, 304, 29), getitem___72013, result_sub_72011)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 304)
        k_72015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'k')
        # Getting the type of 'xi' (line 304)
        xi_72016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 40), 'xi')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___72017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 40), xi_72016, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_72018 = invoke(stypy.reporting.localization.Localization(__file__, 304, 40), getitem___72017, k_72015)
        
        # Applying the binary operator '==' (line 304)
        result_eq_72019 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 29), '==', subscript_call_result_72014, subscript_call_result_72018)
        
        # Applying the binary operator 'and' (line 304)
        result_and_keyword_72020 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 18), 'and', result_le_72008, result_eq_72019)
        
        # Testing the type of an if condition (line 304)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 12), result_and_keyword_72020)
        # SSA begins for while statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 's' (line 305)
        s_72021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 's')
        int_72022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
        # Applying the binary operator '+=' (line 305)
        result_iadd_72023 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', s_72021, int_72022)
        # Assigning a type to the variable 's' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 's', result_iadd_72023)
        
        # SSA join for while statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 's' (line 306)
        s_72024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 's')
        int_72025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'int')
        # Applying the binary operator '-=' (line 306)
        result_isub_72026 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 12), '-=', s_72024, int_72025)
        # Assigning a type to the variable 's' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 's', result_isub_72026)
        
        
        # Assigning a BinOp to a Subscript (line 307):
        
        # Assigning a BinOp to a Subscript (line 307):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 307)
        k_72027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'k')
        # Getting the type of 'self' (line 307)
        self_72028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'self')
        # Obtaining the member 'yi' of a type (line 307)
        yi_72029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 20), self_72028, 'yi')
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___72030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 20), yi_72029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_72031 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), getitem___72030, k_72027)
        
        
        # Call to float(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Call to factorial(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 's' (line 307)
        s_72034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 47), 's', False)
        # Processing the call keyword arguments (line 307)
        kwargs_72035 = {}
        # Getting the type of 'factorial' (line 307)
        factorial_72033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 37), 'factorial', False)
        # Calling factorial(args, kwargs) (line 307)
        factorial_call_result_72036 = invoke(stypy.reporting.localization.Localization(__file__, 307, 37), factorial_72033, *[s_72034], **kwargs_72035)
        
        # Processing the call keyword arguments (line 307)
        kwargs_72037 = {}
        # Getting the type of 'float' (line 307)
        float_72032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 31), 'float', False)
        # Calling float(args, kwargs) (line 307)
        float_call_result_72038 = invoke(stypy.reporting.localization.Localization(__file__, 307, 31), float_72032, *[factorial_call_result_72036], **kwargs_72037)
        
        # Applying the binary operator 'div' (line 307)
        result_div_72039 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 20), 'div', subscript_call_result_72031, float_call_result_72038)
        
        # Getting the type of 'Vk' (line 307)
        Vk_72040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'Vk')
        int_72041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 15), 'int')
        # Storing an element on a container (line 307)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), Vk_72040, (int_72041, result_div_72039))
        
        
        # Call to xrange(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'k' (line 308)
        k_72043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'k', False)
        # Getting the type of 's' (line 308)
        s_72044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 30), 's', False)
        # Applying the binary operator '-' (line 308)
        result_sub_72045 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 28), '-', k_72043, s_72044)
        
        # Processing the call keyword arguments (line 308)
        kwargs_72046 = {}
        # Getting the type of 'xrange' (line 308)
        xrange_72042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 308)
        xrange_call_result_72047 = invoke(stypy.reporting.localization.Localization(__file__, 308, 21), xrange_72042, *[result_sub_72045], **kwargs_72046)
        
        # Testing the type of a for loop iterable (line 308)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 308, 12), xrange_call_result_72047)
        # Getting the type of the for loop variable (line 308)
        for_loop_var_72048 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 308, 12), xrange_call_result_72047)
        # Assigning a type to the variable 'i' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'i', for_loop_var_72048)
        # SSA begins for a for statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 309)
        i_72049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 22), 'i')
        # Getting the type of 'xi' (line 309)
        xi_72050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'xi')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___72051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), xi_72050, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_72052 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), getitem___72051, i_72049)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 309)
        k_72053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'k')
        # Getting the type of 'xi' (line 309)
        xi_72054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'xi')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___72055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 28), xi_72054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_72056 = invoke(stypy.reporting.localization.Localization(__file__, 309, 28), getitem___72055, k_72053)
        
        # Applying the binary operator '==' (line 309)
        result_eq_72057 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 19), '==', subscript_call_result_72052, subscript_call_result_72056)
        
        # Testing the type of an if condition (line 309)
        if_condition_72058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 16), result_eq_72057)
        # Assigning a type to the variable 'if_condition_72058' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'if_condition_72058', if_condition_72058)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 310)
        # Processing the call arguments (line 310)
        str_72060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 37), 'str', "Elements if `xi` can't be equal.")
        # Processing the call keyword arguments (line 310)
        kwargs_72061 = {}
        # Getting the type of 'ValueError' (line 310)
        ValueError_72059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 310)
        ValueError_call_result_72062 = invoke(stypy.reporting.localization.Localization(__file__, 310, 26), ValueError_72059, *[str_72060], **kwargs_72061)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 310, 20), ValueError_call_result_72062, 'raise parameter', BaseException)
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 's' (line 311)
        s_72063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 's')
        int_72064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 24), 'int')
        # Applying the binary operator '==' (line 311)
        result_eq_72065 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 19), '==', s_72063, int_72064)
        
        # Testing the type of an if condition (line 311)
        if_condition_72066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 16), result_eq_72065)
        # Assigning a type to the variable 'if_condition_72066' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'if_condition_72066', if_condition_72066)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 312):
        
        # Assigning a BinOp to a Subscript (line 312):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_72067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), 'i')
        # Getting the type of 'c' (line 312)
        c_72068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 31), 'c')
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___72069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 31), c_72068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_72070 = invoke(stypy.reporting.localization.Localization(__file__, 312, 31), getitem___72069, i_72067)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_72071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), 'i')
        # Getting the type of 'Vk' (line 312)
        Vk_72072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 36), 'Vk')
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___72073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 36), Vk_72072, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_72074 = invoke(stypy.reporting.localization.Localization(__file__, 312, 36), getitem___72073, i_72071)
        
        # Applying the binary operator '-' (line 312)
        result_sub_72075 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 31), '-', subscript_call_result_72070, subscript_call_result_72074)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_72076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 47), 'i')
        # Getting the type of 'xi' (line 312)
        xi_72077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 44), 'xi')
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___72078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 44), xi_72077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_72079 = invoke(stypy.reporting.localization.Localization(__file__, 312, 44), getitem___72078, i_72076)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 312)
        k_72080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 53), 'k')
        # Getting the type of 'xi' (line 312)
        xi_72081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 50), 'xi')
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___72082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 50), xi_72081, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_72083 = invoke(stypy.reporting.localization.Localization(__file__, 312, 50), getitem___72082, k_72080)
        
        # Applying the binary operator '-' (line 312)
        result_sub_72084 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 44), '-', subscript_call_result_72079, subscript_call_result_72083)
        
        # Applying the binary operator 'div' (line 312)
        result_div_72085 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 30), 'div', result_sub_72075, result_sub_72084)
        
        # Getting the type of 'Vk' (line 312)
        Vk_72086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'Vk')
        # Getting the type of 'i' (line 312)
        i_72087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'i')
        int_72088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 25), 'int')
        # Applying the binary operator '+' (line 312)
        result_add_72089 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 23), '+', i_72087, int_72088)
        
        # Storing an element on a container (line 312)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 20), Vk_72086, (result_add_72089, result_div_72085))
        # SSA branch for the else part of an if statement (line 311)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Subscript (line 314):
        
        # Assigning a BinOp to a Subscript (line 314):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 314)
        i_72090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'i')
        int_72091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'int')
        # Applying the binary operator '+' (line 314)
        result_add_72092 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 34), '+', i_72090, int_72091)
        
        # Getting the type of 'Vk' (line 314)
        Vk_72093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 31), 'Vk')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___72094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 31), Vk_72093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_72095 = invoke(stypy.reporting.localization.Localization(__file__, 314, 31), getitem___72094, result_add_72092)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 314)
        i_72096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'i')
        # Getting the type of 'Vk' (line 314)
        Vk_72097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 39), 'Vk')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___72098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 39), Vk_72097, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_72099 = invoke(stypy.reporting.localization.Localization(__file__, 314, 39), getitem___72098, i_72096)
        
        # Applying the binary operator '-' (line 314)
        result_sub_72100 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 31), '-', subscript_call_result_72095, subscript_call_result_72099)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 314)
        i_72101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 50), 'i')
        # Getting the type of 'xi' (line 314)
        xi_72102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 47), 'xi')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___72103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 47), xi_72102, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_72104 = invoke(stypy.reporting.localization.Localization(__file__, 314, 47), getitem___72103, i_72101)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 314)
        k_72105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 56), 'k')
        # Getting the type of 'xi' (line 314)
        xi_72106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 53), 'xi')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___72107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 53), xi_72106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_72108 = invoke(stypy.reporting.localization.Localization(__file__, 314, 53), getitem___72107, k_72105)
        
        # Applying the binary operator '-' (line 314)
        result_sub_72109 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 47), '-', subscript_call_result_72104, subscript_call_result_72108)
        
        # Applying the binary operator 'div' (line 314)
        result_div_72110 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 30), 'div', result_sub_72100, result_sub_72109)
        
        # Getting the type of 'Vk' (line 314)
        Vk_72111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'Vk')
        # Getting the type of 'i' (line 314)
        i_72112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 'i')
        int_72113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'int')
        # Applying the binary operator '+' (line 314)
        result_add_72114 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 23), '+', i_72112, int_72113)
        
        # Storing an element on a container (line 314)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 20), Vk_72111, (result_add_72114, result_div_72110))
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Subscript (line 315):
        
        # Assigning a Subscript to a Subscript (line 315):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 315)
        k_72115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'k')
        # Getting the type of 's' (line 315)
        s_72116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 's')
        # Applying the binary operator '-' (line 315)
        result_sub_72117 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 22), '-', k_72115, s_72116)
        
        # Getting the type of 'Vk' (line 315)
        Vk_72118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'Vk')
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___72119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 19), Vk_72118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_72120 = invoke(stypy.reporting.localization.Localization(__file__, 315, 19), getitem___72119, result_sub_72117)
        
        # Getting the type of 'c' (line 315)
        c_72121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'c')
        # Getting the type of 'k' (line 315)
        k_72122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 14), 'k')
        # Storing an element on a container (line 315)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 12), c_72121, (k_72122, subscript_call_result_72120))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 316):
        
        # Assigning a Name to a Attribute (line 316):
        # Getting the type of 'c' (line 316)
        c_72123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 17), 'c')
        # Getting the type of 'self' (line 316)
        self_72124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self')
        # Setting the type of the member 'c' of a type (line 316)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_72124, 'c', c_72123)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _evaluate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_evaluate'
        module_type_store = module_type_store.open_function_context('_evaluate', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_localization', localization)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_type_store', module_type_store)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_function_name', 'KroghInterpolator._evaluate')
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_param_names_list', ['x'])
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_varargs_param_name', None)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_call_defaults', defaults)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_call_varargs', varargs)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KroghInterpolator._evaluate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KroghInterpolator._evaluate', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_evaluate', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_evaluate(...)' code ##################

        
        # Assigning a Num to a Name (line 319):
        
        # Assigning a Num to a Name (line 319):
        int_72125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 13), 'int')
        # Assigning a type to the variable 'pi' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'pi', int_72125)
        
        # Assigning a Call to a Name (line 320):
        
        # Assigning a Call to a Name (line 320):
        
        # Call to zeros(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_72128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        
        # Call to len(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'x' (line 320)
        x_72130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'x', False)
        # Processing the call keyword arguments (line 320)
        kwargs_72131 = {}
        # Getting the type of 'len' (line 320)
        len_72129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 22), 'len', False)
        # Calling len(args, kwargs) (line 320)
        len_call_result_72132 = invoke(stypy.reporting.localization.Localization(__file__, 320, 22), len_72129, *[x_72130], **kwargs_72131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 22), tuple_72128, len_call_result_72132)
        # Adding element type (line 320)
        # Getting the type of 'self' (line 320)
        self_72133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'self', False)
        # Obtaining the member 'r' of a type (line 320)
        r_72134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 30), self_72133, 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 22), tuple_72128, r_72134)
        
        # Processing the call keyword arguments (line 320)
        # Getting the type of 'self' (line 320)
        self_72135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 45), 'self', False)
        # Obtaining the member 'dtype' of a type (line 320)
        dtype_72136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 45), self_72135, 'dtype')
        keyword_72137 = dtype_72136
        kwargs_72138 = {'dtype': keyword_72137}
        # Getting the type of 'np' (line 320)
        np_72126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 320)
        zeros_72127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), np_72126, 'zeros')
        # Calling zeros(args, kwargs) (line 320)
        zeros_call_result_72139 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), zeros_72127, *[tuple_72128], **kwargs_72138)
        
        # Assigning a type to the variable 'p' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'p', zeros_call_result_72139)
        
        # Getting the type of 'p' (line 321)
        p_72140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'p')
        
        # Obtaining the type of the subscript
        int_72141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 20), 'int')
        # Getting the type of 'np' (line 321)
        np_72142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 22), 'np')
        # Obtaining the member 'newaxis' of a type (line 321)
        newaxis_72143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 22), np_72142, 'newaxis')
        slice_72144 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 321, 13), None, None, None)
        # Getting the type of 'self' (line 321)
        self_72145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 13), 'self')
        # Obtaining the member 'c' of a type (line 321)
        c_72146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 13), self_72145, 'c')
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___72147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 13), c_72146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_72148 = invoke(stypy.reporting.localization.Localization(__file__, 321, 13), getitem___72147, (int_72141, newaxis_72143, slice_72144))
        
        # Applying the binary operator '+=' (line 321)
        result_iadd_72149 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), '+=', p_72140, subscript_call_result_72148)
        # Assigning a type to the variable 'p' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'p', result_iadd_72149)
        
        
        
        # Call to range(...): (line 322)
        # Processing the call arguments (line 322)
        int_72151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 23), 'int')
        # Getting the type of 'self' (line 322)
        self_72152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'self', False)
        # Obtaining the member 'n' of a type (line 322)
        n_72153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 26), self_72152, 'n')
        # Processing the call keyword arguments (line 322)
        kwargs_72154 = {}
        # Getting the type of 'range' (line 322)
        range_72150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'range', False)
        # Calling range(args, kwargs) (line 322)
        range_call_result_72155 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), range_72150, *[int_72151, n_72153], **kwargs_72154)
        
        # Testing the type of a for loop iterable (line 322)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 322, 8), range_call_result_72155)
        # Getting the type of the for loop variable (line 322)
        for_loop_var_72156 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 322, 8), range_call_result_72155)
        # Assigning a type to the variable 'k' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'k', for_loop_var_72156)
        # SSA begins for a for statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 323):
        
        # Assigning a BinOp to a Name (line 323):
        # Getting the type of 'x' (line 323)
        x_72157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 323)
        k_72158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 28), 'k')
        int_72159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 30), 'int')
        # Applying the binary operator '-' (line 323)
        result_sub_72160 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 28), '-', k_72158, int_72159)
        
        # Getting the type of 'self' (line 323)
        self_72161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'self')
        # Obtaining the member 'xi' of a type (line 323)
        xi_72162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 20), self_72161, 'xi')
        # Obtaining the member '__getitem__' of a type (line 323)
        getitem___72163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 20), xi_72162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 323)
        subscript_call_result_72164 = invoke(stypy.reporting.localization.Localization(__file__, 323, 20), getitem___72163, result_sub_72160)
        
        # Applying the binary operator '-' (line 323)
        result_sub_72165 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 16), '-', x_72157, subscript_call_result_72164)
        
        # Assigning a type to the variable 'w' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'w', result_sub_72165)
        
        # Assigning a BinOp to a Name (line 324):
        
        # Assigning a BinOp to a Name (line 324):
        # Getting the type of 'w' (line 324)
        w_72166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 17), 'w')
        # Getting the type of 'pi' (line 324)
        pi_72167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'pi')
        # Applying the binary operator '*' (line 324)
        result_mul_72168 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 17), '*', w_72166, pi_72167)
        
        # Assigning a type to the variable 'pi' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'pi', result_mul_72168)
        
        # Getting the type of 'p' (line 325)
        p_72169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'p')
        
        # Obtaining the type of the subscript
        slice_72170 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 325, 17), None, None, None)
        # Getting the type of 'np' (line 325)
        np_72171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'np')
        # Obtaining the member 'newaxis' of a type (line 325)
        newaxis_72172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 22), np_72171, 'newaxis')
        # Getting the type of 'pi' (line 325)
        pi_72173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), 'pi')
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___72174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 17), pi_72173, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_72175 = invoke(stypy.reporting.localization.Localization(__file__, 325, 17), getitem___72174, (slice_72170, newaxis_72172))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 325)
        k_72176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 43), 'k')
        # Getting the type of 'self' (line 325)
        self_72177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 36), 'self')
        # Obtaining the member 'c' of a type (line 325)
        c_72178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 36), self_72177, 'c')
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___72179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 36), c_72178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_72180 = invoke(stypy.reporting.localization.Localization(__file__, 325, 36), getitem___72179, k_72176)
        
        # Applying the binary operator '*' (line 325)
        result_mul_72181 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 17), '*', subscript_call_result_72175, subscript_call_result_72180)
        
        # Applying the binary operator '+=' (line 325)
        result_iadd_72182 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 12), '+=', p_72169, result_mul_72181)
        # Assigning a type to the variable 'p' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'p', result_iadd_72182)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'p' (line 326)
        p_72183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'stypy_return_type', p_72183)
        
        # ################# End of '_evaluate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_evaluate' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_72184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_evaluate'
        return stypy_return_type_72184


    @norecursion
    def _evaluate_derivatives(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 328)
        None_72185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 43), 'None')
        defaults = [None_72185]
        # Create a new context for function '_evaluate_derivatives'
        module_type_store = module_type_store.open_function_context('_evaluate_derivatives', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_localization', localization)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_type_store', module_type_store)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_function_name', 'KroghInterpolator._evaluate_derivatives')
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_param_names_list', ['x', 'der'])
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_varargs_param_name', None)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_call_defaults', defaults)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_call_varargs', varargs)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KroghInterpolator._evaluate_derivatives.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KroghInterpolator._evaluate_derivatives', ['x', 'der'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_evaluate_derivatives', localization, ['x', 'der'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_evaluate_derivatives(...)' code ##################

        
        # Assigning a Attribute to a Name (line 329):
        
        # Assigning a Attribute to a Name (line 329):
        # Getting the type of 'self' (line 329)
        self_72186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'self')
        # Obtaining the member 'n' of a type (line 329)
        n_72187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), self_72186, 'n')
        # Assigning a type to the variable 'n' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'n', n_72187)
        
        # Assigning a Attribute to a Name (line 330):
        
        # Assigning a Attribute to a Name (line 330):
        # Getting the type of 'self' (line 330)
        self_72188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'self')
        # Obtaining the member 'r' of a type (line 330)
        r_72189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), self_72188, 'r')
        # Assigning a type to the variable 'r' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'r', r_72189)
        
        # Type idiom detected: calculating its left and rigth part (line 332)
        # Getting the type of 'der' (line 332)
        der_72190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'der')
        # Getting the type of 'None' (line 332)
        None_72191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'None')
        
        (may_be_72192, more_types_in_union_72193) = may_be_none(der_72190, None_72191)

        if may_be_72192:

            if more_types_in_union_72193:
                # Runtime conditional SSA (line 332)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 333):
            
            # Assigning a Attribute to a Name (line 333):
            # Getting the type of 'self' (line 333)
            self_72194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 18), 'self')
            # Obtaining the member 'n' of a type (line 333)
            n_72195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 18), self_72194, 'n')
            # Assigning a type to the variable 'der' (line 333)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'der', n_72195)

            if more_types_in_union_72193:
                # SSA join for if statement (line 332)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to zeros(...): (line 334)
        # Processing the call arguments (line 334)
        
        # Obtaining an instance of the builtin type 'tuple' (line 334)
        tuple_72198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 334)
        # Adding element type (line 334)
        # Getting the type of 'n' (line 334)
        n_72199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 23), tuple_72198, n_72199)
        # Adding element type (line 334)
        
        # Call to len(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'x' (line 334)
        x_72201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 30), 'x', False)
        # Processing the call keyword arguments (line 334)
        kwargs_72202 = {}
        # Getting the type of 'len' (line 334)
        len_72200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 26), 'len', False)
        # Calling len(args, kwargs) (line 334)
        len_call_result_72203 = invoke(stypy.reporting.localization.Localization(__file__, 334, 26), len_72200, *[x_72201], **kwargs_72202)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 23), tuple_72198, len_call_result_72203)
        
        # Processing the call keyword arguments (line 334)
        kwargs_72204 = {}
        # Getting the type of 'np' (line 334)
        np_72196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 334)
        zeros_72197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 13), np_72196, 'zeros')
        # Calling zeros(args, kwargs) (line 334)
        zeros_call_result_72205 = invoke(stypy.reporting.localization.Localization(__file__, 334, 13), zeros_72197, *[tuple_72198], **kwargs_72204)
        
        # Assigning a type to the variable 'pi' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'pi', zeros_call_result_72205)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to zeros(...): (line 335)
        # Processing the call arguments (line 335)
        
        # Obtaining an instance of the builtin type 'tuple' (line 335)
        tuple_72208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 335)
        # Adding element type (line 335)
        # Getting the type of 'n' (line 335)
        n_72209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 22), tuple_72208, n_72209)
        # Adding element type (line 335)
        
        # Call to len(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'x' (line 335)
        x_72211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 29), 'x', False)
        # Processing the call keyword arguments (line 335)
        kwargs_72212 = {}
        # Getting the type of 'len' (line 335)
        len_72210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'len', False)
        # Calling len(args, kwargs) (line 335)
        len_call_result_72213 = invoke(stypy.reporting.localization.Localization(__file__, 335, 25), len_72210, *[x_72211], **kwargs_72212)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 22), tuple_72208, len_call_result_72213)
        
        # Processing the call keyword arguments (line 335)
        kwargs_72214 = {}
        # Getting the type of 'np' (line 335)
        np_72206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 335)
        zeros_72207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), np_72206, 'zeros')
        # Calling zeros(args, kwargs) (line 335)
        zeros_call_result_72215 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), zeros_72207, *[tuple_72208], **kwargs_72214)
        
        # Assigning a type to the variable 'w' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'w', zeros_call_result_72215)
        
        # Assigning a Num to a Subscript (line 336):
        
        # Assigning a Num to a Subscript (line 336):
        int_72216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'int')
        # Getting the type of 'pi' (line 336)
        pi_72217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'pi')
        int_72218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 11), 'int')
        # Storing an element on a container (line 336)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), pi_72217, (int_72218, int_72216))
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Call to zeros(...): (line 337)
        # Processing the call arguments (line 337)
        
        # Obtaining an instance of the builtin type 'tuple' (line 337)
        tuple_72221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 337)
        # Adding element type (line 337)
        
        # Call to len(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'x' (line 337)
        x_72223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 26), 'x', False)
        # Processing the call keyword arguments (line 337)
        kwargs_72224 = {}
        # Getting the type of 'len' (line 337)
        len_72222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'len', False)
        # Calling len(args, kwargs) (line 337)
        len_call_result_72225 = invoke(stypy.reporting.localization.Localization(__file__, 337, 22), len_72222, *[x_72223], **kwargs_72224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 22), tuple_72221, len_call_result_72225)
        # Adding element type (line 337)
        # Getting the type of 'self' (line 337)
        self_72226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'self', False)
        # Obtaining the member 'r' of a type (line 337)
        r_72227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 30), self_72226, 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 22), tuple_72221, r_72227)
        
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'self' (line 337)
        self_72228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 45), 'self', False)
        # Obtaining the member 'dtype' of a type (line 337)
        dtype_72229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 45), self_72228, 'dtype')
        keyword_72230 = dtype_72229
        kwargs_72231 = {'dtype': keyword_72230}
        # Getting the type of 'np' (line 337)
        np_72219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 337)
        zeros_72220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), np_72219, 'zeros')
        # Calling zeros(args, kwargs) (line 337)
        zeros_call_result_72232 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), zeros_72220, *[tuple_72221], **kwargs_72231)
        
        # Assigning a type to the variable 'p' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'p', zeros_call_result_72232)
        
        # Getting the type of 'p' (line 338)
        p_72233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'p')
        
        # Obtaining the type of the subscript
        int_72234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'int')
        # Getting the type of 'np' (line 338)
        np_72235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 23), 'np')
        # Obtaining the member 'newaxis' of a type (line 338)
        newaxis_72236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 23), np_72235, 'newaxis')
        slice_72237 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 13), None, None, None)
        # Getting the type of 'self' (line 338)
        self_72238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'self')
        # Obtaining the member 'c' of a type (line 338)
        c_72239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 13), self_72238, 'c')
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___72240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 13), c_72239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_72241 = invoke(stypy.reporting.localization.Localization(__file__, 338, 13), getitem___72240, (int_72234, newaxis_72236, slice_72237))
        
        # Applying the binary operator '+=' (line 338)
        result_iadd_72242 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 8), '+=', p_72233, subscript_call_result_72241)
        # Assigning a type to the variable 'p' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'p', result_iadd_72242)
        
        
        
        # Call to xrange(...): (line 340)
        # Processing the call arguments (line 340)
        int_72244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 24), 'int')
        # Getting the type of 'n' (line 340)
        n_72245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), 'n', False)
        # Processing the call keyword arguments (line 340)
        kwargs_72246 = {}
        # Getting the type of 'xrange' (line 340)
        xrange_72243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 340)
        xrange_call_result_72247 = invoke(stypy.reporting.localization.Localization(__file__, 340, 17), xrange_72243, *[int_72244, n_72245], **kwargs_72246)
        
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), xrange_call_result_72247)
        # Getting the type of the for loop variable (line 340)
        for_loop_var_72248 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), xrange_call_result_72247)
        # Assigning a type to the variable 'k' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'k', for_loop_var_72248)
        # SSA begins for a for statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 341):
        
        # Assigning a BinOp to a Subscript (line 341):
        # Getting the type of 'x' (line 341)
        x_72249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 341)
        k_72250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 33), 'k')
        int_72251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'int')
        # Applying the binary operator '-' (line 341)
        result_sub_72252 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 33), '-', k_72250, int_72251)
        
        # Getting the type of 'self' (line 341)
        self_72253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'self')
        # Obtaining the member 'xi' of a type (line 341)
        xi_72254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 25), self_72253, 'xi')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___72255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 25), xi_72254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_72256 = invoke(stypy.reporting.localization.Localization(__file__, 341, 25), getitem___72255, result_sub_72252)
        
        # Applying the binary operator '-' (line 341)
        result_sub_72257 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 21), '-', x_72249, subscript_call_result_72256)
        
        # Getting the type of 'w' (line 341)
        w_72258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'w')
        # Getting the type of 'k' (line 341)
        k_72259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 14), 'k')
        int_72260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 16), 'int')
        # Applying the binary operator '-' (line 341)
        result_sub_72261 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 14), '-', k_72259, int_72260)
        
        # Storing an element on a container (line 341)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 12), w_72258, (result_sub_72261, result_sub_72257))
        
        # Assigning a BinOp to a Subscript (line 342):
        
        # Assigning a BinOp to a Subscript (line 342):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 342)
        k_72262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'k')
        int_72263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 24), 'int')
        # Applying the binary operator '-' (line 342)
        result_sub_72264 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 22), '-', k_72262, int_72263)
        
        # Getting the type of 'w' (line 342)
        w_72265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'w')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___72266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), w_72265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_72267 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), getitem___72266, result_sub_72264)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 342)
        k_72268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 32), 'k')
        int_72269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 34), 'int')
        # Applying the binary operator '-' (line 342)
        result_sub_72270 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 32), '-', k_72268, int_72269)
        
        # Getting the type of 'pi' (line 342)
        pi_72271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'pi')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___72272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 29), pi_72271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_72273 = invoke(stypy.reporting.localization.Localization(__file__, 342, 29), getitem___72272, result_sub_72270)
        
        # Applying the binary operator '*' (line 342)
        result_mul_72274 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 20), '*', subscript_call_result_72267, subscript_call_result_72273)
        
        # Getting the type of 'pi' (line 342)
        pi_72275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'pi')
        # Getting the type of 'k' (line 342)
        k_72276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'k')
        # Storing an element on a container (line 342)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 12), pi_72275, (k_72276, result_mul_72274))
        
        # Getting the type of 'p' (line 343)
        p_72277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'p')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 343)
        k_72278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'k')
        slice_72279 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 343, 17), None, None, None)
        # Getting the type of 'np' (line 343)
        np_72280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'np')
        # Obtaining the member 'newaxis' of a type (line 343)
        newaxis_72281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 26), np_72280, 'newaxis')
        # Getting the type of 'pi' (line 343)
        pi_72282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 17), 'pi')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___72283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 17), pi_72282, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_72284 = invoke(stypy.reporting.localization.Localization(__file__, 343, 17), getitem___72283, (k_72278, slice_72279, newaxis_72281))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 343)
        k_72285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'k')
        # Getting the type of 'self' (line 343)
        self_72286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'self')
        # Obtaining the member 'c' of a type (line 343)
        c_72287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 40), self_72286, 'c')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___72288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 40), c_72287, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_72289 = invoke(stypy.reporting.localization.Localization(__file__, 343, 40), getitem___72288, k_72285)
        
        # Applying the binary operator '*' (line 343)
        result_mul_72290 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 17), '*', subscript_call_result_72284, subscript_call_result_72289)
        
        # Applying the binary operator '+=' (line 343)
        result_iadd_72291 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 12), '+=', p_72277, result_mul_72290)
        # Assigning a type to the variable 'p' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'p', result_iadd_72291)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 345):
        
        # Assigning a Call to a Name (line 345):
        
        # Call to zeros(...): (line 345)
        # Processing the call arguments (line 345)
        
        # Obtaining an instance of the builtin type 'tuple' (line 345)
        tuple_72294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 345)
        # Adding element type (line 345)
        
        # Call to max(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'der' (line 345)
        der_72296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 27), 'der', False)
        # Getting the type of 'n' (line 345)
        n_72297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 32), 'n', False)
        int_72298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 34), 'int')
        # Applying the binary operator '+' (line 345)
        result_add_72299 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 32), '+', n_72297, int_72298)
        
        # Processing the call keyword arguments (line 345)
        kwargs_72300 = {}
        # Getting the type of 'max' (line 345)
        max_72295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 'max', False)
        # Calling max(args, kwargs) (line 345)
        max_call_result_72301 = invoke(stypy.reporting.localization.Localization(__file__, 345, 23), max_72295, *[der_72296, result_add_72299], **kwargs_72300)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 23), tuple_72294, max_call_result_72301)
        # Adding element type (line 345)
        
        # Call to len(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'x' (line 345)
        x_72303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 42), 'x', False)
        # Processing the call keyword arguments (line 345)
        kwargs_72304 = {}
        # Getting the type of 'len' (line 345)
        len_72302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 38), 'len', False)
        # Calling len(args, kwargs) (line 345)
        len_call_result_72305 = invoke(stypy.reporting.localization.Localization(__file__, 345, 38), len_72302, *[x_72303], **kwargs_72304)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 23), tuple_72294, len_call_result_72305)
        # Adding element type (line 345)
        # Getting the type of 'r' (line 345)
        r_72306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 46), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 23), tuple_72294, r_72306)
        
        # Processing the call keyword arguments (line 345)
        # Getting the type of 'self' (line 345)
        self_72307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 56), 'self', False)
        # Obtaining the member 'dtype' of a type (line 345)
        dtype_72308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 56), self_72307, 'dtype')
        keyword_72309 = dtype_72308
        kwargs_72310 = {'dtype': keyword_72309}
        # Getting the type of 'np' (line 345)
        np_72292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 345)
        zeros_72293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 13), np_72292, 'zeros')
        # Calling zeros(args, kwargs) (line 345)
        zeros_call_result_72311 = invoke(stypy.reporting.localization.Localization(__file__, 345, 13), zeros_72293, *[tuple_72294], **kwargs_72310)
        
        # Assigning a type to the variable 'cn' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'cn', zeros_call_result_72311)
        
        # Getting the type of 'cn' (line 346)
        cn_72312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'cn')
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 346)
        n_72313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'n')
        int_72314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 14), 'int')
        # Applying the binary operator '+' (line 346)
        result_add_72315 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 12), '+', n_72313, int_72314)
        
        slice_72316 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 8), None, result_add_72315, None)
        slice_72317 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 8), None, None, None)
        slice_72318 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 8), None, None, None)
        # Getting the type of 'cn' (line 346)
        cn_72319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'cn')
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___72320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), cn_72319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_72321 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), getitem___72320, (slice_72316, slice_72317, slice_72318))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 346)
        n_72322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 34), 'n')
        int_72323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 36), 'int')
        # Applying the binary operator '+' (line 346)
        result_add_72324 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 34), '+', n_72322, int_72323)
        
        slice_72325 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 26), None, result_add_72324, None)
        # Getting the type of 'np' (line 346)
        np_72326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 39), 'np')
        # Obtaining the member 'newaxis' of a type (line 346)
        newaxis_72327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 39), np_72326, 'newaxis')
        slice_72328 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 26), None, None, None)
        # Getting the type of 'self' (line 346)
        self_72329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 26), 'self')
        # Obtaining the member 'c' of a type (line 346)
        c_72330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 26), self_72329, 'c')
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___72331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 26), c_72330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_72332 = invoke(stypy.reporting.localization.Localization(__file__, 346, 26), getitem___72331, (slice_72325, newaxis_72327, slice_72328))
        
        # Applying the binary operator '+=' (line 346)
        result_iadd_72333 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 8), '+=', subscript_call_result_72321, subscript_call_result_72332)
        # Getting the type of 'cn' (line 346)
        cn_72334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'cn')
        # Getting the type of 'n' (line 346)
        n_72335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'n')
        int_72336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 14), 'int')
        # Applying the binary operator '+' (line 346)
        result_add_72337 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 12), '+', n_72335, int_72336)
        
        slice_72338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 8), None, result_add_72337, None)
        slice_72339 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 8), None, None, None)
        slice_72340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 8), None, None, None)
        # Storing an element on a container (line 346)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 8), cn_72334, ((slice_72338, slice_72339, slice_72340), result_iadd_72333))
        
        
        # Assigning a Name to a Subscript (line 347):
        
        # Assigning a Name to a Subscript (line 347):
        # Getting the type of 'p' (line 347)
        p_72341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'p')
        # Getting the type of 'cn' (line 347)
        cn_72342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'cn')
        int_72343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 11), 'int')
        # Storing an element on a container (line 347)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), cn_72342, (int_72343, p_72341))
        
        
        # Call to xrange(...): (line 348)
        # Processing the call arguments (line 348)
        int_72345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 24), 'int')
        # Getting the type of 'n' (line 348)
        n_72346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'n', False)
        # Processing the call keyword arguments (line 348)
        kwargs_72347 = {}
        # Getting the type of 'xrange' (line 348)
        xrange_72344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 348)
        xrange_call_result_72348 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), xrange_72344, *[int_72345, n_72346], **kwargs_72347)
        
        # Testing the type of a for loop iterable (line 348)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 348, 8), xrange_call_result_72348)
        # Getting the type of the for loop variable (line 348)
        for_loop_var_72349 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 348, 8), xrange_call_result_72348)
        # Assigning a type to the variable 'k' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'k', for_loop_var_72349)
        # SSA begins for a for statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 349)
        # Processing the call arguments (line 349)
        int_72351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'int')
        # Getting the type of 'n' (line 349)
        n_72352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 31), 'n', False)
        # Getting the type of 'k' (line 349)
        k_72353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 33), 'k', False)
        # Applying the binary operator '-' (line 349)
        result_sub_72354 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 31), '-', n_72352, k_72353)
        
        int_72355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 35), 'int')
        # Applying the binary operator '+' (line 349)
        result_add_72356 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 34), '+', result_sub_72354, int_72355)
        
        # Processing the call keyword arguments (line 349)
        kwargs_72357 = {}
        # Getting the type of 'xrange' (line 349)
        xrange_72350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 349)
        xrange_call_result_72358 = invoke(stypy.reporting.localization.Localization(__file__, 349, 21), xrange_72350, *[int_72351, result_add_72356], **kwargs_72357)
        
        # Testing the type of a for loop iterable (line 349)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 349, 12), xrange_call_result_72358)
        # Getting the type of the for loop variable (line 349)
        for_loop_var_72359 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 349, 12), xrange_call_result_72358)
        # Assigning a type to the variable 'i' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'i', for_loop_var_72359)
        # SSA begins for a for statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 350):
        
        # Assigning a BinOp to a Subscript (line 350):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 350)
        k_72360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'k')
        # Getting the type of 'i' (line 350)
        i_72361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'i')
        # Applying the binary operator '+' (line 350)
        result_add_72362 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 26), '+', k_72360, i_72361)
        
        int_72363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 30), 'int')
        # Applying the binary operator '-' (line 350)
        result_sub_72364 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 29), '-', result_add_72362, int_72363)
        
        # Getting the type of 'w' (line 350)
        w_72365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'w')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___72366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 24), w_72365, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_72367 = invoke(stypy.reporting.localization.Localization(__file__, 350, 24), getitem___72366, result_sub_72364)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 350)
        i_72368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'i')
        int_72369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 38), 'int')
        # Applying the binary operator '-' (line 350)
        result_sub_72370 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 36), '-', i_72368, int_72369)
        
        # Getting the type of 'pi' (line 350)
        pi_72371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'pi')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___72372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 33), pi_72371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_72373 = invoke(stypy.reporting.localization.Localization(__file__, 350, 33), getitem___72372, result_sub_72370)
        
        # Applying the binary operator '*' (line 350)
        result_mul_72374 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 24), '*', subscript_call_result_72367, subscript_call_result_72373)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 350)
        i_72375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 46), 'i')
        # Getting the type of 'pi' (line 350)
        pi_72376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 43), 'pi')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___72377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 43), pi_72376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_72378 = invoke(stypy.reporting.localization.Localization(__file__, 350, 43), getitem___72377, i_72375)
        
        # Applying the binary operator '+' (line 350)
        result_add_72379 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 24), '+', result_mul_72374, subscript_call_result_72378)
        
        # Getting the type of 'pi' (line 350)
        pi_72380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'pi')
        # Getting the type of 'i' (line 350)
        i_72381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'i')
        # Storing an element on a container (line 350)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 16), pi_72380, (i_72381, result_add_72379))
        
        # Assigning a BinOp to a Subscript (line 351):
        
        # Assigning a BinOp to a Subscript (line 351):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 351)
        k_72382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 27), 'k')
        # Getting the type of 'cn' (line 351)
        cn_72383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'cn')
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___72384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), cn_72383, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_72385 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), getitem___72384, k_72382)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 351)
        i_72386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 35), 'i')
        slice_72387 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 351, 32), None, None, None)
        # Getting the type of 'np' (line 351)
        np_72388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 41), 'np')
        # Obtaining the member 'newaxis' of a type (line 351)
        newaxis_72389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 41), np_72388, 'newaxis')
        # Getting the type of 'pi' (line 351)
        pi_72390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'pi')
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___72391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 32), pi_72390, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_72392 = invoke(stypy.reporting.localization.Localization(__file__, 351, 32), getitem___72391, (i_72386, slice_72387, newaxis_72389))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 351)
        k_72393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 56), 'k')
        # Getting the type of 'i' (line 351)
        i_72394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 58), 'i')
        # Applying the binary operator '+' (line 351)
        result_add_72395 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 56), '+', k_72393, i_72394)
        
        # Getting the type of 'cn' (line 351)
        cn_72396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 53), 'cn')
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___72397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 53), cn_72396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_72398 = invoke(stypy.reporting.localization.Localization(__file__, 351, 53), getitem___72397, result_add_72395)
        
        # Applying the binary operator '*' (line 351)
        result_mul_72399 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 32), '*', subscript_call_result_72392, subscript_call_result_72398)
        
        # Applying the binary operator '+' (line 351)
        result_add_72400 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 24), '+', subscript_call_result_72385, result_mul_72399)
        
        # Getting the type of 'cn' (line 351)
        cn_72401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'cn')
        # Getting the type of 'k' (line 351)
        k_72402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'k')
        # Storing an element on a container (line 351)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 16), cn_72401, (k_72402, result_add_72400))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cn' (line 352)
        cn_72403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'cn')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 352)
        k_72404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'k')
        # Getting the type of 'cn' (line 352)
        cn_72405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'cn')
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___72406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), cn_72405, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_72407 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), getitem___72406, k_72404)
        
        
        # Call to factorial(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'k' (line 352)
        k_72409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 31), 'k', False)
        # Processing the call keyword arguments (line 352)
        kwargs_72410 = {}
        # Getting the type of 'factorial' (line 352)
        factorial_72408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 21), 'factorial', False)
        # Calling factorial(args, kwargs) (line 352)
        factorial_call_result_72411 = invoke(stypy.reporting.localization.Localization(__file__, 352, 21), factorial_72408, *[k_72409], **kwargs_72410)
        
        # Applying the binary operator '*=' (line 352)
        result_imul_72412 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 12), '*=', subscript_call_result_72407, factorial_call_result_72411)
        # Getting the type of 'cn' (line 352)
        cn_72413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'cn')
        # Getting the type of 'k' (line 352)
        k_72414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'k')
        # Storing an element on a container (line 352)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 12), cn_72413, (k_72414, result_imul_72412))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Subscript (line 354):
        
        # Assigning a Num to a Subscript (line 354):
        int_72415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 22), 'int')
        # Getting the type of 'cn' (line 354)
        cn_72416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'cn')
        # Getting the type of 'n' (line 354)
        n_72417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'n')
        slice_72418 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 8), None, None, None)
        slice_72419 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 8), None, None, None)
        # Storing an element on a container (line 354)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), cn_72416, ((n_72417, slice_72418, slice_72419), int_72415))
        
        # Obtaining the type of the subscript
        # Getting the type of 'der' (line 355)
        der_72420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'der')
        slice_72421 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 355, 15), None, der_72420, None)
        # Getting the type of 'cn' (line 355)
        cn_72422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), 'cn')
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___72423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), cn_72422, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_72424 = invoke(stypy.reporting.localization.Localization(__file__, 355, 15), getitem___72423, slice_72421)
        
        # Assigning a type to the variable 'stypy_return_type' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'stypy_return_type', subscript_call_result_72424)
        
        # ################# End of '_evaluate_derivatives(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_evaluate_derivatives' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_72425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_evaluate_derivatives'
        return stypy_return_type_72425


# Assigning a type to the variable 'KroghInterpolator' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'KroghInterpolator', KroghInterpolator)

@norecursion
def krogh_interpolate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_72426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 37), 'int')
    int_72427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 45), 'int')
    defaults = [int_72426, int_72427]
    # Create a new context for function 'krogh_interpolate'
    module_type_store = module_type_store.open_function_context('krogh_interpolate', 358, 0, False)
    
    # Passed parameters checking function
    krogh_interpolate.stypy_localization = localization
    krogh_interpolate.stypy_type_of_self = None
    krogh_interpolate.stypy_type_store = module_type_store
    krogh_interpolate.stypy_function_name = 'krogh_interpolate'
    krogh_interpolate.stypy_param_names_list = ['xi', 'yi', 'x', 'der', 'axis']
    krogh_interpolate.stypy_varargs_param_name = None
    krogh_interpolate.stypy_kwargs_param_name = None
    krogh_interpolate.stypy_call_defaults = defaults
    krogh_interpolate.stypy_call_varargs = varargs
    krogh_interpolate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'krogh_interpolate', ['xi', 'yi', 'x', 'der', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'krogh_interpolate', localization, ['xi', 'yi', 'x', 'der', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'krogh_interpolate(...)' code ##################

    str_72428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, (-1)), 'str', "\n    Convenience function for polynomial interpolation.\n\n    See `KroghInterpolator` for more details.\n\n    Parameters\n    ----------\n    xi : array_like\n        Known x-coordinates.\n    yi : array_like\n        Known y-coordinates, of shape ``(xi.size, R)``.  Interpreted as\n        vectors of length R, or scalars if R=1.\n    x : array_like\n        Point or points at which to evaluate the derivatives.\n    der : int or list, optional\n        How many derivatives to extract; None for all potentially\n        nonzero derivatives (that is a number equal to the number\n        of points), or a list of derivatives to extract. This number\n        includes the function value as 0th derivative.\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values.\n\n    Returns\n    -------\n    d : ndarray\n        If the interpolator's values are R-dimensional then the\n        returned array will be the number of derivatives by N by R.\n        If `x` is a scalar, the middle dimension will be dropped; if\n        the `yi` are scalars then the last dimension will be dropped.\n\n    See Also\n    --------\n    KroghInterpolator\n\n    Notes\n    -----\n    Construction of the interpolating polynomial is a relatively expensive\n    process. If you want to evaluate it repeatedly consider using the class\n    KroghInterpolator (which is what this function uses).\n\n    ")
    
    # Assigning a Call to a Name (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to KroghInterpolator(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'xi' (line 400)
    xi_72430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 26), 'xi', False)
    # Getting the type of 'yi' (line 400)
    yi_72431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 30), 'yi', False)
    # Processing the call keyword arguments (line 400)
    # Getting the type of 'axis' (line 400)
    axis_72432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 39), 'axis', False)
    keyword_72433 = axis_72432
    kwargs_72434 = {'axis': keyword_72433}
    # Getting the type of 'KroghInterpolator' (line 400)
    KroghInterpolator_72429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'KroghInterpolator', False)
    # Calling KroghInterpolator(args, kwargs) (line 400)
    KroghInterpolator_call_result_72435 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), KroghInterpolator_72429, *[xi_72430, yi_72431], **kwargs_72434)
    
    # Assigning a type to the variable 'P' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'P', KroghInterpolator_call_result_72435)
    
    
    # Getting the type of 'der' (line 401)
    der_72436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 7), 'der')
    int_72437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 14), 'int')
    # Applying the binary operator '==' (line 401)
    result_eq_72438 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 7), '==', der_72436, int_72437)
    
    # Testing the type of an if condition (line 401)
    if_condition_72439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 4), result_eq_72438)
    # Assigning a type to the variable 'if_condition_72439' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'if_condition_72439', if_condition_72439)
    # SSA begins for if statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to P(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'x' (line 402)
    x_72441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 17), 'x', False)
    # Processing the call keyword arguments (line 402)
    kwargs_72442 = {}
    # Getting the type of 'P' (line 402)
    P_72440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'P', False)
    # Calling P(args, kwargs) (line 402)
    P_call_result_72443 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), P_72440, *[x_72441], **kwargs_72442)
    
    # Assigning a type to the variable 'stypy_return_type' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'stypy_return_type', P_call_result_72443)
    # SSA branch for the else part of an if statement (line 401)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to _isscalar(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'der' (line 403)
    der_72445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'der', False)
    # Processing the call keyword arguments (line 403)
    kwargs_72446 = {}
    # Getting the type of '_isscalar' (line 403)
    _isscalar_72444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 9), '_isscalar', False)
    # Calling _isscalar(args, kwargs) (line 403)
    _isscalar_call_result_72447 = invoke(stypy.reporting.localization.Localization(__file__, 403, 9), _isscalar_72444, *[der_72445], **kwargs_72446)
    
    # Testing the type of an if condition (line 403)
    if_condition_72448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 9), _isscalar_call_result_72447)
    # Assigning a type to the variable 'if_condition_72448' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 9), 'if_condition_72448', if_condition_72448)
    # SSA begins for if statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to derivative(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'x' (line 404)
    x_72451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 28), 'x', False)
    # Processing the call keyword arguments (line 404)
    # Getting the type of 'der' (line 404)
    der_72452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 34), 'der', False)
    keyword_72453 = der_72452
    kwargs_72454 = {'der': keyword_72453}
    # Getting the type of 'P' (line 404)
    P_72449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'P', False)
    # Obtaining the member 'derivative' of a type (line 404)
    derivative_72450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), P_72449, 'derivative')
    # Calling derivative(args, kwargs) (line 404)
    derivative_call_result_72455 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), derivative_72450, *[x_72451], **kwargs_72454)
    
    # Assigning a type to the variable 'stypy_return_type' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', derivative_call_result_72455)
    # SSA branch for the else part of an if statement (line 403)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    # Getting the type of 'der' (line 406)
    der_72456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 51), 'der')
    
    # Call to derivatives(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'x' (line 406)
    x_72459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 29), 'x', False)
    # Processing the call keyword arguments (line 406)
    
    # Call to amax(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'der' (line 406)
    der_72462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 43), 'der', False)
    # Processing the call keyword arguments (line 406)
    kwargs_72463 = {}
    # Getting the type of 'np' (line 406)
    np_72460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'np', False)
    # Obtaining the member 'amax' of a type (line 406)
    amax_72461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 35), np_72460, 'amax')
    # Calling amax(args, kwargs) (line 406)
    amax_call_result_72464 = invoke(stypy.reporting.localization.Localization(__file__, 406, 35), amax_72461, *[der_72462], **kwargs_72463)
    
    int_72465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 48), 'int')
    # Applying the binary operator '+' (line 406)
    result_add_72466 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 35), '+', amax_call_result_72464, int_72465)
    
    keyword_72467 = result_add_72466
    kwargs_72468 = {'der': keyword_72467}
    # Getting the type of 'P' (line 406)
    P_72457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'P', False)
    # Obtaining the member 'derivatives' of a type (line 406)
    derivatives_72458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 15), P_72457, 'derivatives')
    # Calling derivatives(args, kwargs) (line 406)
    derivatives_call_result_72469 = invoke(stypy.reporting.localization.Localization(__file__, 406, 15), derivatives_72458, *[x_72459], **kwargs_72468)
    
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___72470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 15), derivatives_call_result_72469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_72471 = invoke(stypy.reporting.localization.Localization(__file__, 406, 15), getitem___72470, der_72456)
    
    # Assigning a type to the variable 'stypy_return_type' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'stypy_return_type', subscript_call_result_72471)
    # SSA join for if statement (line 403)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 401)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'krogh_interpolate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'krogh_interpolate' in the type store
    # Getting the type of 'stypy_return_type' (line 358)
    stypy_return_type_72472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'krogh_interpolate'
    return stypy_return_type_72472

# Assigning a type to the variable 'krogh_interpolate' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'krogh_interpolate', krogh_interpolate)

@norecursion
def approximate_taylor_polynomial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 409)
    None_72473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 57), 'None')
    defaults = [None_72473]
    # Create a new context for function 'approximate_taylor_polynomial'
    module_type_store = module_type_store.open_function_context('approximate_taylor_polynomial', 409, 0, False)
    
    # Passed parameters checking function
    approximate_taylor_polynomial.stypy_localization = localization
    approximate_taylor_polynomial.stypy_type_of_self = None
    approximate_taylor_polynomial.stypy_type_store = module_type_store
    approximate_taylor_polynomial.stypy_function_name = 'approximate_taylor_polynomial'
    approximate_taylor_polynomial.stypy_param_names_list = ['f', 'x', 'degree', 'scale', 'order']
    approximate_taylor_polynomial.stypy_varargs_param_name = None
    approximate_taylor_polynomial.stypy_kwargs_param_name = None
    approximate_taylor_polynomial.stypy_call_defaults = defaults
    approximate_taylor_polynomial.stypy_call_varargs = varargs
    approximate_taylor_polynomial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'approximate_taylor_polynomial', ['f', 'x', 'degree', 'scale', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'approximate_taylor_polynomial', localization, ['f', 'x', 'degree', 'scale', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'approximate_taylor_polynomial(...)' code ##################

    str_72474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, (-1)), 'str', '\n    Estimate the Taylor polynomial of f at x by polynomial fitting.\n\n    Parameters\n    ----------\n    f : callable\n        The function whose Taylor polynomial is sought. Should accept\n        a vector of `x` values.\n    x : scalar\n        The point at which the polynomial is to be evaluated.\n    degree : int\n        The degree of the Taylor polynomial\n    scale : scalar\n        The width of the interval to use to evaluate the Taylor polynomial.\n        Function values spread over a range this wide are used to fit the\n        polynomial. Must be chosen carefully.\n    order : int or None, optional\n        The order of the polynomial to be used in the fitting; `f` will be\n        evaluated ``order+1`` times. If None, use `degree`.\n\n    Returns\n    -------\n    p : poly1d instance\n        The Taylor polynomial (translated to the origin, so that\n        for example p(0)=f(x)).\n\n    Notes\n    -----\n    The appropriate choice of "scale" is a trade-off; too large and the\n    function differs from its Taylor polynomial too much to get a good\n    answer, too small and round-off errors overwhelm the higher-order terms.\n    The algorithm used becomes numerically unstable around order 30 even\n    under ideal circumstances.\n\n    Choosing order somewhat larger than degree may improve the higher-order\n    terms.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 448)
    # Getting the type of 'order' (line 448)
    order_72475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 7), 'order')
    # Getting the type of 'None' (line 448)
    None_72476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'None')
    
    (may_be_72477, more_types_in_union_72478) = may_be_none(order_72475, None_72476)

    if may_be_72477:

        if more_types_in_union_72478:
            # Runtime conditional SSA (line 448)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 449):
        
        # Assigning a Name to a Name (line 449):
        # Getting the type of 'degree' (line 449)
        degree_72479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'degree')
        # Assigning a type to the variable 'order' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'order', degree_72479)

        if more_types_in_union_72478:
            # SSA join for if statement (line 448)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 451):
    
    # Assigning a BinOp to a Name (line 451):
    # Getting the type of 'order' (line 451)
    order_72480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'order')
    int_72481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 14), 'int')
    # Applying the binary operator '+' (line 451)
    result_add_72482 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 8), '+', order_72480, int_72481)
    
    # Assigning a type to the variable 'n' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'n', result_add_72482)
    
    # Assigning a BinOp to a Name (line 456):
    
    # Assigning a BinOp to a Name (line 456):
    # Getting the type of 'scale' (line 456)
    scale_72483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 9), 'scale')
    
    # Call to cos(...): (line 456)
    # Processing the call arguments (line 456)
    
    # Call to linspace(...): (line 456)
    # Processing the call arguments (line 456)
    int_72488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 34), 'int')
    # Getting the type of 'np' (line 456)
    np_72489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'np', False)
    # Obtaining the member 'pi' of a type (line 456)
    pi_72490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 36), np_72489, 'pi')
    # Getting the type of 'n' (line 456)
    n_72491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'n', False)
    # Processing the call keyword arguments (line 456)
    # Getting the type of 'n' (line 456)
    n_72492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 53), 'n', False)
    int_72493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'int')
    # Applying the binary operator '%' (line 456)
    result_mod_72494 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 53), '%', n_72492, int_72493)
    
    keyword_72495 = result_mod_72494
    kwargs_72496 = {'endpoint': keyword_72495}
    # Getting the type of 'np' (line 456)
    np_72486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 22), 'np', False)
    # Obtaining the member 'linspace' of a type (line 456)
    linspace_72487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 22), np_72486, 'linspace')
    # Calling linspace(args, kwargs) (line 456)
    linspace_call_result_72497 = invoke(stypy.reporting.localization.Localization(__file__, 456, 22), linspace_72487, *[int_72488, pi_72490, n_72491], **kwargs_72496)
    
    # Processing the call keyword arguments (line 456)
    kwargs_72498 = {}
    # Getting the type of 'np' (line 456)
    np_72484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'np', False)
    # Obtaining the member 'cos' of a type (line 456)
    cos_72485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 15), np_72484, 'cos')
    # Calling cos(args, kwargs) (line 456)
    cos_call_result_72499 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), cos_72485, *[linspace_call_result_72497], **kwargs_72498)
    
    # Applying the binary operator '*' (line 456)
    result_mul_72500 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 9), '*', scale_72483, cos_call_result_72499)
    
    # Getting the type of 'x' (line 456)
    x_72501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 63), 'x')
    # Applying the binary operator '+' (line 456)
    result_add_72502 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 9), '+', result_mul_72500, x_72501)
    
    # Assigning a type to the variable 'xs' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'xs', result_add_72502)
    
    # Assigning a Call to a Name (line 458):
    
    # Assigning a Call to a Name (line 458):
    
    # Call to KroghInterpolator(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'xs' (line 458)
    xs_72504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 26), 'xs', False)
    
    # Call to f(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'xs' (line 458)
    xs_72506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 32), 'xs', False)
    # Processing the call keyword arguments (line 458)
    kwargs_72507 = {}
    # Getting the type of 'f' (line 458)
    f_72505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 30), 'f', False)
    # Calling f(args, kwargs) (line 458)
    f_call_result_72508 = invoke(stypy.reporting.localization.Localization(__file__, 458, 30), f_72505, *[xs_72506], **kwargs_72507)
    
    # Processing the call keyword arguments (line 458)
    kwargs_72509 = {}
    # Getting the type of 'KroghInterpolator' (line 458)
    KroghInterpolator_72503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'KroghInterpolator', False)
    # Calling KroghInterpolator(args, kwargs) (line 458)
    KroghInterpolator_call_result_72510 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), KroghInterpolator_72503, *[xs_72504, f_call_result_72508], **kwargs_72509)
    
    # Assigning a type to the variable 'P' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'P', KroghInterpolator_call_result_72510)
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to derivatives(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'x' (line 459)
    x_72513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 22), 'x', False)
    # Processing the call keyword arguments (line 459)
    # Getting the type of 'degree' (line 459)
    degree_72514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 28), 'degree', False)
    int_72515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 35), 'int')
    # Applying the binary operator '+' (line 459)
    result_add_72516 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 28), '+', degree_72514, int_72515)
    
    keyword_72517 = result_add_72516
    kwargs_72518 = {'der': keyword_72517}
    # Getting the type of 'P' (line 459)
    P_72511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'P', False)
    # Obtaining the member 'derivatives' of a type (line 459)
    derivatives_72512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), P_72511, 'derivatives')
    # Calling derivatives(args, kwargs) (line 459)
    derivatives_call_result_72519 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), derivatives_72512, *[x_72513], **kwargs_72518)
    
    # Assigning a type to the variable 'd' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'd', derivatives_call_result_72519)
    
    # Call to poly1d(...): (line 461)
    # Processing the call arguments (line 461)
    
    # Obtaining the type of the subscript
    int_72522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 58), 'int')
    slice_72523 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 461, 22), None, None, int_72522)
    # Getting the type of 'd' (line 461)
    d_72524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 22), 'd', False)
    
    # Call to factorial(...): (line 461)
    # Processing the call arguments (line 461)
    
    # Call to arange(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'degree' (line 461)
    degree_72528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 44), 'degree', False)
    int_72529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 51), 'int')
    # Applying the binary operator '+' (line 461)
    result_add_72530 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 44), '+', degree_72528, int_72529)
    
    # Processing the call keyword arguments (line 461)
    kwargs_72531 = {}
    # Getting the type of 'np' (line 461)
    np_72526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 34), 'np', False)
    # Obtaining the member 'arange' of a type (line 461)
    arange_72527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 34), np_72526, 'arange')
    # Calling arange(args, kwargs) (line 461)
    arange_call_result_72532 = invoke(stypy.reporting.localization.Localization(__file__, 461, 34), arange_72527, *[result_add_72530], **kwargs_72531)
    
    # Processing the call keyword arguments (line 461)
    kwargs_72533 = {}
    # Getting the type of 'factorial' (line 461)
    factorial_72525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'factorial', False)
    # Calling factorial(args, kwargs) (line 461)
    factorial_call_result_72534 = invoke(stypy.reporting.localization.Localization(__file__, 461, 24), factorial_72525, *[arange_call_result_72532], **kwargs_72533)
    
    # Applying the binary operator 'div' (line 461)
    result_div_72535 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 22), 'div', d_72524, factorial_call_result_72534)
    
    # Obtaining the member '__getitem__' of a type (line 461)
    getitem___72536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 22), result_div_72535, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 461)
    subscript_call_result_72537 = invoke(stypy.reporting.localization.Localization(__file__, 461, 22), getitem___72536, slice_72523)
    
    # Processing the call keyword arguments (line 461)
    kwargs_72538 = {}
    # Getting the type of 'np' (line 461)
    np_72520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 11), 'np', False)
    # Obtaining the member 'poly1d' of a type (line 461)
    poly1d_72521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 11), np_72520, 'poly1d')
    # Calling poly1d(args, kwargs) (line 461)
    poly1d_call_result_72539 = invoke(stypy.reporting.localization.Localization(__file__, 461, 11), poly1d_72521, *[subscript_call_result_72537], **kwargs_72538)
    
    # Assigning a type to the variable 'stypy_return_type' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type', poly1d_call_result_72539)
    
    # ################# End of 'approximate_taylor_polynomial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'approximate_taylor_polynomial' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_72540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72540)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'approximate_taylor_polynomial'
    return stypy_return_type_72540

# Assigning a type to the variable 'approximate_taylor_polynomial' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'approximate_taylor_polynomial', approximate_taylor_polynomial)
# Declaration of the 'BarycentricInterpolator' class
# Getting the type of '_Interpolator1D' (line 464)
_Interpolator1D_72541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 30), '_Interpolator1D')

class BarycentricInterpolator(_Interpolator1D_72541, ):
    str_72542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'str', 'The interpolating polynomial for a set of points\n\n    Constructs a polynomial that passes through a given set of points.\n    Allows evaluation of the polynomial, efficient changing of the y\n    values to be interpolated, and updating by adding more x values.\n    For reasons of numerical stability, this function does not compute\n    the coefficients of the polynomial.\n\n    The values yi need to be provided before the function is\n    evaluated, but none of the preprocessing depends on them, so rapid\n    updates are possible.\n\n    Parameters\n    ----------\n    xi : array_like\n        1-d array of x coordinates of the points the polynomial\n        should pass through\n    yi : array_like, optional\n        The y coordinates of the points the polynomial should pass through.\n        If None, the y values will be supplied later via the `set_y` method.\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values.\n\n    Notes\n    -----\n    This class uses a "barycentric interpolation" method that treats\n    the problem as a special case of rational function interpolation.\n    This algorithm is quite stable, numerically, but even in a world of\n    exact computation, unless the x coordinates are chosen very\n    carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -\n    polynomial interpolation itself is a very ill-conditioned process\n    due to the Runge phenomenon.\n\n    Based on Berrut and Trefethen 2004, "Barycentric Lagrange Interpolation".\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 501)
        None_72543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 30), 'None')
        int_72544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 41), 'int')
        defaults = [None_72543, int_72544]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BarycentricInterpolator.__init__', ['xi', 'yi', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xi', 'yi', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'self' (line 502)
        self_72547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 33), 'self', False)
        # Getting the type of 'xi' (line 502)
        xi_72548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 39), 'xi', False)
        # Getting the type of 'yi' (line 502)
        yi_72549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 43), 'yi', False)
        # Getting the type of 'axis' (line 502)
        axis_72550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 47), 'axis', False)
        # Processing the call keyword arguments (line 502)
        kwargs_72551 = {}
        # Getting the type of '_Interpolator1D' (line 502)
        _Interpolator1D_72545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), '_Interpolator1D', False)
        # Obtaining the member '__init__' of a type (line 502)
        init___72546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), _Interpolator1D_72545, '__init__')
        # Calling __init__(args, kwargs) (line 502)
        init___call_result_72552 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), init___72546, *[self_72547, xi_72548, yi_72549, axis_72550], **kwargs_72551)
        
        
        # Assigning a Call to a Attribute (line 504):
        
        # Assigning a Call to a Attribute (line 504):
        
        # Call to asarray(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'xi' (line 504)
        xi_72555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 29), 'xi', False)
        # Processing the call keyword arguments (line 504)
        kwargs_72556 = {}
        # Getting the type of 'np' (line 504)
        np_72553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 504)
        asarray_72554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 18), np_72553, 'asarray')
        # Calling asarray(args, kwargs) (line 504)
        asarray_call_result_72557 = invoke(stypy.reporting.localization.Localization(__file__, 504, 18), asarray_72554, *[xi_72555], **kwargs_72556)
        
        # Getting the type of 'self' (line 504)
        self_72558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'self')
        # Setting the type of the member 'xi' of a type (line 504)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), self_72558, 'xi', asarray_call_result_72557)
        
        # Call to set_yi(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'yi' (line 505)
        yi_72561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'yi', False)
        # Processing the call keyword arguments (line 505)
        kwargs_72562 = {}
        # Getting the type of 'self' (line 505)
        self_72559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'self', False)
        # Obtaining the member 'set_yi' of a type (line 505)
        set_yi_72560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 8), self_72559, 'set_yi')
        # Calling set_yi(args, kwargs) (line 505)
        set_yi_call_result_72563 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), set_yi_72560, *[yi_72561], **kwargs_72562)
        
        
        # Assigning a Call to a Attribute (line 506):
        
        # Assigning a Call to a Attribute (line 506):
        
        # Call to len(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'self' (line 506)
        self_72565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 21), 'self', False)
        # Obtaining the member 'xi' of a type (line 506)
        xi_72566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 21), self_72565, 'xi')
        # Processing the call keyword arguments (line 506)
        kwargs_72567 = {}
        # Getting the type of 'len' (line 506)
        len_72564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'len', False)
        # Calling len(args, kwargs) (line 506)
        len_call_result_72568 = invoke(stypy.reporting.localization.Localization(__file__, 506, 17), len_72564, *[xi_72566], **kwargs_72567)
        
        # Getting the type of 'self' (line 506)
        self_72569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'self')
        # Setting the type of the member 'n' of a type (line 506)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), self_72569, 'n', len_call_result_72568)
        
        # Assigning a Call to a Attribute (line 508):
        
        # Assigning a Call to a Attribute (line 508):
        
        # Call to zeros(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'self' (line 508)
        self_72572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 508)
        n_72573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 27), self_72572, 'n')
        # Processing the call keyword arguments (line 508)
        kwargs_72574 = {}
        # Getting the type of 'np' (line 508)
        np_72570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 18), 'np', False)
        # Obtaining the member 'zeros' of a type (line 508)
        zeros_72571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 18), np_72570, 'zeros')
        # Calling zeros(args, kwargs) (line 508)
        zeros_call_result_72575 = invoke(stypy.reporting.localization.Localization(__file__, 508, 18), zeros_72571, *[n_72573], **kwargs_72574)
        
        # Getting the type of 'self' (line 508)
        self_72576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'self')
        # Setting the type of the member 'wi' of a type (line 508)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), self_72576, 'wi', zeros_call_result_72575)
        
        # Assigning a Num to a Subscript (line 509):
        
        # Assigning a Num to a Subscript (line 509):
        int_72577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 21), 'int')
        # Getting the type of 'self' (line 509)
        self_72578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'self')
        # Obtaining the member 'wi' of a type (line 509)
        wi_72579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), self_72578, 'wi')
        int_72580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 16), 'int')
        # Storing an element on a container (line 509)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 8), wi_72579, (int_72580, int_72577))
        
        
        # Call to xrange(...): (line 510)
        # Processing the call arguments (line 510)
        int_72582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 24), 'int')
        # Getting the type of 'self' (line 510)
        self_72583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'self', False)
        # Obtaining the member 'n' of a type (line 510)
        n_72584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), self_72583, 'n')
        # Processing the call keyword arguments (line 510)
        kwargs_72585 = {}
        # Getting the type of 'xrange' (line 510)
        xrange_72581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 510)
        xrange_call_result_72586 = invoke(stypy.reporting.localization.Localization(__file__, 510, 17), xrange_72581, *[int_72582, n_72584], **kwargs_72585)
        
        # Testing the type of a for loop iterable (line 510)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 510, 8), xrange_call_result_72586)
        # Getting the type of the for loop variable (line 510)
        for_loop_var_72587 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 510, 8), xrange_call_result_72586)
        # Assigning a type to the variable 'j' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'j', for_loop_var_72587)
        # SSA begins for a for statement (line 510)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 511)
        self_72588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'self')
        # Obtaining the member 'wi' of a type (line 511)
        wi_72589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), self_72588, 'wi')
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 511)
        j_72590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 21), 'j')
        slice_72591 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 511, 12), None, j_72590, None)
        # Getting the type of 'self' (line 511)
        self_72592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'self')
        # Obtaining the member 'wi' of a type (line 511)
        wi_72593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), self_72592, 'wi')
        # Obtaining the member '__getitem__' of a type (line 511)
        getitem___72594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), wi_72593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 511)
        subscript_call_result_72595 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), getitem___72594, slice_72591)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 511)
        j_72596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 36), 'j')
        # Getting the type of 'self' (line 511)
        self_72597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 28), 'self')
        # Obtaining the member 'xi' of a type (line 511)
        xi_72598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 28), self_72597, 'xi')
        # Obtaining the member '__getitem__' of a type (line 511)
        getitem___72599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 28), xi_72598, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 511)
        subscript_call_result_72600 = invoke(stypy.reporting.localization.Localization(__file__, 511, 28), getitem___72599, j_72596)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 511)
        j_72601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 48), 'j')
        slice_72602 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 511, 39), None, j_72601, None)
        # Getting the type of 'self' (line 511)
        self_72603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 39), 'self')
        # Obtaining the member 'xi' of a type (line 511)
        xi_72604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 39), self_72603, 'xi')
        # Obtaining the member '__getitem__' of a type (line 511)
        getitem___72605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 39), xi_72604, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 511)
        subscript_call_result_72606 = invoke(stypy.reporting.localization.Localization(__file__, 511, 39), getitem___72605, slice_72602)
        
        # Applying the binary operator '-' (line 511)
        result_sub_72607 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 28), '-', subscript_call_result_72600, subscript_call_result_72606)
        
        # Applying the binary operator '*=' (line 511)
        result_imul_72608 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 12), '*=', subscript_call_result_72595, result_sub_72607)
        # Getting the type of 'self' (line 511)
        self_72609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'self')
        # Obtaining the member 'wi' of a type (line 511)
        wi_72610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), self_72609, 'wi')
        # Getting the type of 'j' (line 511)
        j_72611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 21), 'j')
        slice_72612 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 511, 12), None, j_72611, None)
        # Storing an element on a container (line 511)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 12), wi_72610, (slice_72612, result_imul_72608))
        
        
        # Assigning a Call to a Subscript (line 512):
        
        # Assigning a Call to a Subscript (line 512):
        
        # Call to reduce(...): (line 512)
        # Processing the call arguments (line 512)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 512)
        j_72616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 53), 'j', False)
        slice_72617 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 512, 44), None, j_72616, None)
        # Getting the type of 'self' (line 512)
        self_72618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 44), 'self', False)
        # Obtaining the member 'xi' of a type (line 512)
        xi_72619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 44), self_72618, 'xi')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___72620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 44), xi_72619, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_72621 = invoke(stypy.reporting.localization.Localization(__file__, 512, 44), getitem___72620, slice_72617)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 512)
        j_72622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 64), 'j', False)
        # Getting the type of 'self' (line 512)
        self_72623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 56), 'self', False)
        # Obtaining the member 'xi' of a type (line 512)
        xi_72624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 56), self_72623, 'xi')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___72625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 56), xi_72624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_72626 = invoke(stypy.reporting.localization.Localization(__file__, 512, 56), getitem___72625, j_72622)
        
        # Applying the binary operator '-' (line 512)
        result_sub_72627 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 44), '-', subscript_call_result_72621, subscript_call_result_72626)
        
        # Processing the call keyword arguments (line 512)
        kwargs_72628 = {}
        # Getting the type of 'np' (line 512)
        np_72613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 25), 'np', False)
        # Obtaining the member 'multiply' of a type (line 512)
        multiply_72614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 25), np_72613, 'multiply')
        # Obtaining the member 'reduce' of a type (line 512)
        reduce_72615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 25), multiply_72614, 'reduce')
        # Calling reduce(args, kwargs) (line 512)
        reduce_call_result_72629 = invoke(stypy.reporting.localization.Localization(__file__, 512, 25), reduce_72615, *[result_sub_72627], **kwargs_72628)
        
        # Getting the type of 'self' (line 512)
        self_72630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'self')
        # Obtaining the member 'wi' of a type (line 512)
        wi_72631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), self_72630, 'wi')
        # Getting the type of 'j' (line 512)
        j_72632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 20), 'j')
        # Storing an element on a container (line 512)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), wi_72631, (j_72632, reduce_call_result_72629))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 513)
        self_72633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self')
        # Obtaining the member 'wi' of a type (line 513)
        wi_72634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_72633, 'wi')
        int_72635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 20), 'int')
        # Applying the binary operator '**=' (line 513)
        result_ipow_72636 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 8), '**=', wi_72634, int_72635)
        # Getting the type of 'self' (line 513)
        self_72637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self')
        # Setting the type of the member 'wi' of a type (line 513)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_72637, 'wi', result_ipow_72636)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_yi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 515)
        None_72638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 30), 'None')
        defaults = [None_72638]
        # Create a new context for function 'set_yi'
        module_type_store = module_type_store.open_function_context('set_yi', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_localization', localization)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_type_store', module_type_store)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_function_name', 'BarycentricInterpolator.set_yi')
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_param_names_list', ['yi', 'axis'])
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_varargs_param_name', None)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_call_defaults', defaults)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_call_varargs', varargs)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BarycentricInterpolator.set_yi.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BarycentricInterpolator.set_yi', ['yi', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_yi', localization, ['yi', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_yi(...)' code ##################

        str_72639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, (-1)), 'str', '\n        Update the y values to be interpolated\n\n        The barycentric interpolation algorithm requires the calculation\n        of weights, but these depend only on the xi. The yi can be changed\n        at any time.\n\n        Parameters\n        ----------\n        yi : array_like\n            The y coordinates of the points the polynomial should pass through.\n            If None, the y values will be supplied later.\n        axis : int, optional\n            Axis in the yi array corresponding to the x-coordinate values.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 532)
        # Getting the type of 'yi' (line 532)
        yi_72640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'yi')
        # Getting the type of 'None' (line 532)
        None_72641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 'None')
        
        (may_be_72642, more_types_in_union_72643) = may_be_none(yi_72640, None_72641)

        if may_be_72642:

            if more_types_in_union_72643:
                # Runtime conditional SSA (line 532)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 533):
            
            # Assigning a Name to a Attribute (line 533):
            # Getting the type of 'None' (line 533)
            None_72644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'None')
            # Getting the type of 'self' (line 533)
            self_72645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'self')
            # Setting the type of the member 'yi' of a type (line 533)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 12), self_72645, 'yi', None_72644)
            # Assigning a type to the variable 'stypy_return_type' (line 534)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_72643:
                # SSA join for if statement (line 532)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _set_yi(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'yi' (line 535)
        yi_72648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 21), 'yi', False)
        # Processing the call keyword arguments (line 535)
        # Getting the type of 'self' (line 535)
        self_72649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 28), 'self', False)
        # Obtaining the member 'xi' of a type (line 535)
        xi_72650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 28), self_72649, 'xi')
        keyword_72651 = xi_72650
        # Getting the type of 'axis' (line 535)
        axis_72652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 42), 'axis', False)
        keyword_72653 = axis_72652
        kwargs_72654 = {'xi': keyword_72651, 'axis': keyword_72653}
        # Getting the type of 'self' (line 535)
        self_72646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'self', False)
        # Obtaining the member '_set_yi' of a type (line 535)
        _set_yi_72647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), self_72646, '_set_yi')
        # Calling _set_yi(args, kwargs) (line 535)
        _set_yi_call_result_72655 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), _set_yi_72647, *[yi_72648], **kwargs_72654)
        
        
        # Assigning a Call to a Attribute (line 536):
        
        # Assigning a Call to a Attribute (line 536):
        
        # Call to _reshape_yi(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'yi' (line 536)
        yi_72658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 35), 'yi', False)
        # Processing the call keyword arguments (line 536)
        kwargs_72659 = {}
        # Getting the type of 'self' (line 536)
        self_72656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 18), 'self', False)
        # Obtaining the member '_reshape_yi' of a type (line 536)
        _reshape_yi_72657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 18), self_72656, '_reshape_yi')
        # Calling _reshape_yi(args, kwargs) (line 536)
        _reshape_yi_call_result_72660 = invoke(stypy.reporting.localization.Localization(__file__, 536, 18), _reshape_yi_72657, *[yi_72658], **kwargs_72659)
        
        # Getting the type of 'self' (line 536)
        self_72661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'self')
        # Setting the type of the member 'yi' of a type (line 536)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), self_72661, 'yi', _reshape_yi_call_result_72660)
        
        # Assigning a Attribute to a Tuple (line 537):
        
        # Assigning a Subscript to a Name (line 537):
        
        # Obtaining the type of the subscript
        int_72662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 8), 'int')
        # Getting the type of 'self' (line 537)
        self_72663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 25), 'self')
        # Obtaining the member 'yi' of a type (line 537)
        yi_72664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 25), self_72663, 'yi')
        # Obtaining the member 'shape' of a type (line 537)
        shape_72665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 25), yi_72664, 'shape')
        # Obtaining the member '__getitem__' of a type (line 537)
        getitem___72666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), shape_72665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 537)
        subscript_call_result_72667 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), getitem___72666, int_72662)
        
        # Assigning a type to the variable 'tuple_var_assignment_71417' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'tuple_var_assignment_71417', subscript_call_result_72667)
        
        # Assigning a Subscript to a Name (line 537):
        
        # Obtaining the type of the subscript
        int_72668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 8), 'int')
        # Getting the type of 'self' (line 537)
        self_72669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 25), 'self')
        # Obtaining the member 'yi' of a type (line 537)
        yi_72670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 25), self_72669, 'yi')
        # Obtaining the member 'shape' of a type (line 537)
        shape_72671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 25), yi_72670, 'shape')
        # Obtaining the member '__getitem__' of a type (line 537)
        getitem___72672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), shape_72671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 537)
        subscript_call_result_72673 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), getitem___72672, int_72668)
        
        # Assigning a type to the variable 'tuple_var_assignment_71418' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'tuple_var_assignment_71418', subscript_call_result_72673)
        
        # Assigning a Name to a Attribute (line 537):
        # Getting the type of 'tuple_var_assignment_71417' (line 537)
        tuple_var_assignment_71417_72674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'tuple_var_assignment_71417')
        # Getting the type of 'self' (line 537)
        self_72675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'self')
        # Setting the type of the member 'n' of a type (line 537)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), self_72675, 'n', tuple_var_assignment_71417_72674)
        
        # Assigning a Name to a Attribute (line 537):
        # Getting the type of 'tuple_var_assignment_71418' (line 537)
        tuple_var_assignment_71418_72676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'tuple_var_assignment_71418')
        # Getting the type of 'self' (line 537)
        self_72677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'self')
        # Setting the type of the member 'r' of a type (line 537)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 16), self_72677, 'r', tuple_var_assignment_71418_72676)
        
        # ################# End of 'set_yi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_yi' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_72678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_yi'
        return stypy_return_type_72678


    @norecursion
    def add_xi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 539)
        None_72679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 28), 'None')
        defaults = [None_72679]
        # Create a new context for function 'add_xi'
        module_type_store = module_type_store.open_function_context('add_xi', 539, 4, False)
        # Assigning a type to the variable 'self' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_localization', localization)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_type_store', module_type_store)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_function_name', 'BarycentricInterpolator.add_xi')
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_param_names_list', ['xi', 'yi'])
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_varargs_param_name', None)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_call_defaults', defaults)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_call_varargs', varargs)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BarycentricInterpolator.add_xi.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BarycentricInterpolator.add_xi', ['xi', 'yi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_xi', localization, ['xi', 'yi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_xi(...)' code ##################

        str_72680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, (-1)), 'str', '\n        Add more x values to the set to be interpolated\n\n        The barycentric interpolation algorithm allows easy updating by\n        adding more points for the polynomial to pass through.\n\n        Parameters\n        ----------\n        xi : array_like\n            The x coordinates of the points that the polynomial should pass\n            through.\n        yi : array_like, optional\n            The y coordinates of the points the polynomial should pass through.\n            Should have shape ``(xi.size, R)``; if R > 1 then the polynomial is\n            vector-valued.\n            If `yi` is not given, the y values will be supplied later. `yi` should\n            be given if and only if the interpolator has y values specified.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 559)
        # Getting the type of 'yi' (line 559)
        yi_72681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'yi')
        # Getting the type of 'None' (line 559)
        None_72682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 21), 'None')
        
        (may_be_72683, more_types_in_union_72684) = may_not_be_none(yi_72681, None_72682)

        if may_be_72683:

            if more_types_in_union_72684:
                # Runtime conditional SSA (line 559)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 560)
            # Getting the type of 'self' (line 560)
            self_72685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'self')
            # Obtaining the member 'yi' of a type (line 560)
            yi_72686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 15), self_72685, 'yi')
            # Getting the type of 'None' (line 560)
            None_72687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 26), 'None')
            
            (may_be_72688, more_types_in_union_72689) = may_be_none(yi_72686, None_72687)

            if may_be_72688:

                if more_types_in_union_72689:
                    # Runtime conditional SSA (line 560)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 561)
                # Processing the call arguments (line 561)
                str_72691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 33), 'str', 'No previous yi value to update!')
                # Processing the call keyword arguments (line 561)
                kwargs_72692 = {}
                # Getting the type of 'ValueError' (line 561)
                ValueError_72690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 561)
                ValueError_call_result_72693 = invoke(stypy.reporting.localization.Localization(__file__, 561, 22), ValueError_72690, *[str_72691], **kwargs_72692)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 561, 16), ValueError_call_result_72693, 'raise parameter', BaseException)

                if more_types_in_union_72689:
                    # SSA join for if statement (line 560)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 562):
            
            # Assigning a Call to a Name (line 562):
            
            # Call to _reshape_yi(...): (line 562)
            # Processing the call arguments (line 562)
            # Getting the type of 'yi' (line 562)
            yi_72696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 34), 'yi', False)
            # Processing the call keyword arguments (line 562)
            # Getting the type of 'True' (line 562)
            True_72697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 44), 'True', False)
            keyword_72698 = True_72697
            kwargs_72699 = {'check': keyword_72698}
            # Getting the type of 'self' (line 562)
            self_72694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 17), 'self', False)
            # Obtaining the member '_reshape_yi' of a type (line 562)
            _reshape_yi_72695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 17), self_72694, '_reshape_yi')
            # Calling _reshape_yi(args, kwargs) (line 562)
            _reshape_yi_call_result_72700 = invoke(stypy.reporting.localization.Localization(__file__, 562, 17), _reshape_yi_72695, *[yi_72696], **kwargs_72699)
            
            # Assigning a type to the variable 'yi' (line 562)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'yi', _reshape_yi_call_result_72700)
            
            # Assigning a Call to a Attribute (line 563):
            
            # Assigning a Call to a Attribute (line 563):
            
            # Call to vstack(...): (line 563)
            # Processing the call arguments (line 563)
            
            # Obtaining an instance of the builtin type 'tuple' (line 563)
            tuple_72703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 563)
            # Adding element type (line 563)
            # Getting the type of 'self' (line 563)
            self_72704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 33), 'self', False)
            # Obtaining the member 'yi' of a type (line 563)
            yi_72705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 33), self_72704, 'yi')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 33), tuple_72703, yi_72705)
            # Adding element type (line 563)
            # Getting the type of 'yi' (line 563)
            yi_72706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 41), 'yi', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 33), tuple_72703, yi_72706)
            
            # Processing the call keyword arguments (line 563)
            kwargs_72707 = {}
            # Getting the type of 'np' (line 563)
            np_72701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 22), 'np', False)
            # Obtaining the member 'vstack' of a type (line 563)
            vstack_72702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 22), np_72701, 'vstack')
            # Calling vstack(args, kwargs) (line 563)
            vstack_call_result_72708 = invoke(stypy.reporting.localization.Localization(__file__, 563, 22), vstack_72702, *[tuple_72703], **kwargs_72707)
            
            # Getting the type of 'self' (line 563)
            self_72709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'self')
            # Setting the type of the member 'yi' of a type (line 563)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 12), self_72709, 'yi', vstack_call_result_72708)

            if more_types_in_union_72684:
                # Runtime conditional SSA for else branch (line 559)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_72683) or more_types_in_union_72684):
            
            
            # Getting the type of 'self' (line 565)
            self_72710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 15), 'self')
            # Obtaining the member 'yi' of a type (line 565)
            yi_72711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 15), self_72710, 'yi')
            # Getting the type of 'None' (line 565)
            None_72712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 30), 'None')
            # Applying the binary operator 'isnot' (line 565)
            result_is_not_72713 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 15), 'isnot', yi_72711, None_72712)
            
            # Testing the type of an if condition (line 565)
            if_condition_72714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 12), result_is_not_72713)
            # Assigning a type to the variable 'if_condition_72714' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'if_condition_72714', if_condition_72714)
            # SSA begins for if statement (line 565)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 566)
            # Processing the call arguments (line 566)
            str_72716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 33), 'str', 'No update to yi provided!')
            # Processing the call keyword arguments (line 566)
            kwargs_72717 = {}
            # Getting the type of 'ValueError' (line 566)
            ValueError_72715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 566)
            ValueError_call_result_72718 = invoke(stypy.reporting.localization.Localization(__file__, 566, 22), ValueError_72715, *[str_72716], **kwargs_72717)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 566, 16), ValueError_call_result_72718, 'raise parameter', BaseException)
            # SSA join for if statement (line 565)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_72683 and more_types_in_union_72684):
                # SSA join for if statement (line 559)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 567):
        
        # Assigning a Attribute to a Name (line 567):
        # Getting the type of 'self' (line 567)
        self_72719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 16), 'self')
        # Obtaining the member 'n' of a type (line 567)
        n_72720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 16), self_72719, 'n')
        # Assigning a type to the variable 'old_n' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'old_n', n_72720)
        
        # Assigning a Call to a Attribute (line 568):
        
        # Assigning a Call to a Attribute (line 568):
        
        # Call to concatenate(...): (line 568)
        # Processing the call arguments (line 568)
        
        # Obtaining an instance of the builtin type 'tuple' (line 568)
        tuple_72723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 568)
        # Adding element type (line 568)
        # Getting the type of 'self' (line 568)
        self_72724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 34), 'self', False)
        # Obtaining the member 'xi' of a type (line 568)
        xi_72725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 34), self_72724, 'xi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 34), tuple_72723, xi_72725)
        # Adding element type (line 568)
        # Getting the type of 'xi' (line 568)
        xi_72726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 42), 'xi', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 34), tuple_72723, xi_72726)
        
        # Processing the call keyword arguments (line 568)
        kwargs_72727 = {}
        # Getting the type of 'np' (line 568)
        np_72721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 18), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 568)
        concatenate_72722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 18), np_72721, 'concatenate')
        # Calling concatenate(args, kwargs) (line 568)
        concatenate_call_result_72728 = invoke(stypy.reporting.localization.Localization(__file__, 568, 18), concatenate_72722, *[tuple_72723], **kwargs_72727)
        
        # Getting the type of 'self' (line 568)
        self_72729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'self')
        # Setting the type of the member 'xi' of a type (line 568)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 8), self_72729, 'xi', concatenate_call_result_72728)
        
        # Assigning a Call to a Attribute (line 569):
        
        # Assigning a Call to a Attribute (line 569):
        
        # Call to len(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'self' (line 569)
        self_72731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'self', False)
        # Obtaining the member 'xi' of a type (line 569)
        xi_72732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), self_72731, 'xi')
        # Processing the call keyword arguments (line 569)
        kwargs_72733 = {}
        # Getting the type of 'len' (line 569)
        len_72730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 17), 'len', False)
        # Calling len(args, kwargs) (line 569)
        len_call_result_72734 = invoke(stypy.reporting.localization.Localization(__file__, 569, 17), len_72730, *[xi_72732], **kwargs_72733)
        
        # Getting the type of 'self' (line 569)
        self_72735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'self')
        # Setting the type of the member 'n' of a type (line 569)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), self_72735, 'n', len_call_result_72734)
        
        # Getting the type of 'self' (line 570)
        self_72736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self')
        # Obtaining the member 'wi' of a type (line 570)
        wi_72737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), self_72736, 'wi')
        int_72738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 20), 'int')
        # Applying the binary operator '**=' (line 570)
        result_ipow_72739 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 8), '**=', wi_72737, int_72738)
        # Getting the type of 'self' (line 570)
        self_72740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self')
        # Setting the type of the member 'wi' of a type (line 570)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), self_72740, 'wi', result_ipow_72739)
        
        
        # Assigning a Attribute to a Name (line 571):
        
        # Assigning a Attribute to a Name (line 571):
        # Getting the type of 'self' (line 571)
        self_72741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 17), 'self')
        # Obtaining the member 'wi' of a type (line 571)
        wi_72742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 17), self_72741, 'wi')
        # Assigning a type to the variable 'old_wi' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'old_wi', wi_72742)
        
        # Assigning a Call to a Attribute (line 572):
        
        # Assigning a Call to a Attribute (line 572):
        
        # Call to zeros(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'self' (line 572)
        self_72745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 572)
        n_72746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 27), self_72745, 'n')
        # Processing the call keyword arguments (line 572)
        kwargs_72747 = {}
        # Getting the type of 'np' (line 572)
        np_72743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 18), 'np', False)
        # Obtaining the member 'zeros' of a type (line 572)
        zeros_72744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 18), np_72743, 'zeros')
        # Calling zeros(args, kwargs) (line 572)
        zeros_call_result_72748 = invoke(stypy.reporting.localization.Localization(__file__, 572, 18), zeros_72744, *[n_72746], **kwargs_72747)
        
        # Getting the type of 'self' (line 572)
        self_72749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'self')
        # Setting the type of the member 'wi' of a type (line 572)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), self_72749, 'wi', zeros_call_result_72748)
        
        # Assigning a Name to a Subscript (line 573):
        
        # Assigning a Name to a Subscript (line 573):
        # Getting the type of 'old_wi' (line 573)
        old_wi_72750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 26), 'old_wi')
        # Getting the type of 'self' (line 573)
        self_72751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'self')
        # Obtaining the member 'wi' of a type (line 573)
        wi_72752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), self_72751, 'wi')
        # Getting the type of 'old_n' (line 573)
        old_n_72753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 17), 'old_n')
        slice_72754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 573, 8), None, old_n_72753, None)
        # Storing an element on a container (line 573)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 8), wi_72752, (slice_72754, old_wi_72750))
        
        
        # Call to xrange(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'old_n' (line 574)
        old_n_72756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'old_n', False)
        # Getting the type of 'self' (line 574)
        self_72757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 30), 'self', False)
        # Obtaining the member 'n' of a type (line 574)
        n_72758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 30), self_72757, 'n')
        # Processing the call keyword arguments (line 574)
        kwargs_72759 = {}
        # Getting the type of 'xrange' (line 574)
        xrange_72755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 574)
        xrange_call_result_72760 = invoke(stypy.reporting.localization.Localization(__file__, 574, 17), xrange_72755, *[old_n_72756, n_72758], **kwargs_72759)
        
        # Testing the type of a for loop iterable (line 574)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 574, 8), xrange_call_result_72760)
        # Getting the type of the for loop variable (line 574)
        for_loop_var_72761 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 574, 8), xrange_call_result_72760)
        # Assigning a type to the variable 'j' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'j', for_loop_var_72761)
        # SSA begins for a for statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 575)
        self_72762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self')
        # Obtaining the member 'wi' of a type (line 575)
        wi_72763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_72762, 'wi')
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 575)
        j_72764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'j')
        slice_72765 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 12), None, j_72764, None)
        # Getting the type of 'self' (line 575)
        self_72766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self')
        # Obtaining the member 'wi' of a type (line 575)
        wi_72767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_72766, 'wi')
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___72768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), wi_72767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_72769 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), getitem___72768, slice_72765)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 575)
        j_72770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 36), 'j')
        # Getting the type of 'self' (line 575)
        self_72771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 28), 'self')
        # Obtaining the member 'xi' of a type (line 575)
        xi_72772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 28), self_72771, 'xi')
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___72773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 28), xi_72772, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_72774 = invoke(stypy.reporting.localization.Localization(__file__, 575, 28), getitem___72773, j_72770)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 575)
        j_72775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 48), 'j')
        slice_72776 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 39), None, j_72775, None)
        # Getting the type of 'self' (line 575)
        self_72777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 39), 'self')
        # Obtaining the member 'xi' of a type (line 575)
        xi_72778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 39), self_72777, 'xi')
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___72779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 39), xi_72778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_72780 = invoke(stypy.reporting.localization.Localization(__file__, 575, 39), getitem___72779, slice_72776)
        
        # Applying the binary operator '-' (line 575)
        result_sub_72781 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 28), '-', subscript_call_result_72774, subscript_call_result_72780)
        
        # Applying the binary operator '*=' (line 575)
        result_imul_72782 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 12), '*=', subscript_call_result_72769, result_sub_72781)
        # Getting the type of 'self' (line 575)
        self_72783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self')
        # Obtaining the member 'wi' of a type (line 575)
        wi_72784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_72783, 'wi')
        # Getting the type of 'j' (line 575)
        j_72785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'j')
        slice_72786 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 12), None, j_72785, None)
        # Storing an element on a container (line 575)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 12), wi_72784, (slice_72786, result_imul_72782))
        
        
        # Assigning a Call to a Subscript (line 576):
        
        # Assigning a Call to a Subscript (line 576):
        
        # Call to reduce(...): (line 576)
        # Processing the call arguments (line 576)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 576)
        j_72790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 53), 'j', False)
        slice_72791 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 576, 44), None, j_72790, None)
        # Getting the type of 'self' (line 576)
        self_72792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 44), 'self', False)
        # Obtaining the member 'xi' of a type (line 576)
        xi_72793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 44), self_72792, 'xi')
        # Obtaining the member '__getitem__' of a type (line 576)
        getitem___72794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 44), xi_72793, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 576)
        subscript_call_result_72795 = invoke(stypy.reporting.localization.Localization(__file__, 576, 44), getitem___72794, slice_72791)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 576)
        j_72796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 64), 'j', False)
        # Getting the type of 'self' (line 576)
        self_72797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 56), 'self', False)
        # Obtaining the member 'xi' of a type (line 576)
        xi_72798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 56), self_72797, 'xi')
        # Obtaining the member '__getitem__' of a type (line 576)
        getitem___72799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 56), xi_72798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 576)
        subscript_call_result_72800 = invoke(stypy.reporting.localization.Localization(__file__, 576, 56), getitem___72799, j_72796)
        
        # Applying the binary operator '-' (line 576)
        result_sub_72801 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 44), '-', subscript_call_result_72795, subscript_call_result_72800)
        
        # Processing the call keyword arguments (line 576)
        kwargs_72802 = {}
        # Getting the type of 'np' (line 576)
        np_72787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 25), 'np', False)
        # Obtaining the member 'multiply' of a type (line 576)
        multiply_72788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 25), np_72787, 'multiply')
        # Obtaining the member 'reduce' of a type (line 576)
        reduce_72789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 25), multiply_72788, 'reduce')
        # Calling reduce(args, kwargs) (line 576)
        reduce_call_result_72803 = invoke(stypy.reporting.localization.Localization(__file__, 576, 25), reduce_72789, *[result_sub_72801], **kwargs_72802)
        
        # Getting the type of 'self' (line 576)
        self_72804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'self')
        # Obtaining the member 'wi' of a type (line 576)
        wi_72805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 12), self_72804, 'wi')
        # Getting the type of 'j' (line 576)
        j_72806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 20), 'j')
        # Storing an element on a container (line 576)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 12), wi_72805, (j_72806, reduce_call_result_72803))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 577)
        self_72807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'self')
        # Obtaining the member 'wi' of a type (line 577)
        wi_72808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), self_72807, 'wi')
        int_72809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 20), 'int')
        # Applying the binary operator '**=' (line 577)
        result_ipow_72810 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 8), '**=', wi_72808, int_72809)
        # Getting the type of 'self' (line 577)
        self_72811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'self')
        # Setting the type of the member 'wi' of a type (line 577)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), self_72811, 'wi', result_ipow_72810)
        
        
        # ################# End of 'add_xi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_xi' in the type store
        # Getting the type of 'stypy_return_type' (line 539)
        stypy_return_type_72812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_xi'
        return stypy_return_type_72812


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 579, 4, False)
        # Assigning a type to the variable 'self' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_localization', localization)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_function_name', 'BarycentricInterpolator.__call__')
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BarycentricInterpolator.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BarycentricInterpolator.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_72813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, (-1)), 'str', 'Evaluate the interpolating polynomial at the points x\n\n        Parameters\n        ----------\n        x : array_like\n            Points to evaluate the interpolant at.\n\n        Returns\n        -------\n        y : array_like\n            Interpolated values. Shape is determined by replacing\n            the interpolation axis in the original array with the shape of x.\n\n        Notes\n        -----\n        Currently the code computes an outer product between x and the\n        weights, that is, it constructs an intermediate array of size\n        N by len(x), where N is the degree of the polynomial.\n        ')
        
        # Call to __call__(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 'self' (line 599)
        self_72816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 40), 'self', False)
        # Getting the type of 'x' (line 599)
        x_72817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 46), 'x', False)
        # Processing the call keyword arguments (line 599)
        kwargs_72818 = {}
        # Getting the type of '_Interpolator1D' (line 599)
        _Interpolator1D_72814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), '_Interpolator1D', False)
        # Obtaining the member '__call__' of a type (line 599)
        call___72815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 15), _Interpolator1D_72814, '__call__')
        # Calling __call__(args, kwargs) (line 599)
        call___call_result_72819 = invoke(stypy.reporting.localization.Localization(__file__, 599, 15), call___72815, *[self_72816, x_72817], **kwargs_72818)
        
        # Assigning a type to the variable 'stypy_return_type' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'stypy_return_type', call___call_result_72819)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 579)
        stypy_return_type_72820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72820)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_72820


    @norecursion
    def _evaluate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_evaluate'
        module_type_store = module_type_store.open_function_context('_evaluate', 601, 4, False)
        # Assigning a type to the variable 'self' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_localization', localization)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_type_store', module_type_store)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_function_name', 'BarycentricInterpolator._evaluate')
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_param_names_list', ['x'])
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_varargs_param_name', None)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_call_defaults', defaults)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_call_varargs', varargs)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BarycentricInterpolator._evaluate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BarycentricInterpolator._evaluate', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_evaluate', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_evaluate(...)' code ##################

        
        
        # Getting the type of 'x' (line 602)
        x_72821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 11), 'x')
        # Obtaining the member 'size' of a type (line 602)
        size_72822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 11), x_72821, 'size')
        int_72823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 21), 'int')
        # Applying the binary operator '==' (line 602)
        result_eq_72824 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 11), '==', size_72822, int_72823)
        
        # Testing the type of an if condition (line 602)
        if_condition_72825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 602, 8), result_eq_72824)
        # Assigning a type to the variable 'if_condition_72825' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'if_condition_72825', if_condition_72825)
        # SSA begins for if statement (line 602)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 603):
        
        # Assigning a Call to a Name (line 603):
        
        # Call to zeros(...): (line 603)
        # Processing the call arguments (line 603)
        
        # Obtaining an instance of the builtin type 'tuple' (line 603)
        tuple_72828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 603)
        # Adding element type (line 603)
        int_72829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 26), tuple_72828, int_72829)
        # Adding element type (line 603)
        # Getting the type of 'self' (line 603)
        self_72830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 29), 'self', False)
        # Obtaining the member 'r' of a type (line 603)
        r_72831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 29), self_72830, 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 26), tuple_72828, r_72831)
        
        # Processing the call keyword arguments (line 603)
        # Getting the type of 'self' (line 603)
        self_72832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 44), 'self', False)
        # Obtaining the member 'dtype' of a type (line 603)
        dtype_72833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 44), self_72832, 'dtype')
        keyword_72834 = dtype_72833
        kwargs_72835 = {'dtype': keyword_72834}
        # Getting the type of 'np' (line 603)
        np_72826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 16), 'np', False)
        # Obtaining the member 'zeros' of a type (line 603)
        zeros_72827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 16), np_72826, 'zeros')
        # Calling zeros(args, kwargs) (line 603)
        zeros_call_result_72836 = invoke(stypy.reporting.localization.Localization(__file__, 603, 16), zeros_72827, *[tuple_72828], **kwargs_72835)
        
        # Assigning a type to the variable 'p' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'p', zeros_call_result_72836)
        # SSA branch for the else part of an if statement (line 602)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 605):
        
        # Assigning a BinOp to a Name (line 605):
        
        # Obtaining the type of the subscript
        Ellipsis_72837 = Ellipsis
        # Getting the type of 'np' (line 605)
        np_72838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 22), 'np')
        # Obtaining the member 'newaxis' of a type (line 605)
        newaxis_72839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 22), np_72838, 'newaxis')
        # Getting the type of 'x' (line 605)
        x_72840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 16), 'x')
        # Obtaining the member '__getitem__' of a type (line 605)
        getitem___72841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 16), x_72840, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 605)
        subscript_call_result_72842 = invoke(stypy.reporting.localization.Localization(__file__, 605, 16), getitem___72841, (Ellipsis_72837, newaxis_72839))
        
        # Getting the type of 'self' (line 605)
        self_72843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 34), 'self')
        # Obtaining the member 'xi' of a type (line 605)
        xi_72844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 34), self_72843, 'xi')
        # Applying the binary operator '-' (line 605)
        result_sub_72845 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 16), '-', subscript_call_result_72842, xi_72844)
        
        # Assigning a type to the variable 'c' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'c', result_sub_72845)
        
        # Assigning a Compare to a Name (line 606):
        
        # Assigning a Compare to a Name (line 606):
        
        # Getting the type of 'c' (line 606)
        c_72846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'c')
        int_72847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 21), 'int')
        # Applying the binary operator '==' (line 606)
        result_eq_72848 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 16), '==', c_72846, int_72847)
        
        # Assigning a type to the variable 'z' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'z', result_eq_72848)
        
        # Assigning a Num to a Subscript (line 607):
        
        # Assigning a Num to a Subscript (line 607):
        int_72849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 19), 'int')
        # Getting the type of 'c' (line 607)
        c_72850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'c')
        # Getting the type of 'z' (line 607)
        z_72851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 14), 'z')
        # Storing an element on a container (line 607)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 12), c_72850, (z_72851, int_72849))
        
        # Assigning a BinOp to a Name (line 608):
        
        # Assigning a BinOp to a Name (line 608):
        # Getting the type of 'self' (line 608)
        self_72852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 16), 'self')
        # Obtaining the member 'wi' of a type (line 608)
        wi_72853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 16), self_72852, 'wi')
        # Getting the type of 'c' (line 608)
        c_72854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 24), 'c')
        # Applying the binary operator 'div' (line 608)
        result_div_72855 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 16), 'div', wi_72853, c_72854)
        
        # Assigning a type to the variable 'c' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'c', result_div_72855)
        
        # Assigning a BinOp to a Name (line 609):
        
        # Assigning a BinOp to a Name (line 609):
        
        # Call to dot(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'c' (line 609)
        c_72858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 23), 'c', False)
        # Getting the type of 'self' (line 609)
        self_72859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 25), 'self', False)
        # Obtaining the member 'yi' of a type (line 609)
        yi_72860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 25), self_72859, 'yi')
        # Processing the call keyword arguments (line 609)
        kwargs_72861 = {}
        # Getting the type of 'np' (line 609)
        np_72856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'np', False)
        # Obtaining the member 'dot' of a type (line 609)
        dot_72857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), np_72856, 'dot')
        # Calling dot(args, kwargs) (line 609)
        dot_call_result_72862 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), dot_72857, *[c_72858, yi_72860], **kwargs_72861)
        
        
        # Obtaining the type of the subscript
        Ellipsis_72863 = Ellipsis
        # Getting the type of 'np' (line 609)
        np_72864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 56), 'np')
        # Obtaining the member 'newaxis' of a type (line 609)
        newaxis_72865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 56), np_72864, 'newaxis')
        
        # Call to sum(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'c' (line 609)
        c_72868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 41), 'c', False)
        # Processing the call keyword arguments (line 609)
        int_72869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 48), 'int')
        keyword_72870 = int_72869
        kwargs_72871 = {'axis': keyword_72870}
        # Getting the type of 'np' (line 609)
        np_72866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 34), 'np', False)
        # Obtaining the member 'sum' of a type (line 609)
        sum_72867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 34), np_72866, 'sum')
        # Calling sum(args, kwargs) (line 609)
        sum_call_result_72872 = invoke(stypy.reporting.localization.Localization(__file__, 609, 34), sum_72867, *[c_72868], **kwargs_72871)
        
        # Obtaining the member '__getitem__' of a type (line 609)
        getitem___72873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 34), sum_call_result_72872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 609)
        subscript_call_result_72874 = invoke(stypy.reporting.localization.Localization(__file__, 609, 34), getitem___72873, (Ellipsis_72863, newaxis_72865))
        
        # Applying the binary operator 'div' (line 609)
        result_div_72875 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 16), 'div', dot_call_result_72862, subscript_call_result_72874)
        
        # Assigning a type to the variable 'p' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'p', result_div_72875)
        
        # Assigning a Call to a Name (line 611):
        
        # Assigning a Call to a Name (line 611):
        
        # Call to nonzero(...): (line 611)
        # Processing the call arguments (line 611)
        # Getting the type of 'z' (line 611)
        z_72878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 27), 'z', False)
        # Processing the call keyword arguments (line 611)
        kwargs_72879 = {}
        # Getting the type of 'np' (line 611)
        np_72876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 611)
        nonzero_72877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), np_72876, 'nonzero')
        # Calling nonzero(args, kwargs) (line 611)
        nonzero_call_result_72880 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), nonzero_72877, *[z_72878], **kwargs_72879)
        
        # Assigning a type to the variable 'r' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'r', nonzero_call_result_72880)
        
        
        
        # Call to len(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'r' (line 612)
        r_72882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 19), 'r', False)
        # Processing the call keyword arguments (line 612)
        kwargs_72883 = {}
        # Getting the type of 'len' (line 612)
        len_72881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'len', False)
        # Calling len(args, kwargs) (line 612)
        len_call_result_72884 = invoke(stypy.reporting.localization.Localization(__file__, 612, 15), len_72881, *[r_72882], **kwargs_72883)
        
        int_72885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 25), 'int')
        # Applying the binary operator '==' (line 612)
        result_eq_72886 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 15), '==', len_call_result_72884, int_72885)
        
        # Testing the type of an if condition (line 612)
        if_condition_72887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 12), result_eq_72886)
        # Assigning a type to the variable 'if_condition_72887' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'if_condition_72887', if_condition_72887)
        # SSA begins for if statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 613)
        # Processing the call arguments (line 613)
        
        # Obtaining the type of the subscript
        int_72889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 25), 'int')
        # Getting the type of 'r' (line 613)
        r_72890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 23), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 613)
        getitem___72891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 23), r_72890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 613)
        subscript_call_result_72892 = invoke(stypy.reporting.localization.Localization(__file__, 613, 23), getitem___72891, int_72889)
        
        # Processing the call keyword arguments (line 613)
        kwargs_72893 = {}
        # Getting the type of 'len' (line 613)
        len_72888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 19), 'len', False)
        # Calling len(args, kwargs) (line 613)
        len_call_result_72894 = invoke(stypy.reporting.localization.Localization(__file__, 613, 19), len_72888, *[subscript_call_result_72892], **kwargs_72893)
        
        int_72895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 31), 'int')
        # Applying the binary operator '>' (line 613)
        result_gt_72896 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 19), '>', len_call_result_72894, int_72895)
        
        # Testing the type of an if condition (line 613)
        if_condition_72897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 16), result_gt_72896)
        # Assigning a type to the variable 'if_condition_72897' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 16), 'if_condition_72897', if_condition_72897)
        # SSA begins for if statement (line 613)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 614):
        
        # Assigning a Subscript to a Name (line 614):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_72898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 37), 'int')
        
        # Obtaining the type of the subscript
        int_72899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 34), 'int')
        # Getting the type of 'r' (line 614)
        r_72900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 32), 'r')
        # Obtaining the member '__getitem__' of a type (line 614)
        getitem___72901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 32), r_72900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 614)
        subscript_call_result_72902 = invoke(stypy.reporting.localization.Localization(__file__, 614, 32), getitem___72901, int_72899)
        
        # Obtaining the member '__getitem__' of a type (line 614)
        getitem___72903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 32), subscript_call_result_72902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 614)
        subscript_call_result_72904 = invoke(stypy.reporting.localization.Localization(__file__, 614, 32), getitem___72903, int_72898)
        
        # Getting the type of 'self' (line 614)
        self_72905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 24), 'self')
        # Obtaining the member 'yi' of a type (line 614)
        yi_72906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 24), self_72905, 'yi')
        # Obtaining the member '__getitem__' of a type (line 614)
        getitem___72907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 24), yi_72906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 614)
        subscript_call_result_72908 = invoke(stypy.reporting.localization.Localization(__file__, 614, 24), getitem___72907, subscript_call_result_72904)
        
        # Assigning a type to the variable 'p' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'p', subscript_call_result_72908)
        # SSA join for if statement (line 613)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 612)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Subscript (line 616):
        
        # Assigning a Subscript to a Subscript (line 616):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_72909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 38), 'int')
        # Getting the type of 'r' (line 616)
        r_72910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 36), 'r')
        # Obtaining the member '__getitem__' of a type (line 616)
        getitem___72911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 36), r_72910, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 616)
        subscript_call_result_72912 = invoke(stypy.reporting.localization.Localization(__file__, 616, 36), getitem___72911, int_72909)
        
        # Getting the type of 'self' (line 616)
        self_72913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 28), 'self')
        # Obtaining the member 'yi' of a type (line 616)
        yi_72914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 28), self_72913, 'yi')
        # Obtaining the member '__getitem__' of a type (line 616)
        getitem___72915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 28), yi_72914, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 616)
        subscript_call_result_72916 = invoke(stypy.reporting.localization.Localization(__file__, 616, 28), getitem___72915, subscript_call_result_72912)
        
        # Getting the type of 'p' (line 616)
        p_72917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'p')
        
        # Obtaining the type of the subscript
        int_72918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 21), 'int')
        slice_72919 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 616, 18), None, int_72918, None)
        # Getting the type of 'r' (line 616)
        r_72920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 18), 'r')
        # Obtaining the member '__getitem__' of a type (line 616)
        getitem___72921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 18), r_72920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 616)
        subscript_call_result_72922 = invoke(stypy.reporting.localization.Localization(__file__, 616, 18), getitem___72921, slice_72919)
        
        # Storing an element on a container (line 616)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 16), p_72917, (subscript_call_result_72922, subscript_call_result_72916))
        # SSA join for if statement (line 612)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 602)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'p' (line 617)
        p_72923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'stypy_return_type', p_72923)
        
        # ################# End of '_evaluate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_evaluate' in the type store
        # Getting the type of 'stypy_return_type' (line 601)
        stypy_return_type_72924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72924)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_evaluate'
        return stypy_return_type_72924


# Assigning a type to the variable 'BarycentricInterpolator' (line 464)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 0), 'BarycentricInterpolator', BarycentricInterpolator)

@norecursion
def barycentric_interpolate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_72925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 44), 'int')
    defaults = [int_72925]
    # Create a new context for function 'barycentric_interpolate'
    module_type_store = module_type_store.open_function_context('barycentric_interpolate', 620, 0, False)
    
    # Passed parameters checking function
    barycentric_interpolate.stypy_localization = localization
    barycentric_interpolate.stypy_type_of_self = None
    barycentric_interpolate.stypy_type_store = module_type_store
    barycentric_interpolate.stypy_function_name = 'barycentric_interpolate'
    barycentric_interpolate.stypy_param_names_list = ['xi', 'yi', 'x', 'axis']
    barycentric_interpolate.stypy_varargs_param_name = None
    barycentric_interpolate.stypy_kwargs_param_name = None
    barycentric_interpolate.stypy_call_defaults = defaults
    barycentric_interpolate.stypy_call_varargs = varargs
    barycentric_interpolate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'barycentric_interpolate', ['xi', 'yi', 'x', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'barycentric_interpolate', localization, ['xi', 'yi', 'x', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'barycentric_interpolate(...)' code ##################

    str_72926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, (-1)), 'str', '\n    Convenience function for polynomial interpolation.\n\n    Constructs a polynomial that passes through a given set of points,\n    then evaluates the polynomial. For reasons of numerical stability,\n    this function does not compute the coefficients of the polynomial.\n\n    This function uses a "barycentric interpolation" method that treats\n    the problem as a special case of rational function interpolation.\n    This algorithm is quite stable, numerically, but even in a world of\n    exact computation, unless the `x` coordinates are chosen very\n    carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -\n    polynomial interpolation itself is a very ill-conditioned process\n    due to the Runge phenomenon.\n\n    Parameters\n    ----------\n    xi : array_like\n        1-d array of x coordinates of the points the polynomial should\n        pass through\n    yi : array_like\n        The y coordinates of the points the polynomial should pass through.\n    x : scalar or array_like\n        Points to evaluate the interpolator at.\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values.\n\n    Returns\n    -------\n    y : scalar or array_like\n        Interpolated values. Shape is determined by replacing\n        the interpolation axis in the original array with the shape of x.\n\n    See Also\n    --------\n    BarycentricInterpolator\n\n    Notes\n    -----\n    Construction of the interpolation weights is a relatively slow process.\n    If you want to call this many times with the same xi (but possibly\n    varying yi or x) you should use the class `BarycentricInterpolator`.\n    This is what this function uses internally.\n\n    ')
    
    # Call to (...): (line 666)
    # Processing the call arguments (line 666)
    # Getting the type of 'x' (line 666)
    x_72934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 54), 'x', False)
    # Processing the call keyword arguments (line 666)
    kwargs_72935 = {}
    
    # Call to BarycentricInterpolator(...): (line 666)
    # Processing the call arguments (line 666)
    # Getting the type of 'xi' (line 666)
    xi_72928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 35), 'xi', False)
    # Getting the type of 'yi' (line 666)
    yi_72929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 39), 'yi', False)
    # Processing the call keyword arguments (line 666)
    # Getting the type of 'axis' (line 666)
    axis_72930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 48), 'axis', False)
    keyword_72931 = axis_72930
    kwargs_72932 = {'axis': keyword_72931}
    # Getting the type of 'BarycentricInterpolator' (line 666)
    BarycentricInterpolator_72927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 11), 'BarycentricInterpolator', False)
    # Calling BarycentricInterpolator(args, kwargs) (line 666)
    BarycentricInterpolator_call_result_72933 = invoke(stypy.reporting.localization.Localization(__file__, 666, 11), BarycentricInterpolator_72927, *[xi_72928, yi_72929], **kwargs_72932)
    
    # Calling (args, kwargs) (line 666)
    _call_result_72936 = invoke(stypy.reporting.localization.Localization(__file__, 666, 11), BarycentricInterpolator_call_result_72933, *[x_72934], **kwargs_72935)
    
    # Assigning a type to the variable 'stypy_return_type' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'stypy_return_type', _call_result_72936)
    
    # ################# End of 'barycentric_interpolate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'barycentric_interpolate' in the type store
    # Getting the type of 'stypy_return_type' (line 620)
    stypy_return_type_72937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72937)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'barycentric_interpolate'
    return stypy_return_type_72937

# Assigning a type to the variable 'barycentric_interpolate' (line 620)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 0), 'barycentric_interpolate', barycentric_interpolate)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
