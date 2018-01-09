
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import math
5: import warnings
6: 
7: # trapz is a public function for scipy.integrate,
8: # even though it's actually a numpy function.
9: from numpy import trapz
10: from scipy.special import roots_legendre
11: from scipy.special import gammaln
12: from scipy._lib.six import xrange
13: 
14: __all__ = ['fixed_quad', 'quadrature', 'romberg', 'trapz', 'simps', 'romb',
15:            'cumtrapz', 'newton_cotes']
16: 
17: 
18: class AccuracyWarning(Warning):
19:     pass
20: 
21: 
22: def _cached_roots_legendre(n):
23:     '''
24:     Cache roots_legendre results to speed up calls of the fixed_quad
25:     function.
26:     '''
27:     if n in _cached_roots_legendre.cache:
28:         return _cached_roots_legendre.cache[n]
29: 
30:     _cached_roots_legendre.cache[n] = roots_legendre(n)
31:     return _cached_roots_legendre.cache[n]
32: _cached_roots_legendre.cache = dict()
33: 
34: 
35: def fixed_quad(func, a, b, args=(), n=5):
36:     '''
37:     Compute a definite integral using fixed-order Gaussian quadrature.
38: 
39:     Integrate `func` from `a` to `b` using Gaussian quadrature of
40:     order `n`.
41: 
42:     Parameters
43:     ----------
44:     func : callable
45:         A Python function or method to integrate (must accept vector inputs).
46:         If integrating a vector-valued function, the returned array must have
47:         shape ``(..., len(x))``.
48:     a : float
49:         Lower limit of integration.
50:     b : float
51:         Upper limit of integration.
52:     args : tuple, optional
53:         Extra arguments to pass to function, if any.
54:     n : int, optional
55:         Order of quadrature integration. Default is 5.
56: 
57:     Returns
58:     -------
59:     val : float
60:         Gaussian quadrature approximation to the integral
61:     none : None
62:         Statically returned value of None
63: 
64: 
65:     See Also
66:     --------
67:     quad : adaptive quadrature using QUADPACK
68:     dblquad : double integrals
69:     tplquad : triple integrals
70:     romberg : adaptive Romberg quadrature
71:     quadrature : adaptive Gaussian quadrature
72:     romb : integrators for sampled data
73:     simps : integrators for sampled data
74:     cumtrapz : cumulative integration for sampled data
75:     ode : ODE integrator
76:     odeint : ODE integrator
77: 
78:     '''
79:     x, w = _cached_roots_legendre(n)
80:     x = np.real(x)
81:     if np.isinf(a) or np.isinf(b):
82:         raise ValueError("Gaussian quadrature is only available for "
83:                          "finite limits.")
84:     y = (b-a)*(x+1)/2.0 + a
85:     return (b-a)/2.0 * np.sum(w*func(y, *args), axis=-1), None
86: 
87: 
88: def vectorize1(func, args=(), vec_func=False):
89:     '''Vectorize the call to a function.
90: 
91:     This is an internal utility function used by `romberg` and
92:     `quadrature` to create a vectorized version of a function.
93: 
94:     If `vec_func` is True, the function `func` is assumed to take vector
95:     arguments.
96: 
97:     Parameters
98:     ----------
99:     func : callable
100:         User defined function.
101:     args : tuple, optional
102:         Extra arguments for the function.
103:     vec_func : bool, optional
104:         True if the function func takes vector arguments.
105: 
106:     Returns
107:     -------
108:     vfunc : callable
109:         A function that will take a vector argument and return the
110:         result.
111: 
112:     '''
113:     if vec_func:
114:         def vfunc(x):
115:             return func(x, *args)
116:     else:
117:         def vfunc(x):
118:             if np.isscalar(x):
119:                 return func(x, *args)
120:             x = np.asarray(x)
121:             # call with first point to get output type
122:             y0 = func(x[0], *args)
123:             n = len(x)
124:             dtype = getattr(y0, 'dtype', type(y0))
125:             output = np.empty((n,), dtype=dtype)
126:             output[0] = y0
127:             for i in xrange(1, n):
128:                 output[i] = func(x[i], *args)
129:             return output
130:     return vfunc
131: 
132: 
133: def quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50,
134:                vec_func=True, miniter=1):
135:     '''
136:     Compute a definite integral using fixed-tolerance Gaussian quadrature.
137: 
138:     Integrate `func` from `a` to `b` using Gaussian quadrature
139:     with absolute tolerance `tol`.
140: 
141:     Parameters
142:     ----------
143:     func : function
144:         A Python function or method to integrate.
145:     a : float
146:         Lower limit of integration.
147:     b : float
148:         Upper limit of integration.
149:     args : tuple, optional
150:         Extra arguments to pass to function.
151:     tol, rtol : float, optional
152:         Iteration stops when error between last two iterates is less than
153:         `tol` OR the relative change is less than `rtol`.
154:     maxiter : int, optional
155:         Maximum order of Gaussian quadrature.
156:     vec_func : bool, optional
157:         True or False if func handles arrays as arguments (is
158:         a "vector" function). Default is True.
159:     miniter : int, optional
160:         Minimum order of Gaussian quadrature.
161: 
162:     Returns
163:     -------
164:     val : float
165:         Gaussian quadrature approximation (within tolerance) to integral.
166:     err : float
167:         Difference between last two estimates of the integral.
168: 
169:     See also
170:     --------
171:     romberg: adaptive Romberg quadrature
172:     fixed_quad: fixed-order Gaussian quadrature
173:     quad: adaptive quadrature using QUADPACK
174:     dblquad: double integrals
175:     tplquad: triple integrals
176:     romb: integrator for sampled data
177:     simps: integrator for sampled data
178:     cumtrapz: cumulative integration for sampled data
179:     ode: ODE integrator
180:     odeint: ODE integrator
181: 
182:     '''
183:     if not isinstance(args, tuple):
184:         args = (args,)
185:     vfunc = vectorize1(func, args, vec_func=vec_func)
186:     val = np.inf
187:     err = np.inf
188:     maxiter = max(miniter+1, maxiter)
189:     for n in xrange(miniter, maxiter+1):
190:         newval = fixed_quad(vfunc, a, b, (), n)[0]
191:         err = abs(newval-val)
192:         val = newval
193: 
194:         if err < tol or err < rtol*abs(val):
195:             break
196:     else:
197:         warnings.warn(
198:             "maxiter (%d) exceeded. Latest difference = %e" % (maxiter, err),
199:             AccuracyWarning)
200:     return val, err
201: 
202: 
203: def tupleset(t, i, value):
204:     l = list(t)
205:     l[i] = value
206:     return tuple(l)
207: 
208: 
209: def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
210:     '''
211:     Cumulatively integrate y(x) using the composite trapezoidal rule.
212: 
213:     Parameters
214:     ----------
215:     y : array_like
216:         Values to integrate.
217:     x : array_like, optional
218:         The coordinate to integrate along.  If None (default), use spacing `dx`
219:         between consecutive elements in `y`.
220:     dx : float, optional
221:         Spacing between elements of `y`.  Only used if `x` is None.
222:     axis : int, optional
223:         Specifies the axis to cumulate.  Default is -1 (last axis).
224:     initial : scalar, optional
225:         If given, uses this value as the first value in the returned result.
226:         Typically this value should be 0.  Default is None, which means no
227:         value at ``x[0]`` is returned and `res` has one element less than `y`
228:         along the axis of integration.
229: 
230:     Returns
231:     -------
232:     res : ndarray
233:         The result of cumulative integration of `y` along `axis`.
234:         If `initial` is None, the shape is such that the axis of integration
235:         has one less value than `y`.  If `initial` is given, the shape is equal
236:         to that of `y`.
237: 
238:     See Also
239:     --------
240:     numpy.cumsum, numpy.cumprod
241:     quad: adaptive quadrature using QUADPACK
242:     romberg: adaptive Romberg quadrature
243:     quadrature: adaptive Gaussian quadrature
244:     fixed_quad: fixed-order Gaussian quadrature
245:     dblquad: double integrals
246:     tplquad: triple integrals
247:     romb: integrators for sampled data
248:     ode: ODE integrators
249:     odeint: ODE integrators
250: 
251:     Examples
252:     --------
253:     >>> from scipy import integrate
254:     >>> import matplotlib.pyplot as plt
255: 
256:     >>> x = np.linspace(-2, 2, num=20)
257:     >>> y = x
258:     >>> y_int = integrate.cumtrapz(y, x, initial=0)
259:     >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
260:     >>> plt.show()
261: 
262:     '''
263:     y = np.asarray(y)
264:     if x is None:
265:         d = dx
266:     else:
267:         x = np.asarray(x)
268:         if x.ndim == 1:
269:             d = np.diff(x)
270:             # reshape to correct shape
271:             shape = [1] * y.ndim
272:             shape[axis] = -1
273:             d = d.reshape(shape)
274:         elif len(x.shape) != len(y.shape):
275:             raise ValueError("If given, shape of x must be 1-d or the "
276:                              "same as y.")
277:         else:
278:             d = np.diff(x, axis=axis)
279: 
280:         if d.shape[axis] != y.shape[axis] - 1:
281:             raise ValueError("If given, length of x along axis must be the "
282:                              "same as y.")
283: 
284:     nd = len(y.shape)
285:     slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
286:     slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
287:     res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
288: 
289:     if initial is not None:
290:         if not np.isscalar(initial):
291:             raise ValueError("`initial` parameter should be a scalar.")
292: 
293:         shape = list(res.shape)
294:         shape[axis] = 1
295:         res = np.concatenate([np.ones(shape, dtype=res.dtype) * initial, res],
296:                              axis=axis)
297: 
298:     return res
299: 
300: 
301: def _basic_simps(y, start, stop, x, dx, axis):
302:     nd = len(y.shape)
303:     if start is None:
304:         start = 0
305:     step = 2
306:     slice_all = (slice(None),)*nd
307:     slice0 = tupleset(slice_all, axis, slice(start, stop, step))
308:     slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
309:     slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))
310: 
311:     if x is None:  # Even spaced Simpson's rule.
312:         result = np.sum(dx/3.0 * (y[slice0]+4*y[slice1]+y[slice2]),
313:                         axis=axis)
314:     else:
315:         # Account for possibly different spacings.
316:         #    Simpson's rule changes a bit.
317:         h = np.diff(x, axis=axis)
318:         sl0 = tupleset(slice_all, axis, slice(start, stop, step))
319:         sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
320:         h0 = h[sl0]
321:         h1 = h[sl1]
322:         hsum = h0 + h1
323:         hprod = h0 * h1
324:         h0divh1 = h0 / h1
325:         tmp = hsum/6.0 * (y[slice0]*(2-1.0/h0divh1) +
326:                           y[slice1]*hsum*hsum/hprod +
327:                           y[slice2]*(2-h0divh1))
328:         result = np.sum(tmp, axis=axis)
329:     return result
330: 
331: 
332: def simps(y, x=None, dx=1, axis=-1, even='avg'):
333:     '''
334:     Integrate y(x) using samples along the given axis and the composite
335:     Simpson's rule.  If x is None, spacing of dx is assumed.
336: 
337:     If there are an even number of samples, N, then there are an odd
338:     number of intervals (N-1), but Simpson's rule requires an even number
339:     of intervals.  The parameter 'even' controls how this is handled.
340: 
341:     Parameters
342:     ----------
343:     y : array_like
344:         Array to be integrated.
345:     x : array_like, optional
346:         If given, the points at which `y` is sampled.
347:     dx : int, optional
348:         Spacing of integration points along axis of `y`. Only used when
349:         `x` is None. Default is 1.
350:     axis : int, optional
351:         Axis along which to integrate. Default is the last axis.
352:     even : str {'avg', 'first', 'last'}, optional
353:         'avg' : Average two results:1) use the first N-2 intervals with
354:                   a trapezoidal rule on the last interval and 2) use the last
355:                   N-2 intervals with a trapezoidal rule on the first interval.
356: 
357:         'first' : Use Simpson's rule for the first N-2 intervals with
358:                 a trapezoidal rule on the last interval.
359: 
360:         'last' : Use Simpson's rule for the last N-2 intervals with a
361:                trapezoidal rule on the first interval.
362: 
363:     See Also
364:     --------
365:     quad: adaptive quadrature using QUADPACK
366:     romberg: adaptive Romberg quadrature
367:     quadrature: adaptive Gaussian quadrature
368:     fixed_quad: fixed-order Gaussian quadrature
369:     dblquad: double integrals
370:     tplquad: triple integrals
371:     romb: integrators for sampled data
372:     cumtrapz: cumulative integration for sampled data
373:     ode: ODE integrators
374:     odeint: ODE integrators
375: 
376:     Notes
377:     -----
378:     For an odd number of samples that are equally spaced the result is
379:     exact if the function is a polynomial of order 3 or less.  If
380:     the samples are not equally spaced, then the result is exact only
381:     if the function is a polynomial of order 2 or less.
382: 
383:     '''
384:     y = np.asarray(y)
385:     nd = len(y.shape)
386:     N = y.shape[axis]
387:     last_dx = dx
388:     first_dx = dx
389:     returnshape = 0
390:     if x is not None:
391:         x = np.asarray(x)
392:         if len(x.shape) == 1:
393:             shapex = [1] * nd
394:             shapex[axis] = x.shape[0]
395:             saveshape = x.shape
396:             returnshape = 1
397:             x = x.reshape(tuple(shapex))
398:         elif len(x.shape) != len(y.shape):
399:             raise ValueError("If given, shape of x must be 1-d or the "
400:                              "same as y.")
401:         if x.shape[axis] != N:
402:             raise ValueError("If given, length of x along axis must be the "
403:                              "same as y.")
404:     if N % 2 == 0:
405:         val = 0.0
406:         result = 0.0
407:         slice1 = (slice(None),)*nd
408:         slice2 = (slice(None),)*nd
409:         if even not in ['avg', 'last', 'first']:
410:             raise ValueError("Parameter 'even' must be "
411:                              "'avg', 'last', or 'first'.")
412:         # Compute using Simpson's rule on first intervals
413:         if even in ['avg', 'first']:
414:             slice1 = tupleset(slice1, axis, -1)
415:             slice2 = tupleset(slice2, axis, -2)
416:             if x is not None:
417:                 last_dx = x[slice1] - x[slice2]
418:             val += 0.5*last_dx*(y[slice1]+y[slice2])
419:             result = _basic_simps(y, 0, N-3, x, dx, axis)
420:         # Compute using Simpson's rule on last set of intervals
421:         if even in ['avg', 'last']:
422:             slice1 = tupleset(slice1, axis, 0)
423:             slice2 = tupleset(slice2, axis, 1)
424:             if x is not None:
425:                 first_dx = x[tuple(slice2)] - x[tuple(slice1)]
426:             val += 0.5*first_dx*(y[slice2]+y[slice1])
427:             result += _basic_simps(y, 1, N-2, x, dx, axis)
428:         if even == 'avg':
429:             val /= 2.0
430:             result /= 2.0
431:         result = result + val
432:     else:
433:         result = _basic_simps(y, 0, N-2, x, dx, axis)
434:     if returnshape:
435:         x = x.reshape(saveshape)
436:     return result
437: 
438: 
439: def romb(y, dx=1.0, axis=-1, show=False):
440:     '''
441:     Romberg integration using samples of a function.
442: 
443:     Parameters
444:     ----------
445:     y : array_like
446:         A vector of ``2**k + 1`` equally-spaced samples of a function.
447:     dx : float, optional
448:         The sample spacing. Default is 1.
449:     axis : int, optional
450:         The axis along which to integrate. Default is -1 (last axis).
451:     show : bool, optional
452:         When `y` is a single 1-D array, then if this argument is True
453:         print the table showing Richardson extrapolation from the
454:         samples. Default is False.
455: 
456:     Returns
457:     -------
458:     romb : ndarray
459:         The integrated result for `axis`.
460: 
461:     See also
462:     --------
463:     quad : adaptive quadrature using QUADPACK
464:     romberg : adaptive Romberg quadrature
465:     quadrature : adaptive Gaussian quadrature
466:     fixed_quad : fixed-order Gaussian quadrature
467:     dblquad : double integrals
468:     tplquad : triple integrals
469:     simps : integrators for sampled data
470:     cumtrapz : cumulative integration for sampled data
471:     ode : ODE integrators
472:     odeint : ODE integrators
473: 
474:     '''
475:     y = np.asarray(y)
476:     nd = len(y.shape)
477:     Nsamps = y.shape[axis]
478:     Ninterv = Nsamps-1
479:     n = 1
480:     k = 0
481:     while n < Ninterv:
482:         n <<= 1
483:         k += 1
484:     if n != Ninterv:
485:         raise ValueError("Number of samples must be one plus a "
486:                          "non-negative power of 2.")
487: 
488:     R = {}
489:     slice_all = (slice(None),) * nd
490:     slice0 = tupleset(slice_all, axis, 0)
491:     slicem1 = tupleset(slice_all, axis, -1)
492:     h = Ninterv * np.asarray(dx, dtype=float)
493:     R[(0, 0)] = (y[slice0] + y[slicem1])/2.0*h
494:     slice_R = slice_all
495:     start = stop = step = Ninterv
496:     for i in xrange(1, k+1):
497:         start >>= 1
498:         slice_R = tupleset(slice_R, axis, slice(start, stop, step))
499:         step >>= 1
500:         R[(i, 0)] = 0.5*(R[(i-1, 0)] + h*y[slice_R].sum(axis=axis))
501:         for j in xrange(1, i+1):
502:             prev = R[(i, j-1)]
503:             R[(i, j)] = prev + (prev-R[(i-1, j-1)]) / ((1 << (2*j))-1)
504:         h /= 2.0
505: 
506:     if show:
507:         if not np.isscalar(R[(0, 0)]):
508:             print("*** Printing table only supported for integrals" +
509:                   " of a single data set.")
510:         else:
511:             try:
512:                 precis = show[0]
513:             except (TypeError, IndexError):
514:                 precis = 5
515:             try:
516:                 width = show[1]
517:             except (TypeError, IndexError):
518:                 width = 8
519:             formstr = "%%%d.%df" % (width, precis)
520: 
521:             title = "Richardson Extrapolation Table for Romberg Integration"
522:             print("", title.center(68), "=" * 68, sep="\n", end="")
523:             for i in xrange(k+1):
524:                 for j in xrange(i+1):
525:                     print(formstr % R[(i, j)], end=" ")
526:                 print()
527:             print("=" * 68)
528:             print()
529: 
530:     return R[(k, k)]
531: 
532: # Romberg quadratures for numeric integration.
533: #
534: # Written by Scott M. Ransom <ransom@cfa.harvard.edu>
535: # last revision: 14 Nov 98
536: #
537: # Cosmetic changes by Konrad Hinsen <hinsen@cnrs-orleans.fr>
538: # last revision: 1999-7-21
539: #
540: # Adapted to scipy by Travis Oliphant <oliphant.travis@ieee.org>
541: # last revision: Dec 2001
542: 
543: 
544: def _difftrap(function, interval, numtraps):
545:     '''
546:     Perform part of the trapezoidal rule to integrate a function.
547:     Assume that we had called difftrap with all lower powers-of-2
548:     starting with 1.  Calling difftrap only returns the summation
549:     of the new ordinates.  It does _not_ multiply by the width
550:     of the trapezoids.  This must be performed by the caller.
551:         'function' is the function to evaluate (must accept vector arguments).
552:         'interval' is a sequence with lower and upper limits
553:                    of integration.
554:         'numtraps' is the number of trapezoids to use (must be a
555:                    power-of-2).
556:     '''
557:     if numtraps <= 0:
558:         raise ValueError("numtraps must be > 0 in difftrap().")
559:     elif numtraps == 1:
560:         return 0.5*(function(interval[0])+function(interval[1]))
561:     else:
562:         numtosum = numtraps/2
563:         h = float(interval[1]-interval[0])/numtosum
564:         lox = interval[0] + 0.5 * h
565:         points = lox + h * np.arange(numtosum)
566:         s = np.sum(function(points), axis=0)
567:         return s
568: 
569: 
570: def _romberg_diff(b, c, k):
571:     '''
572:     Compute the differences for the Romberg quadrature corrections.
573:     See Forman Acton's "Real Computing Made Real," p 143.
574:     '''
575:     tmp = 4.0**k
576:     return (tmp * c - b)/(tmp - 1.0)
577: 
578: 
579: def _printresmat(function, interval, resmat):
580:     # Print the Romberg result matrix.
581:     i = j = 0
582:     print('Romberg integration of', repr(function), end=' ')
583:     print('from', interval)
584:     print('')
585:     print('%6s %9s %9s' % ('Steps', 'StepSize', 'Results'))
586:     for i in xrange(len(resmat)):
587:         print('%6d %9f' % (2**i, (interval[1]-interval[0])/(2.**i)), end=' ')
588:         for j in xrange(i+1):
589:             print('%9f' % (resmat[i][j]), end=' ')
590:         print('')
591:     print('')
592:     print('The final result is', resmat[i][j], end=' ')
593:     print('after', 2**(len(resmat)-1)+1, 'function evaluations.')
594: 
595: 
596: def romberg(function, a, b, args=(), tol=1.48e-8, rtol=1.48e-8, show=False,
597:             divmax=10, vec_func=False):
598:     '''
599:     Romberg integration of a callable function or method.
600: 
601:     Returns the integral of `function` (a function of one variable)
602:     over the interval (`a`, `b`).
603: 
604:     If `show` is 1, the triangular array of the intermediate results
605:     will be printed.  If `vec_func` is True (default is False), then
606:     `function` is assumed to support vector arguments.
607: 
608:     Parameters
609:     ----------
610:     function : callable
611:         Function to be integrated.
612:     a : float
613:         Lower limit of integration.
614:     b : float
615:         Upper limit of integration.
616: 
617:     Returns
618:     -------
619:     results  : float
620:         Result of the integration.
621: 
622:     Other Parameters
623:     ----------------
624:     args : tuple, optional
625:         Extra arguments to pass to function. Each element of `args` will
626:         be passed as a single argument to `func`. Default is to pass no
627:         extra arguments.
628:     tol, rtol : float, optional
629:         The desired absolute and relative tolerances. Defaults are 1.48e-8.
630:     show : bool, optional
631:         Whether to print the results. Default is False.
632:     divmax : int, optional
633:         Maximum order of extrapolation. Default is 10.
634:     vec_func : bool, optional
635:         Whether `func` handles arrays as arguments (i.e whether it is a
636:         "vector" function). Default is False.
637: 
638:     See Also
639:     --------
640:     fixed_quad : Fixed-order Gaussian quadrature.
641:     quad : Adaptive quadrature using QUADPACK.
642:     dblquad : Double integrals.
643:     tplquad : Triple integrals.
644:     romb : Integrators for sampled data.
645:     simps : Integrators for sampled data.
646:     cumtrapz : Cumulative integration for sampled data.
647:     ode : ODE integrator.
648:     odeint : ODE integrator.
649: 
650:     References
651:     ----------
652:     .. [1] 'Romberg's method' http://en.wikipedia.org/wiki/Romberg%27s_method
653: 
654:     Examples
655:     --------
656:     Integrate a gaussian from 0 to 1 and compare to the error function.
657: 
658:     >>> from scipy import integrate
659:     >>> from scipy.special import erf
660:     >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
661:     >>> result = integrate.romberg(gaussian, 0, 1, show=True)
662:     Romberg integration of <function vfunc at ...> from [0, 1]
663: 
664:     ::
665: 
666:        Steps  StepSize  Results
667:            1  1.000000  0.385872
668:            2  0.500000  0.412631  0.421551
669:            4  0.250000  0.419184  0.421368  0.421356
670:            8  0.125000  0.420810  0.421352  0.421350  0.421350
671:           16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
672:           32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
673: 
674:     The final result is 0.421350396475 after 33 function evaluations.
675: 
676:     >>> print("%g %g" % (2*result, erf(1)))
677:     0.842701 0.842701
678: 
679:     '''
680:     if np.isinf(a) or np.isinf(b):
681:         raise ValueError("Romberg integration only available "
682:                          "for finite limits.")
683:     vfunc = vectorize1(function, args, vec_func=vec_func)
684:     n = 1
685:     interval = [a, b]
686:     intrange = b - a
687:     ordsum = _difftrap(vfunc, interval, n)
688:     result = intrange * ordsum
689:     resmat = [[result]]
690:     err = np.inf
691:     last_row = resmat[0]
692:     for i in xrange(1, divmax+1):
693:         n *= 2
694:         ordsum += _difftrap(vfunc, interval, n)
695:         row = [intrange * ordsum / n]
696:         for k in xrange(i):
697:             row.append(_romberg_diff(last_row[k], row[k], k+1))
698:         result = row[i]
699:         lastresult = last_row[i-1]
700:         if show:
701:             resmat.append(row)
702:         err = abs(result - lastresult)
703:         if err < tol or err < rtol * abs(result):
704:             break
705:         last_row = row
706:     else:
707:         warnings.warn(
708:             "divmax (%d) exceeded. Latest difference = %e" % (divmax, err),
709:             AccuracyWarning)
710: 
711:     if show:
712:         _printresmat(vfunc, interval, resmat)
713:     return result
714: 
715: 
716: # Coefficients for Netwon-Cotes quadrature
717: #
718: # These are the points being used
719: #  to construct the local interpolating polynomial
720: #  a are the weights for Newton-Cotes integration
721: #  B is the error coefficient.
722: #  error in these coefficients grows as N gets larger.
723: #  or as samples are closer and closer together
724: 
725: # You can use maxima to find these rational coefficients
726: #  for equally spaced data using the commands
727: #  a(i,N) := integrate(product(r-j,j,0,i-1) * product(r-j,j,i+1,N),r,0,N) / ((N-i)! * i!) * (-1)^(N-i);
728: #  Be(N) := N^(N+2)/(N+2)! * (N/(N+3) - sum((i/N)^(N+2)*a(i,N),i,0,N));
729: #  Bo(N) := N^(N+1)/(N+1)! * (N/(N+2) - sum((i/N)^(N+1)*a(i,N),i,0,N));
730: #  B(N) := (if (mod(N,2)=0) then Be(N) else Bo(N));
731: #
732: # pre-computed for equally-spaced weights
733: #
734: # num_a, den_a, int_a, num_B, den_B = _builtincoeffs[N]
735: #
736: #  a = num_a*array(int_a)/den_a
737: #  B = num_B*1.0 / den_B
738: #
739: #  integrate(f(x),x,x_0,x_N) = dx*sum(a*f(x_i)) + B*(dx)^(2k+3) f^(2k+2)(x*)
740: #    where k = N // 2
741: #
742: _builtincoeffs = {
743:     1: (1,2,[1,1],-1,12),
744:     2: (1,3,[1,4,1],-1,90),
745:     3: (3,8,[1,3,3,1],-3,80),
746:     4: (2,45,[7,32,12,32,7],-8,945),
747:     5: (5,288,[19,75,50,50,75,19],-275,12096),
748:     6: (1,140,[41,216,27,272,27,216,41],-9,1400),
749:     7: (7,17280,[751,3577,1323,2989,2989,1323,3577,751],-8183,518400),
750:     8: (4,14175,[989,5888,-928,10496,-4540,10496,-928,5888,989],
751:         -2368,467775),
752:     9: (9,89600,[2857,15741,1080,19344,5778,5778,19344,1080,
753:                  15741,2857], -4671, 394240),
754:     10: (5,299376,[16067,106300,-48525,272400,-260550,427368,
755:                    -260550,272400,-48525,106300,16067],
756:          -673175, 163459296),
757:     11: (11,87091200,[2171465,13486539,-3237113, 25226685,-9595542,
758:                       15493566,15493566,-9595542,25226685,-3237113,
759:                       13486539,2171465], -2224234463, 237758976000),
760:     12: (1, 5255250, [1364651,9903168,-7587864,35725120,-51491295,
761:                       87516288,-87797136,87516288,-51491295,35725120,
762:                       -7587864,9903168,1364651], -3012, 875875),
763:     13: (13, 402361344000,[8181904909, 56280729661, -31268252574,
764:                            156074417954,-151659573325,206683437987,
765:                            -43111992612,-43111992612,206683437987,
766:                            -151659573325,156074417954,-31268252574,
767:                            56280729661,8181904909], -2639651053,
768:          344881152000),
769:     14: (7, 2501928000, [90241897,710986864,-770720657,3501442784,
770:                          -6625093363,12630121616,-16802270373,19534438464,
771:                          -16802270373,12630121616,-6625093363,3501442784,
772:                          -770720657,710986864,90241897], -3740727473,
773:          1275983280000)
774:     }
775: 
776: 
777: def newton_cotes(rn, equal=0):
778:     '''
779:     Return weights and error coefficient for Newton-Cotes integration.
780: 
781:     Suppose we have (N+1) samples of f at the positions
782:     x_0, x_1, ..., x_N.  Then an N-point Newton-Cotes formula for the
783:     integral between x_0 and x_N is:
784: 
785:     :math:`\\int_{x_0}^{x_N} f(x)dx = \\Delta x \\sum_{i=0}^{N} a_i f(x_i)
786:     + B_N (\\Delta x)^{N+2} f^{N+1} (\\xi)`
787: 
788:     where :math:`\\xi \\in [x_0,x_N]`
789:     and :math:`\\Delta x = \\frac{x_N-x_0}{N}` is the average samples spacing.
790: 
791:     If the samples are equally-spaced and N is even, then the error
792:     term is :math:`B_N (\\Delta x)^{N+3} f^{N+2}(\\xi)`.
793: 
794:     Parameters
795:     ----------
796:     rn : int
797:         The integer order for equally-spaced data or the relative positions of
798:         the samples with the first sample at 0 and the last at N, where N+1 is
799:         the length of `rn`.  N is the order of the Newton-Cotes integration.
800:     equal : int, optional
801:         Set to 1 to enforce equally spaced data.
802: 
803:     Returns
804:     -------
805:     an : ndarray
806:         1-D array of weights to apply to the function at the provided sample
807:         positions.
808:     B : float
809:         Error coefficient.
810: 
811:     Notes
812:     -----
813:     Normally, the Newton-Cotes rules are used on smaller integration
814:     regions and a composite rule is used to return the total integral.
815: 
816:     '''
817:     try:
818:         N = len(rn)-1
819:         if equal:
820:             rn = np.arange(N+1)
821:         elif np.all(np.diff(rn) == 1):
822:             equal = 1
823:     except:
824:         N = rn
825:         rn = np.arange(N+1)
826:         equal = 1
827: 
828:     if equal and N in _builtincoeffs:
829:         na, da, vi, nb, db = _builtincoeffs[N]
830:         an = na * np.array(vi, dtype=float) / da
831:         return an, float(nb)/db
832: 
833:     if (rn[0] != 0) or (rn[-1] != N):
834:         raise ValueError("The sample positions must start at 0"
835:                          " and end at N")
836:     yi = rn / float(N)
837:     ti = 2 * yi - 1
838:     nvec = np.arange(N+1)
839:     C = ti ** nvec[:, np.newaxis]
840:     Cinv = np.linalg.inv(C)
841:     # improve precision of result
842:     for i in range(2):
843:         Cinv = 2*Cinv - Cinv.dot(C).dot(Cinv)
844:     vec = 2.0 / (nvec[::2]+1)
845:     ai = Cinv[:, ::2].dot(vec) * (N / 2.)
846: 
847:     if (N % 2 == 0) and equal:
848:         BN = N/(N+3.)
849:         power = N+2
850:     else:
851:         BN = N/(N+2.)
852:         power = N+1
853: 
854:     BN = BN - np.dot(yi**power, ai)
855:     p1 = power+1
856:     fac = power*math.log(N) - gammaln(p1)
857:     fac = math.exp(fac)
858:     return ai, BN*fac
859: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_29999 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_29999) is not StypyTypeError):

    if (import_29999 != 'pyd_module'):
        __import__(import_29999)
        sys_modules_30000 = sys.modules[import_29999]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_30000.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_29999)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import math' statement (line 4)
import math

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import warnings' statement (line 5)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy import trapz' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_30001 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_30001) is not StypyTypeError):

    if (import_30001 != 'pyd_module'):
        __import__(import_30001)
        sys_modules_30002 = sys.modules[import_30001]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', sys_modules_30002.module_type_store, module_type_store, ['trapz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_30002, sys_modules_30002.module_type_store, module_type_store)
    else:
        from numpy import trapz

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', None, module_type_store, ['trapz'], [trapz])

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_30001)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.special import roots_legendre' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_30003 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special')

if (type(import_30003) is not StypyTypeError):

    if (import_30003 != 'pyd_module'):
        __import__(import_30003)
        sys_modules_30004 = sys.modules[import_30003]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', sys_modules_30004.module_type_store, module_type_store, ['roots_legendre'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_30004, sys_modules_30004.module_type_store, module_type_store)
    else:
        from scipy.special import roots_legendre

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', None, module_type_store, ['roots_legendre'], [roots_legendre])

else:
    # Assigning a type to the variable 'scipy.special' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', import_30003)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.special import gammaln' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_30005 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special')

if (type(import_30005) is not StypyTypeError):

    if (import_30005 != 'pyd_module'):
        __import__(import_30005)
        sys_modules_30006 = sys.modules[import_30005]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', sys_modules_30006.module_type_store, module_type_store, ['gammaln'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_30006, sys_modules_30006.module_type_store, module_type_store)
    else:
        from scipy.special import gammaln

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', None, module_type_store, ['gammaln'], [gammaln])

else:
    # Assigning a type to the variable 'scipy.special' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', import_30005)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib.six import xrange' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_30007 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six')

if (type(import_30007) is not StypyTypeError):

    if (import_30007 != 'pyd_module'):
        __import__(import_30007)
        sys_modules_30008 = sys.modules[import_30007]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', sys_modules_30008.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_30008, sys_modules_30008.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', import_30007)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['fixed_quad', 'quadrature', 'romberg', 'trapz', 'simps', 'romb', 'cumtrapz', 'newton_cotes']
module_type_store.set_exportable_members(['fixed_quad', 'quadrature', 'romberg', 'trapz', 'simps', 'romb', 'cumtrapz', 'newton_cotes'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_30009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_30010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'fixed_quad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30010)
# Adding element type (line 14)
str_30011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'str', 'quadrature')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30011)
# Adding element type (line 14)
str_30012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'str', 'romberg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30012)
# Adding element type (line 14)
str_30013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 50), 'str', 'trapz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30013)
# Adding element type (line 14)
str_30014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 59), 'str', 'simps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30014)
# Adding element type (line 14)
str_30015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 68), 'str', 'romb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30015)
# Adding element type (line 14)
str_30016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'cumtrapz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30016)
# Adding element type (line 14)
str_30017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'str', 'newton_cotes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_30009, str_30017)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_30009)
# Declaration of the 'AccuracyWarning' class
# Getting the type of 'Warning' (line 18)
Warning_30018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'Warning')

class AccuracyWarning(Warning_30018, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 0, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AccuracyWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'AccuracyWarning' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'AccuracyWarning', AccuracyWarning)

@norecursion
def _cached_roots_legendre(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_cached_roots_legendre'
    module_type_store = module_type_store.open_function_context('_cached_roots_legendre', 22, 0, False)
    
    # Passed parameters checking function
    _cached_roots_legendre.stypy_localization = localization
    _cached_roots_legendre.stypy_type_of_self = None
    _cached_roots_legendre.stypy_type_store = module_type_store
    _cached_roots_legendre.stypy_function_name = '_cached_roots_legendre'
    _cached_roots_legendre.stypy_param_names_list = ['n']
    _cached_roots_legendre.stypy_varargs_param_name = None
    _cached_roots_legendre.stypy_kwargs_param_name = None
    _cached_roots_legendre.stypy_call_defaults = defaults
    _cached_roots_legendre.stypy_call_varargs = varargs
    _cached_roots_legendre.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cached_roots_legendre', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cached_roots_legendre', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cached_roots_legendre(...)' code ##################

    str_30019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\n    Cache roots_legendre results to speed up calls of the fixed_quad\n    function.\n    ')
    
    
    # Getting the type of 'n' (line 27)
    n_30020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'n')
    # Getting the type of '_cached_roots_legendre' (line 27)
    _cached_roots_legendre_30021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), '_cached_roots_legendre')
    # Obtaining the member 'cache' of a type (line 27)
    cache_30022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), _cached_roots_legendre_30021, 'cache')
    # Applying the binary operator 'in' (line 27)
    result_contains_30023 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), 'in', n_30020, cache_30022)
    
    # Testing the type of an if condition (line 27)
    if_condition_30024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_contains_30023)
    # Assigning a type to the variable 'if_condition_30024' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_30024', if_condition_30024)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 28)
    n_30025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'n')
    # Getting the type of '_cached_roots_legendre' (line 28)
    _cached_roots_legendre_30026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), '_cached_roots_legendre')
    # Obtaining the member 'cache' of a type (line 28)
    cache_30027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), _cached_roots_legendre_30026, 'cache')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___30028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), cache_30027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_30029 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), getitem___30028, n_30025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', subscript_call_result_30029)
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 30):
    
    # Assigning a Call to a Subscript (line 30):
    
    # Call to roots_legendre(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'n' (line 30)
    n_30031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 53), 'n', False)
    # Processing the call keyword arguments (line 30)
    kwargs_30032 = {}
    # Getting the type of 'roots_legendre' (line 30)
    roots_legendre_30030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 38), 'roots_legendre', False)
    # Calling roots_legendre(args, kwargs) (line 30)
    roots_legendre_call_result_30033 = invoke(stypy.reporting.localization.Localization(__file__, 30, 38), roots_legendre_30030, *[n_30031], **kwargs_30032)
    
    # Getting the type of '_cached_roots_legendre' (line 30)
    _cached_roots_legendre_30034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), '_cached_roots_legendre')
    # Obtaining the member 'cache' of a type (line 30)
    cache_30035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), _cached_roots_legendre_30034, 'cache')
    # Getting the type of 'n' (line 30)
    n_30036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'n')
    # Storing an element on a container (line 30)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), cache_30035, (n_30036, roots_legendre_call_result_30033))
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 31)
    n_30037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'n')
    # Getting the type of '_cached_roots_legendre' (line 31)
    _cached_roots_legendre_30038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), '_cached_roots_legendre')
    # Obtaining the member 'cache' of a type (line 31)
    cache_30039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), _cached_roots_legendre_30038, 'cache')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___30040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), cache_30039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_30041 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), getitem___30040, n_30037)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', subscript_call_result_30041)
    
    # ################# End of '_cached_roots_legendre(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cached_roots_legendre' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_30042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30042)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cached_roots_legendre'
    return stypy_return_type_30042

# Assigning a type to the variable '_cached_roots_legendre' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_cached_roots_legendre', _cached_roots_legendre)

# Assigning a Call to a Attribute (line 32):

# Assigning a Call to a Attribute (line 32):

# Call to dict(...): (line 32)
# Processing the call keyword arguments (line 32)
kwargs_30044 = {}
# Getting the type of 'dict' (line 32)
dict_30043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'dict', False)
# Calling dict(args, kwargs) (line 32)
dict_call_result_30045 = invoke(stypy.reporting.localization.Localization(__file__, 32, 31), dict_30043, *[], **kwargs_30044)

# Getting the type of '_cached_roots_legendre' (line 32)
_cached_roots_legendre_30046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_cached_roots_legendre')
# Setting the type of the member 'cache' of a type (line 32)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 0), _cached_roots_legendre_30046, 'cache', dict_call_result_30045)

@norecursion
def fixed_quad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_30047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    
    int_30048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 38), 'int')
    defaults = [tuple_30047, int_30048]
    # Create a new context for function 'fixed_quad'
    module_type_store = module_type_store.open_function_context('fixed_quad', 35, 0, False)
    
    # Passed parameters checking function
    fixed_quad.stypy_localization = localization
    fixed_quad.stypy_type_of_self = None
    fixed_quad.stypy_type_store = module_type_store
    fixed_quad.stypy_function_name = 'fixed_quad'
    fixed_quad.stypy_param_names_list = ['func', 'a', 'b', 'args', 'n']
    fixed_quad.stypy_varargs_param_name = None
    fixed_quad.stypy_kwargs_param_name = None
    fixed_quad.stypy_call_defaults = defaults
    fixed_quad.stypy_call_varargs = varargs
    fixed_quad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fixed_quad', ['func', 'a', 'b', 'args', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fixed_quad', localization, ['func', 'a', 'b', 'args', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fixed_quad(...)' code ##################

    str_30049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n    Compute a definite integral using fixed-order Gaussian quadrature.\n\n    Integrate `func` from `a` to `b` using Gaussian quadrature of\n    order `n`.\n\n    Parameters\n    ----------\n    func : callable\n        A Python function or method to integrate (must accept vector inputs).\n        If integrating a vector-valued function, the returned array must have\n        shape ``(..., len(x))``.\n    a : float\n        Lower limit of integration.\n    b : float\n        Upper limit of integration.\n    args : tuple, optional\n        Extra arguments to pass to function, if any.\n    n : int, optional\n        Order of quadrature integration. Default is 5.\n\n    Returns\n    -------\n    val : float\n        Gaussian quadrature approximation to the integral\n    none : None\n        Statically returned value of None\n\n\n    See Also\n    --------\n    quad : adaptive quadrature using QUADPACK\n    dblquad : double integrals\n    tplquad : triple integrals\n    romberg : adaptive Romberg quadrature\n    quadrature : adaptive Gaussian quadrature\n    romb : integrators for sampled data\n    simps : integrators for sampled data\n    cumtrapz : cumulative integration for sampled data\n    ode : ODE integrator\n    odeint : ODE integrator\n\n    ')
    
    # Assigning a Call to a Tuple (line 79):
    
    # Assigning a Subscript to a Name (line 79):
    
    # Obtaining the type of the subscript
    int_30050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'int')
    
    # Call to _cached_roots_legendre(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'n' (line 79)
    n_30052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'n', False)
    # Processing the call keyword arguments (line 79)
    kwargs_30053 = {}
    # Getting the type of '_cached_roots_legendre' (line 79)
    _cached_roots_legendre_30051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), '_cached_roots_legendre', False)
    # Calling _cached_roots_legendre(args, kwargs) (line 79)
    _cached_roots_legendre_call_result_30054 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), _cached_roots_legendre_30051, *[n_30052], **kwargs_30053)
    
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___30055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), _cached_roots_legendre_call_result_30054, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_30056 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), getitem___30055, int_30050)
    
    # Assigning a type to the variable 'tuple_var_assignment_29992' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_29992', subscript_call_result_30056)
    
    # Assigning a Subscript to a Name (line 79):
    
    # Obtaining the type of the subscript
    int_30057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'int')
    
    # Call to _cached_roots_legendre(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'n' (line 79)
    n_30059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'n', False)
    # Processing the call keyword arguments (line 79)
    kwargs_30060 = {}
    # Getting the type of '_cached_roots_legendre' (line 79)
    _cached_roots_legendre_30058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), '_cached_roots_legendre', False)
    # Calling _cached_roots_legendre(args, kwargs) (line 79)
    _cached_roots_legendre_call_result_30061 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), _cached_roots_legendre_30058, *[n_30059], **kwargs_30060)
    
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___30062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), _cached_roots_legendre_call_result_30061, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_30063 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), getitem___30062, int_30057)
    
    # Assigning a type to the variable 'tuple_var_assignment_29993' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_29993', subscript_call_result_30063)
    
    # Assigning a Name to a Name (line 79):
    # Getting the type of 'tuple_var_assignment_29992' (line 79)
    tuple_var_assignment_29992_30064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_29992')
    # Assigning a type to the variable 'x' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'x', tuple_var_assignment_29992_30064)
    
    # Assigning a Name to a Name (line 79):
    # Getting the type of 'tuple_var_assignment_29993' (line 79)
    tuple_var_assignment_29993_30065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_29993')
    # Assigning a type to the variable 'w' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'w', tuple_var_assignment_29993_30065)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to real(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'x' (line 80)
    x_30068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'x', False)
    # Processing the call keyword arguments (line 80)
    kwargs_30069 = {}
    # Getting the type of 'np' (line 80)
    np_30066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'np', False)
    # Obtaining the member 'real' of a type (line 80)
    real_30067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), np_30066, 'real')
    # Calling real(args, kwargs) (line 80)
    real_call_result_30070 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), real_30067, *[x_30068], **kwargs_30069)
    
    # Assigning a type to the variable 'x' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'x', real_call_result_30070)
    
    
    # Evaluating a boolean operation
    
    # Call to isinf(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'a' (line 81)
    a_30073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'a', False)
    # Processing the call keyword arguments (line 81)
    kwargs_30074 = {}
    # Getting the type of 'np' (line 81)
    np_30071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'np', False)
    # Obtaining the member 'isinf' of a type (line 81)
    isinf_30072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 7), np_30071, 'isinf')
    # Calling isinf(args, kwargs) (line 81)
    isinf_call_result_30075 = invoke(stypy.reporting.localization.Localization(__file__, 81, 7), isinf_30072, *[a_30073], **kwargs_30074)
    
    
    # Call to isinf(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'b' (line 81)
    b_30078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'b', False)
    # Processing the call keyword arguments (line 81)
    kwargs_30079 = {}
    # Getting the type of 'np' (line 81)
    np_30076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'np', False)
    # Obtaining the member 'isinf' of a type (line 81)
    isinf_30077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 22), np_30076, 'isinf')
    # Calling isinf(args, kwargs) (line 81)
    isinf_call_result_30080 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), isinf_30077, *[b_30078], **kwargs_30079)
    
    # Applying the binary operator 'or' (line 81)
    result_or_keyword_30081 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 7), 'or', isinf_call_result_30075, isinf_call_result_30080)
    
    # Testing the type of an if condition (line 81)
    if_condition_30082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_or_keyword_30081)
    # Assigning a type to the variable 'if_condition_30082' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_30082', if_condition_30082)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 82)
    # Processing the call arguments (line 82)
    str_30084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'str', 'Gaussian quadrature is only available for finite limits.')
    # Processing the call keyword arguments (line 82)
    kwargs_30085 = {}
    # Getting the type of 'ValueError' (line 82)
    ValueError_30083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 82)
    ValueError_call_result_30086 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), ValueError_30083, *[str_30084], **kwargs_30085)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 82, 8), ValueError_call_result_30086, 'raise parameter', BaseException)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 84):
    
    # Assigning a BinOp to a Name (line 84):
    # Getting the type of 'b' (line 84)
    b_30087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'b')
    # Getting the type of 'a' (line 84)
    a_30088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'a')
    # Applying the binary operator '-' (line 84)
    result_sub_30089 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 9), '-', b_30087, a_30088)
    
    # Getting the type of 'x' (line 84)
    x_30090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'x')
    int_30091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 17), 'int')
    # Applying the binary operator '+' (line 84)
    result_add_30092 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), '+', x_30090, int_30091)
    
    # Applying the binary operator '*' (line 84)
    result_mul_30093 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 8), '*', result_sub_30089, result_add_30092)
    
    float_30094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'float')
    # Applying the binary operator 'div' (line 84)
    result_div_30095 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 19), 'div', result_mul_30093, float_30094)
    
    # Getting the type of 'a' (line 84)
    a_30096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'a')
    # Applying the binary operator '+' (line 84)
    result_add_30097 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 8), '+', result_div_30095, a_30096)
    
    # Assigning a type to the variable 'y' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'y', result_add_30097)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_30098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'b' (line 85)
    b_30099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'b')
    # Getting the type of 'a' (line 85)
    a_30100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'a')
    # Applying the binary operator '-' (line 85)
    result_sub_30101 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 12), '-', b_30099, a_30100)
    
    float_30102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'float')
    # Applying the binary operator 'div' (line 85)
    result_div_30103 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), 'div', result_sub_30101, float_30102)
    
    
    # Call to sum(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'w' (line 85)
    w_30106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), 'w', False)
    
    # Call to func(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'y' (line 85)
    y_30108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 37), 'y', False)
    # Getting the type of 'args' (line 85)
    args_30109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'args', False)
    # Processing the call keyword arguments (line 85)
    kwargs_30110 = {}
    # Getting the type of 'func' (line 85)
    func_30107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 32), 'func', False)
    # Calling func(args, kwargs) (line 85)
    func_call_result_30111 = invoke(stypy.reporting.localization.Localization(__file__, 85, 32), func_30107, *[y_30108, args_30109], **kwargs_30110)
    
    # Applying the binary operator '*' (line 85)
    result_mul_30112 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 30), '*', w_30106, func_call_result_30111)
    
    # Processing the call keyword arguments (line 85)
    int_30113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 53), 'int')
    keyword_30114 = int_30113
    kwargs_30115 = {'axis': keyword_30114}
    # Getting the type of 'np' (line 85)
    np_30104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'np', False)
    # Obtaining the member 'sum' of a type (line 85)
    sum_30105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), np_30104, 'sum')
    # Calling sum(args, kwargs) (line 85)
    sum_call_result_30116 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), sum_30105, *[result_mul_30112], **kwargs_30115)
    
    # Applying the binary operator '*' (line 85)
    result_mul_30117 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 21), '*', result_div_30103, sum_call_result_30116)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), tuple_30098, result_mul_30117)
    # Adding element type (line 85)
    # Getting the type of 'None' (line 85)
    None_30118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 58), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), tuple_30098, None_30118)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', tuple_30098)
    
    # ################# End of 'fixed_quad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fixed_quad' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_30119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30119)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fixed_quad'
    return stypy_return_type_30119

# Assigning a type to the variable 'fixed_quad' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'fixed_quad', fixed_quad)

@norecursion
def vectorize1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_30120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    
    # Getting the type of 'False' (line 88)
    False_30121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 39), 'False')
    defaults = [tuple_30120, False_30121]
    # Create a new context for function 'vectorize1'
    module_type_store = module_type_store.open_function_context('vectorize1', 88, 0, False)
    
    # Passed parameters checking function
    vectorize1.stypy_localization = localization
    vectorize1.stypy_type_of_self = None
    vectorize1.stypy_type_store = module_type_store
    vectorize1.stypy_function_name = 'vectorize1'
    vectorize1.stypy_param_names_list = ['func', 'args', 'vec_func']
    vectorize1.stypy_varargs_param_name = None
    vectorize1.stypy_kwargs_param_name = None
    vectorize1.stypy_call_defaults = defaults
    vectorize1.stypy_call_varargs = varargs
    vectorize1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vectorize1', ['func', 'args', 'vec_func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vectorize1', localization, ['func', 'args', 'vec_func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vectorize1(...)' code ##################

    str_30122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'str', 'Vectorize the call to a function.\n\n    This is an internal utility function used by `romberg` and\n    `quadrature` to create a vectorized version of a function.\n\n    If `vec_func` is True, the function `func` is assumed to take vector\n    arguments.\n\n    Parameters\n    ----------\n    func : callable\n        User defined function.\n    args : tuple, optional\n        Extra arguments for the function.\n    vec_func : bool, optional\n        True if the function func takes vector arguments.\n\n    Returns\n    -------\n    vfunc : callable\n        A function that will take a vector argument and return the\n        result.\n\n    ')
    
    # Getting the type of 'vec_func' (line 113)
    vec_func_30123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'vec_func')
    # Testing the type of an if condition (line 113)
    if_condition_30124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), vec_func_30123)
    # Assigning a type to the variable 'if_condition_30124' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_30124', if_condition_30124)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def vfunc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'vfunc'
        module_type_store = module_type_store.open_function_context('vfunc', 114, 8, False)
        
        # Passed parameters checking function
        vfunc.stypy_localization = localization
        vfunc.stypy_type_of_self = None
        vfunc.stypy_type_store = module_type_store
        vfunc.stypy_function_name = 'vfunc'
        vfunc.stypy_param_names_list = ['x']
        vfunc.stypy_varargs_param_name = None
        vfunc.stypy_kwargs_param_name = None
        vfunc.stypy_call_defaults = defaults
        vfunc.stypy_call_varargs = varargs
        vfunc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'vfunc', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'vfunc', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'vfunc(...)' code ##################

        
        # Call to func(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'x' (line 115)
        x_30126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'x', False)
        # Getting the type of 'args' (line 115)
        args_30127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'args', False)
        # Processing the call keyword arguments (line 115)
        kwargs_30128 = {}
        # Getting the type of 'func' (line 115)
        func_30125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'func', False)
        # Calling func(args, kwargs) (line 115)
        func_call_result_30129 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), func_30125, *[x_30126, args_30127], **kwargs_30128)
        
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'stypy_return_type', func_call_result_30129)
        
        # ################# End of 'vfunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'vfunc' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_30130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30130)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'vfunc'
        return stypy_return_type_30130

    # Assigning a type to the variable 'vfunc' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'vfunc', vfunc)
    # SSA branch for the else part of an if statement (line 113)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def vfunc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'vfunc'
        module_type_store = module_type_store.open_function_context('vfunc', 117, 8, False)
        
        # Passed parameters checking function
        vfunc.stypy_localization = localization
        vfunc.stypy_type_of_self = None
        vfunc.stypy_type_store = module_type_store
        vfunc.stypy_function_name = 'vfunc'
        vfunc.stypy_param_names_list = ['x']
        vfunc.stypy_varargs_param_name = None
        vfunc.stypy_kwargs_param_name = None
        vfunc.stypy_call_defaults = defaults
        vfunc.stypy_call_varargs = varargs
        vfunc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'vfunc', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'vfunc', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'vfunc(...)' code ##################

        
        
        # Call to isscalar(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_30133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'x', False)
        # Processing the call keyword arguments (line 118)
        kwargs_30134 = {}
        # Getting the type of 'np' (line 118)
        np_30131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 118)
        isscalar_30132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), np_30131, 'isscalar')
        # Calling isscalar(args, kwargs) (line 118)
        isscalar_call_result_30135 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), isscalar_30132, *[x_30133], **kwargs_30134)
        
        # Testing the type of an if condition (line 118)
        if_condition_30136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), isscalar_call_result_30135)
        # Assigning a type to the variable 'if_condition_30136' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_30136', if_condition_30136)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to func(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'x' (line 119)
        x_30138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'x', False)
        # Getting the type of 'args' (line 119)
        args_30139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 32), 'args', False)
        # Processing the call keyword arguments (line 119)
        kwargs_30140 = {}
        # Getting the type of 'func' (line 119)
        func_30137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'func', False)
        # Calling func(args, kwargs) (line 119)
        func_call_result_30141 = invoke(stypy.reporting.localization.Localization(__file__, 119, 23), func_30137, *[x_30138, args_30139], **kwargs_30140)
        
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'stypy_return_type', func_call_result_30141)
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to asarray(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'x' (line 120)
        x_30144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'x', False)
        # Processing the call keyword arguments (line 120)
        kwargs_30145 = {}
        # Getting the type of 'np' (line 120)
        np_30142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 120)
        asarray_30143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), np_30142, 'asarray')
        # Calling asarray(args, kwargs) (line 120)
        asarray_call_result_30146 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), asarray_30143, *[x_30144], **kwargs_30145)
        
        # Assigning a type to the variable 'x' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'x', asarray_call_result_30146)
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to func(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        int_30148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'int')
        # Getting the type of 'x' (line 122)
        x_30149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___30150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 22), x_30149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_30151 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), getitem___30150, int_30148)
        
        # Getting the type of 'args' (line 122)
        args_30152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'args', False)
        # Processing the call keyword arguments (line 122)
        kwargs_30153 = {}
        # Getting the type of 'func' (line 122)
        func_30147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'func', False)
        # Calling func(args, kwargs) (line 122)
        func_call_result_30154 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), func_30147, *[subscript_call_result_30151, args_30152], **kwargs_30153)
        
        # Assigning a type to the variable 'y0' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'y0', func_call_result_30154)
        
        # Assigning a Call to a Name (line 123):
        
        # Assigning a Call to a Name (line 123):
        
        # Call to len(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'x' (line 123)
        x_30156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'x', False)
        # Processing the call keyword arguments (line 123)
        kwargs_30157 = {}
        # Getting the type of 'len' (line 123)
        len_30155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'len', False)
        # Calling len(args, kwargs) (line 123)
        len_call_result_30158 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), len_30155, *[x_30156], **kwargs_30157)
        
        # Assigning a type to the variable 'n' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'n', len_call_result_30158)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to getattr(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'y0' (line 124)
        y0_30160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'y0', False)
        str_30161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'str', 'dtype')
        
        # Call to type(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'y0' (line 124)
        y0_30163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 46), 'y0', False)
        # Processing the call keyword arguments (line 124)
        kwargs_30164 = {}
        # Getting the type of 'type' (line 124)
        type_30162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'type', False)
        # Calling type(args, kwargs) (line 124)
        type_call_result_30165 = invoke(stypy.reporting.localization.Localization(__file__, 124, 41), type_30162, *[y0_30163], **kwargs_30164)
        
        # Processing the call keyword arguments (line 124)
        kwargs_30166 = {}
        # Getting the type of 'getattr' (line 124)
        getattr_30159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 124)
        getattr_call_result_30167 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), getattr_30159, *[y0_30160, str_30161, type_call_result_30165], **kwargs_30166)
        
        # Assigning a type to the variable 'dtype' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'dtype', getattr_call_result_30167)
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to empty(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_30170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'n' (line 125)
        n_30171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 31), tuple_30170, n_30171)
        
        # Processing the call keyword arguments (line 125)
        # Getting the type of 'dtype' (line 125)
        dtype_30172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'dtype', False)
        keyword_30173 = dtype_30172
        kwargs_30174 = {'dtype': keyword_30173}
        # Getting the type of 'np' (line 125)
        np_30168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'np', False)
        # Obtaining the member 'empty' of a type (line 125)
        empty_30169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 21), np_30168, 'empty')
        # Calling empty(args, kwargs) (line 125)
        empty_call_result_30175 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), empty_30169, *[tuple_30170], **kwargs_30174)
        
        # Assigning a type to the variable 'output' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'output', empty_call_result_30175)
        
        # Assigning a Name to a Subscript (line 126):
        
        # Assigning a Name to a Subscript (line 126):
        # Getting the type of 'y0' (line 126)
        y0_30176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'y0')
        # Getting the type of 'output' (line 126)
        output_30177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'output')
        int_30178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 19), 'int')
        # Storing an element on a container (line 126)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 12), output_30177, (int_30178, y0_30176))
        
        
        # Call to xrange(...): (line 127)
        # Processing the call arguments (line 127)
        int_30180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 28), 'int')
        # Getting the type of 'n' (line 127)
        n_30181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'n', False)
        # Processing the call keyword arguments (line 127)
        kwargs_30182 = {}
        # Getting the type of 'xrange' (line 127)
        xrange_30179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 127)
        xrange_call_result_30183 = invoke(stypy.reporting.localization.Localization(__file__, 127, 21), xrange_30179, *[int_30180, n_30181], **kwargs_30182)
        
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 12), xrange_call_result_30183)
        # Getting the type of the for loop variable (line 127)
        for_loop_var_30184 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 12), xrange_call_result_30183)
        # Assigning a type to the variable 'i' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'i', for_loop_var_30184)
        # SSA begins for a for statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 128):
        
        # Assigning a Call to a Subscript (line 128):
        
        # Call to func(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 128)
        i_30186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'i', False)
        # Getting the type of 'x' (line 128)
        x_30187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___30188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 33), x_30187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_30189 = invoke(stypy.reporting.localization.Localization(__file__, 128, 33), getitem___30188, i_30186)
        
        # Getting the type of 'args' (line 128)
        args_30190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'args', False)
        # Processing the call keyword arguments (line 128)
        kwargs_30191 = {}
        # Getting the type of 'func' (line 128)
        func_30185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'func', False)
        # Calling func(args, kwargs) (line 128)
        func_call_result_30192 = invoke(stypy.reporting.localization.Localization(__file__, 128, 28), func_30185, *[subscript_call_result_30189, args_30190], **kwargs_30191)
        
        # Getting the type of 'output' (line 128)
        output_30193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'output')
        # Getting the type of 'i' (line 128)
        i_30194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'i')
        # Storing an element on a container (line 128)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), output_30193, (i_30194, func_call_result_30192))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'output' (line 129)
        output_30195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'output')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'stypy_return_type', output_30195)
        
        # ################# End of 'vfunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'vfunc' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_30196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'vfunc'
        return stypy_return_type_30196

    # Assigning a type to the variable 'vfunc' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'vfunc', vfunc)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'vfunc' (line 130)
    vfunc_30197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'vfunc')
    # Assigning a type to the variable 'stypy_return_type' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type', vfunc_30197)
    
    # ################# End of 'vectorize1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vectorize1' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_30198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30198)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vectorize1'
    return stypy_return_type_30198

# Assigning a type to the variable 'vectorize1' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'vectorize1', vectorize1)

@norecursion
def quadrature(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_30199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    
    float_30200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 40), 'float')
    float_30201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 54), 'float')
    int_30202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 71), 'int')
    # Getting the type of 'True' (line 134)
    True_30203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'True')
    int_30204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 38), 'int')
    defaults = [tuple_30199, float_30200, float_30201, int_30202, True_30203, int_30204]
    # Create a new context for function 'quadrature'
    module_type_store = module_type_store.open_function_context('quadrature', 133, 0, False)
    
    # Passed parameters checking function
    quadrature.stypy_localization = localization
    quadrature.stypy_type_of_self = None
    quadrature.stypy_type_store = module_type_store
    quadrature.stypy_function_name = 'quadrature'
    quadrature.stypy_param_names_list = ['func', 'a', 'b', 'args', 'tol', 'rtol', 'maxiter', 'vec_func', 'miniter']
    quadrature.stypy_varargs_param_name = None
    quadrature.stypy_kwargs_param_name = None
    quadrature.stypy_call_defaults = defaults
    quadrature.stypy_call_varargs = varargs
    quadrature.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quadrature', ['func', 'a', 'b', 'args', 'tol', 'rtol', 'maxiter', 'vec_func', 'miniter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quadrature', localization, ['func', 'a', 'b', 'args', 'tol', 'rtol', 'maxiter', 'vec_func', 'miniter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quadrature(...)' code ##################

    str_30205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, (-1)), 'str', '\n    Compute a definite integral using fixed-tolerance Gaussian quadrature.\n\n    Integrate `func` from `a` to `b` using Gaussian quadrature\n    with absolute tolerance `tol`.\n\n    Parameters\n    ----------\n    func : function\n        A Python function or method to integrate.\n    a : float\n        Lower limit of integration.\n    b : float\n        Upper limit of integration.\n    args : tuple, optional\n        Extra arguments to pass to function.\n    tol, rtol : float, optional\n        Iteration stops when error between last two iterates is less than\n        `tol` OR the relative change is less than `rtol`.\n    maxiter : int, optional\n        Maximum order of Gaussian quadrature.\n    vec_func : bool, optional\n        True or False if func handles arrays as arguments (is\n        a "vector" function). Default is True.\n    miniter : int, optional\n        Minimum order of Gaussian quadrature.\n\n    Returns\n    -------\n    val : float\n        Gaussian quadrature approximation (within tolerance) to integral.\n    err : float\n        Difference between last two estimates of the integral.\n\n    See also\n    --------\n    romberg: adaptive Romberg quadrature\n    fixed_quad: fixed-order Gaussian quadrature\n    quad: adaptive quadrature using QUADPACK\n    dblquad: double integrals\n    tplquad: triple integrals\n    romb: integrator for sampled data\n    simps: integrator for sampled data\n    cumtrapz: cumulative integration for sampled data\n    ode: ODE integrator\n    odeint: ODE integrator\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 183)
    # Getting the type of 'tuple' (line 183)
    tuple_30206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'tuple')
    # Getting the type of 'args' (line 183)
    args_30207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'args')
    
    (may_be_30208, more_types_in_union_30209) = may_not_be_subtype(tuple_30206, args_30207)

    if may_be_30208:

        if more_types_in_union_30209:
            # Runtime conditional SSA (line 183)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'args', remove_subtype_from_union(args_30207, tuple))
        
        # Assigning a Tuple to a Name (line 184):
        
        # Assigning a Tuple to a Name (line 184):
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_30210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        # Getting the type of 'args' (line 184)
        args_30211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 16), tuple_30210, args_30211)
        
        # Assigning a type to the variable 'args' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'args', tuple_30210)

        if more_types_in_union_30209:
            # SSA join for if statement (line 183)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to vectorize1(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'func' (line 185)
    func_30213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'func', False)
    # Getting the type of 'args' (line 185)
    args_30214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'args', False)
    # Processing the call keyword arguments (line 185)
    # Getting the type of 'vec_func' (line 185)
    vec_func_30215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 44), 'vec_func', False)
    keyword_30216 = vec_func_30215
    kwargs_30217 = {'vec_func': keyword_30216}
    # Getting the type of 'vectorize1' (line 185)
    vectorize1_30212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'vectorize1', False)
    # Calling vectorize1(args, kwargs) (line 185)
    vectorize1_call_result_30218 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), vectorize1_30212, *[func_30213, args_30214], **kwargs_30217)
    
    # Assigning a type to the variable 'vfunc' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'vfunc', vectorize1_call_result_30218)
    
    # Assigning a Attribute to a Name (line 186):
    
    # Assigning a Attribute to a Name (line 186):
    # Getting the type of 'np' (line 186)
    np_30219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 10), 'np')
    # Obtaining the member 'inf' of a type (line 186)
    inf_30220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 10), np_30219, 'inf')
    # Assigning a type to the variable 'val' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'val', inf_30220)
    
    # Assigning a Attribute to a Name (line 187):
    
    # Assigning a Attribute to a Name (line 187):
    # Getting the type of 'np' (line 187)
    np_30221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 10), 'np')
    # Obtaining the member 'inf' of a type (line 187)
    inf_30222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 10), np_30221, 'inf')
    # Assigning a type to the variable 'err' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'err', inf_30222)
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to max(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'miniter' (line 188)
    miniter_30224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'miniter', False)
    int_30225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 26), 'int')
    # Applying the binary operator '+' (line 188)
    result_add_30226 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 18), '+', miniter_30224, int_30225)
    
    # Getting the type of 'maxiter' (line 188)
    maxiter_30227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'maxiter', False)
    # Processing the call keyword arguments (line 188)
    kwargs_30228 = {}
    # Getting the type of 'max' (line 188)
    max_30223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 14), 'max', False)
    # Calling max(args, kwargs) (line 188)
    max_call_result_30229 = invoke(stypy.reporting.localization.Localization(__file__, 188, 14), max_30223, *[result_add_30226, maxiter_30227], **kwargs_30228)
    
    # Assigning a type to the variable 'maxiter' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'maxiter', max_call_result_30229)
    
    
    # Call to xrange(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'miniter' (line 189)
    miniter_30231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'miniter', False)
    # Getting the type of 'maxiter' (line 189)
    maxiter_30232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'maxiter', False)
    int_30233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 37), 'int')
    # Applying the binary operator '+' (line 189)
    result_add_30234 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 29), '+', maxiter_30232, int_30233)
    
    # Processing the call keyword arguments (line 189)
    kwargs_30235 = {}
    # Getting the type of 'xrange' (line 189)
    xrange_30230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 189)
    xrange_call_result_30236 = invoke(stypy.reporting.localization.Localization(__file__, 189, 13), xrange_30230, *[miniter_30231, result_add_30234], **kwargs_30235)
    
    # Testing the type of a for loop iterable (line 189)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 189, 4), xrange_call_result_30236)
    # Getting the type of the for loop variable (line 189)
    for_loop_var_30237 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 189, 4), xrange_call_result_30236)
    # Assigning a type to the variable 'n' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'n', for_loop_var_30237)
    # SSA begins for a for statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 190):
    
    # Assigning a Subscript to a Name (line 190):
    
    # Obtaining the type of the subscript
    int_30238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 48), 'int')
    
    # Call to fixed_quad(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'vfunc' (line 190)
    vfunc_30240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'vfunc', False)
    # Getting the type of 'a' (line 190)
    a_30241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 35), 'a', False)
    # Getting the type of 'b' (line 190)
    b_30242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 38), 'b', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 190)
    tuple_30243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 190)
    
    # Getting the type of 'n' (line 190)
    n_30244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 45), 'n', False)
    # Processing the call keyword arguments (line 190)
    kwargs_30245 = {}
    # Getting the type of 'fixed_quad' (line 190)
    fixed_quad_30239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'fixed_quad', False)
    # Calling fixed_quad(args, kwargs) (line 190)
    fixed_quad_call_result_30246 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), fixed_quad_30239, *[vfunc_30240, a_30241, b_30242, tuple_30243, n_30244], **kwargs_30245)
    
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___30247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 17), fixed_quad_call_result_30246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_30248 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), getitem___30247, int_30238)
    
    # Assigning a type to the variable 'newval' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'newval', subscript_call_result_30248)
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to abs(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'newval' (line 191)
    newval_30250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 18), 'newval', False)
    # Getting the type of 'val' (line 191)
    val_30251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'val', False)
    # Applying the binary operator '-' (line 191)
    result_sub_30252 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 18), '-', newval_30250, val_30251)
    
    # Processing the call keyword arguments (line 191)
    kwargs_30253 = {}
    # Getting the type of 'abs' (line 191)
    abs_30249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'abs', False)
    # Calling abs(args, kwargs) (line 191)
    abs_call_result_30254 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), abs_30249, *[result_sub_30252], **kwargs_30253)
    
    # Assigning a type to the variable 'err' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'err', abs_call_result_30254)
    
    # Assigning a Name to a Name (line 192):
    
    # Assigning a Name to a Name (line 192):
    # Getting the type of 'newval' (line 192)
    newval_30255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 14), 'newval')
    # Assigning a type to the variable 'val' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'val', newval_30255)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'err' (line 194)
    err_30256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'err')
    # Getting the type of 'tol' (line 194)
    tol_30257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'tol')
    # Applying the binary operator '<' (line 194)
    result_lt_30258 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), '<', err_30256, tol_30257)
    
    
    # Getting the type of 'err' (line 194)
    err_30259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'err')
    # Getting the type of 'rtol' (line 194)
    rtol_30260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'rtol')
    
    # Call to abs(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'val' (line 194)
    val_30262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 39), 'val', False)
    # Processing the call keyword arguments (line 194)
    kwargs_30263 = {}
    # Getting the type of 'abs' (line 194)
    abs_30261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 35), 'abs', False)
    # Calling abs(args, kwargs) (line 194)
    abs_call_result_30264 = invoke(stypy.reporting.localization.Localization(__file__, 194, 35), abs_30261, *[val_30262], **kwargs_30263)
    
    # Applying the binary operator '*' (line 194)
    result_mul_30265 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 30), '*', rtol_30260, abs_call_result_30264)
    
    # Applying the binary operator '<' (line 194)
    result_lt_30266 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 24), '<', err_30259, result_mul_30265)
    
    # Applying the binary operator 'or' (line 194)
    result_or_keyword_30267 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), 'or', result_lt_30258, result_lt_30266)
    
    # Testing the type of an if condition (line 194)
    if_condition_30268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_or_keyword_30267)
    # Assigning a type to the variable 'if_condition_30268' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_30268', if_condition_30268)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of a for statement (line 189)
    module_type_store.open_ssa_branch('for loop else')
    
    # Call to warn(...): (line 197)
    # Processing the call arguments (line 197)
    str_30271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 12), 'str', 'maxiter (%d) exceeded. Latest difference = %e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_30272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    # Getting the type of 'maxiter' (line 198)
    maxiter_30273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 63), 'maxiter', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 63), tuple_30272, maxiter_30273)
    # Adding element type (line 198)
    # Getting the type of 'err' (line 198)
    err_30274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 72), 'err', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 63), tuple_30272, err_30274)
    
    # Applying the binary operator '%' (line 198)
    result_mod_30275 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 12), '%', str_30271, tuple_30272)
    
    # Getting the type of 'AccuracyWarning' (line 199)
    AccuracyWarning_30276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'AccuracyWarning', False)
    # Processing the call keyword arguments (line 197)
    kwargs_30277 = {}
    # Getting the type of 'warnings' (line 197)
    warnings_30269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 197)
    warn_30270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), warnings_30269, 'warn')
    # Calling warn(args, kwargs) (line 197)
    warn_call_result_30278 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), warn_30270, *[result_mod_30275, AccuracyWarning_30276], **kwargs_30277)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_30279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    # Getting the type of 'val' (line 200)
    val_30280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 11), tuple_30279, val_30280)
    # Adding element type (line 200)
    # Getting the type of 'err' (line 200)
    err_30281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'err')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 11), tuple_30279, err_30281)
    
    # Assigning a type to the variable 'stypy_return_type' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type', tuple_30279)
    
    # ################# End of 'quadrature(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quadrature' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_30282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quadrature'
    return stypy_return_type_30282

# Assigning a type to the variable 'quadrature' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'quadrature', quadrature)

@norecursion
def tupleset(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tupleset'
    module_type_store = module_type_store.open_function_context('tupleset', 203, 0, False)
    
    # Passed parameters checking function
    tupleset.stypy_localization = localization
    tupleset.stypy_type_of_self = None
    tupleset.stypy_type_store = module_type_store
    tupleset.stypy_function_name = 'tupleset'
    tupleset.stypy_param_names_list = ['t', 'i', 'value']
    tupleset.stypy_varargs_param_name = None
    tupleset.stypy_kwargs_param_name = None
    tupleset.stypy_call_defaults = defaults
    tupleset.stypy_call_varargs = varargs
    tupleset.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tupleset', ['t', 'i', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tupleset', localization, ['t', 'i', 'value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tupleset(...)' code ##################

    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to list(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 't' (line 204)
    t_30284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 13), 't', False)
    # Processing the call keyword arguments (line 204)
    kwargs_30285 = {}
    # Getting the type of 'list' (line 204)
    list_30283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'list', False)
    # Calling list(args, kwargs) (line 204)
    list_call_result_30286 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), list_30283, *[t_30284], **kwargs_30285)
    
    # Assigning a type to the variable 'l' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'l', list_call_result_30286)
    
    # Assigning a Name to a Subscript (line 205):
    
    # Assigning a Name to a Subscript (line 205):
    # Getting the type of 'value' (line 205)
    value_30287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'value')
    # Getting the type of 'l' (line 205)
    l_30288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'l')
    # Getting the type of 'i' (line 205)
    i_30289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 6), 'i')
    # Storing an element on a container (line 205)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 4), l_30288, (i_30289, value_30287))
    
    # Call to tuple(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'l' (line 206)
    l_30291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'l', False)
    # Processing the call keyword arguments (line 206)
    kwargs_30292 = {}
    # Getting the type of 'tuple' (line 206)
    tuple_30290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 206)
    tuple_call_result_30293 = invoke(stypy.reporting.localization.Localization(__file__, 206, 11), tuple_30290, *[l_30291], **kwargs_30292)
    
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', tuple_call_result_30293)
    
    # ################# End of 'tupleset(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tupleset' in the type store
    # Getting the type of 'stypy_return_type' (line 203)
    stypy_return_type_30294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30294)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tupleset'
    return stypy_return_type_30294

# Assigning a type to the variable 'tupleset' (line 203)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'tupleset', tupleset)

@norecursion
def cumtrapz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 209)
    None_30295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 18), 'None')
    float_30296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 27), 'float')
    int_30297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 37), 'int')
    # Getting the type of 'None' (line 209)
    None_30298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 49), 'None')
    defaults = [None_30295, float_30296, int_30297, None_30298]
    # Create a new context for function 'cumtrapz'
    module_type_store = module_type_store.open_function_context('cumtrapz', 209, 0, False)
    
    # Passed parameters checking function
    cumtrapz.stypy_localization = localization
    cumtrapz.stypy_type_of_self = None
    cumtrapz.stypy_type_store = module_type_store
    cumtrapz.stypy_function_name = 'cumtrapz'
    cumtrapz.stypy_param_names_list = ['y', 'x', 'dx', 'axis', 'initial']
    cumtrapz.stypy_varargs_param_name = None
    cumtrapz.stypy_kwargs_param_name = None
    cumtrapz.stypy_call_defaults = defaults
    cumtrapz.stypy_call_varargs = varargs
    cumtrapz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cumtrapz', ['y', 'x', 'dx', 'axis', 'initial'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cumtrapz', localization, ['y', 'x', 'dx', 'axis', 'initial'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cumtrapz(...)' code ##################

    str_30299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, (-1)), 'str', "\n    Cumulatively integrate y(x) using the composite trapezoidal rule.\n\n    Parameters\n    ----------\n    y : array_like\n        Values to integrate.\n    x : array_like, optional\n        The coordinate to integrate along.  If None (default), use spacing `dx`\n        between consecutive elements in `y`.\n    dx : float, optional\n        Spacing between elements of `y`.  Only used if `x` is None.\n    axis : int, optional\n        Specifies the axis to cumulate.  Default is -1 (last axis).\n    initial : scalar, optional\n        If given, uses this value as the first value in the returned result.\n        Typically this value should be 0.  Default is None, which means no\n        value at ``x[0]`` is returned and `res` has one element less than `y`\n        along the axis of integration.\n\n    Returns\n    -------\n    res : ndarray\n        The result of cumulative integration of `y` along `axis`.\n        If `initial` is None, the shape is such that the axis of integration\n        has one less value than `y`.  If `initial` is given, the shape is equal\n        to that of `y`.\n\n    See Also\n    --------\n    numpy.cumsum, numpy.cumprod\n    quad: adaptive quadrature using QUADPACK\n    romberg: adaptive Romberg quadrature\n    quadrature: adaptive Gaussian quadrature\n    fixed_quad: fixed-order Gaussian quadrature\n    dblquad: double integrals\n    tplquad: triple integrals\n    romb: integrators for sampled data\n    ode: ODE integrators\n    odeint: ODE integrators\n\n    Examples\n    --------\n    >>> from scipy import integrate\n    >>> import matplotlib.pyplot as plt\n\n    >>> x = np.linspace(-2, 2, num=20)\n    >>> y = x\n    >>> y_int = integrate.cumtrapz(y, x, initial=0)\n    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')\n    >>> plt.show()\n\n    ")
    
    # Assigning a Call to a Name (line 263):
    
    # Assigning a Call to a Name (line 263):
    
    # Call to asarray(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'y' (line 263)
    y_30302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'y', False)
    # Processing the call keyword arguments (line 263)
    kwargs_30303 = {}
    # Getting the type of 'np' (line 263)
    np_30300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 263)
    asarray_30301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), np_30300, 'asarray')
    # Calling asarray(args, kwargs) (line 263)
    asarray_call_result_30304 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), asarray_30301, *[y_30302], **kwargs_30303)
    
    # Assigning a type to the variable 'y' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'y', asarray_call_result_30304)
    
    # Type idiom detected: calculating its left and rigth part (line 264)
    # Getting the type of 'x' (line 264)
    x_30305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 7), 'x')
    # Getting the type of 'None' (line 264)
    None_30306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'None')
    
    (may_be_30307, more_types_in_union_30308) = may_be_none(x_30305, None_30306)

    if may_be_30307:

        if more_types_in_union_30308:
            # Runtime conditional SSA (line 264)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 265):
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'dx' (line 265)
        dx_30309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'dx')
        # Assigning a type to the variable 'd' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'd', dx_30309)

        if more_types_in_union_30308:
            # Runtime conditional SSA for else branch (line 264)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_30307) or more_types_in_union_30308):
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to asarray(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'x' (line 267)
        x_30312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'x', False)
        # Processing the call keyword arguments (line 267)
        kwargs_30313 = {}
        # Getting the type of 'np' (line 267)
        np_30310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 267)
        asarray_30311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), np_30310, 'asarray')
        # Calling asarray(args, kwargs) (line 267)
        asarray_call_result_30314 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), asarray_30311, *[x_30312], **kwargs_30313)
        
        # Assigning a type to the variable 'x' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'x', asarray_call_result_30314)
        
        
        # Getting the type of 'x' (line 268)
        x_30315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'x')
        # Obtaining the member 'ndim' of a type (line 268)
        ndim_30316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 11), x_30315, 'ndim')
        int_30317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 21), 'int')
        # Applying the binary operator '==' (line 268)
        result_eq_30318 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 11), '==', ndim_30316, int_30317)
        
        # Testing the type of an if condition (line 268)
        if_condition_30319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), result_eq_30318)
        # Assigning a type to the variable 'if_condition_30319' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_30319', if_condition_30319)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to diff(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'x' (line 269)
        x_30322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'x', False)
        # Processing the call keyword arguments (line 269)
        kwargs_30323 = {}
        # Getting the type of 'np' (line 269)
        np_30320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'np', False)
        # Obtaining the member 'diff' of a type (line 269)
        diff_30321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), np_30320, 'diff')
        # Calling diff(args, kwargs) (line 269)
        diff_call_result_30324 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), diff_30321, *[x_30322], **kwargs_30323)
        
        # Assigning a type to the variable 'd' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'd', diff_call_result_30324)
        
        # Assigning a BinOp to a Name (line 271):
        
        # Assigning a BinOp to a Name (line 271):
        
        # Obtaining an instance of the builtin type 'list' (line 271)
        list_30325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 271)
        # Adding element type (line 271)
        int_30326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 20), list_30325, int_30326)
        
        # Getting the type of 'y' (line 271)
        y_30327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'y')
        # Obtaining the member 'ndim' of a type (line 271)
        ndim_30328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 26), y_30327, 'ndim')
        # Applying the binary operator '*' (line 271)
        result_mul_30329 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 20), '*', list_30325, ndim_30328)
        
        # Assigning a type to the variable 'shape' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'shape', result_mul_30329)
        
        # Assigning a Num to a Subscript (line 272):
        
        # Assigning a Num to a Subscript (line 272):
        int_30330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'int')
        # Getting the type of 'shape' (line 272)
        shape_30331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'shape')
        # Getting the type of 'axis' (line 272)
        axis_30332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 18), 'axis')
        # Storing an element on a container (line 272)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 12), shape_30331, (axis_30332, int_30330))
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to reshape(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'shape' (line 273)
        shape_30335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 26), 'shape', False)
        # Processing the call keyword arguments (line 273)
        kwargs_30336 = {}
        # Getting the type of 'd' (line 273)
        d_30333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'd', False)
        # Obtaining the member 'reshape' of a type (line 273)
        reshape_30334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), d_30333, 'reshape')
        # Calling reshape(args, kwargs) (line 273)
        reshape_call_result_30337 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), reshape_30334, *[shape_30335], **kwargs_30336)
        
        # Assigning a type to the variable 'd' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'd', reshape_call_result_30337)
        # SSA branch for the else part of an if statement (line 268)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'x' (line 274)
        x_30339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 17), 'x', False)
        # Obtaining the member 'shape' of a type (line 274)
        shape_30340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 17), x_30339, 'shape')
        # Processing the call keyword arguments (line 274)
        kwargs_30341 = {}
        # Getting the type of 'len' (line 274)
        len_30338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 13), 'len', False)
        # Calling len(args, kwargs) (line 274)
        len_call_result_30342 = invoke(stypy.reporting.localization.Localization(__file__, 274, 13), len_30338, *[shape_30340], **kwargs_30341)
        
        
        # Call to len(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'y' (line 274)
        y_30344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 33), 'y', False)
        # Obtaining the member 'shape' of a type (line 274)
        shape_30345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 33), y_30344, 'shape')
        # Processing the call keyword arguments (line 274)
        kwargs_30346 = {}
        # Getting the type of 'len' (line 274)
        len_30343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'len', False)
        # Calling len(args, kwargs) (line 274)
        len_call_result_30347 = invoke(stypy.reporting.localization.Localization(__file__, 274, 29), len_30343, *[shape_30345], **kwargs_30346)
        
        # Applying the binary operator '!=' (line 274)
        result_ne_30348 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 13), '!=', len_call_result_30342, len_call_result_30347)
        
        # Testing the type of an if condition (line 274)
        if_condition_30349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 13), result_ne_30348)
        # Assigning a type to the variable 'if_condition_30349' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 13), 'if_condition_30349', if_condition_30349)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 275)
        # Processing the call arguments (line 275)
        str_30351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 29), 'str', 'If given, shape of x must be 1-d or the same as y.')
        # Processing the call keyword arguments (line 275)
        kwargs_30352 = {}
        # Getting the type of 'ValueError' (line 275)
        ValueError_30350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 275)
        ValueError_call_result_30353 = invoke(stypy.reporting.localization.Localization(__file__, 275, 18), ValueError_30350, *[str_30351], **kwargs_30352)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 275, 12), ValueError_call_result_30353, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 274)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to diff(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'x' (line 278)
        x_30356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'x', False)
        # Processing the call keyword arguments (line 278)
        # Getting the type of 'axis' (line 278)
        axis_30357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'axis', False)
        keyword_30358 = axis_30357
        kwargs_30359 = {'axis': keyword_30358}
        # Getting the type of 'np' (line 278)
        np_30354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'np', False)
        # Obtaining the member 'diff' of a type (line 278)
        diff_30355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), np_30354, 'diff')
        # Calling diff(args, kwargs) (line 278)
        diff_call_result_30360 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), diff_30355, *[x_30356], **kwargs_30359)
        
        # Assigning a type to the variable 'd' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'd', diff_call_result_30360)
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 280)
        axis_30361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'axis')
        # Getting the type of 'd' (line 280)
        d_30362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'd')
        # Obtaining the member 'shape' of a type (line 280)
        shape_30363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), d_30362, 'shape')
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___30364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), shape_30363, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_30365 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), getitem___30364, axis_30361)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 280)
        axis_30366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'axis')
        # Getting the type of 'y' (line 280)
        y_30367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'y')
        # Obtaining the member 'shape' of a type (line 280)
        shape_30368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 28), y_30367, 'shape')
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___30369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 28), shape_30368, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_30370 = invoke(stypy.reporting.localization.Localization(__file__, 280, 28), getitem___30369, axis_30366)
        
        int_30371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 44), 'int')
        # Applying the binary operator '-' (line 280)
        result_sub_30372 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 28), '-', subscript_call_result_30370, int_30371)
        
        # Applying the binary operator '!=' (line 280)
        result_ne_30373 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 11), '!=', subscript_call_result_30365, result_sub_30372)
        
        # Testing the type of an if condition (line 280)
        if_condition_30374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), result_ne_30373)
        # Assigning a type to the variable 'if_condition_30374' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_30374', if_condition_30374)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 281)
        # Processing the call arguments (line 281)
        str_30376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 29), 'str', 'If given, length of x along axis must be the same as y.')
        # Processing the call keyword arguments (line 281)
        kwargs_30377 = {}
        # Getting the type of 'ValueError' (line 281)
        ValueError_30375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 281)
        ValueError_call_result_30378 = invoke(stypy.reporting.localization.Localization(__file__, 281, 18), ValueError_30375, *[str_30376], **kwargs_30377)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 281, 12), ValueError_call_result_30378, 'raise parameter', BaseException)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_30307 and more_types_in_union_30308):
            # SSA join for if statement (line 264)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to len(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'y' (line 284)
    y_30380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 13), 'y', False)
    # Obtaining the member 'shape' of a type (line 284)
    shape_30381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 13), y_30380, 'shape')
    # Processing the call keyword arguments (line 284)
    kwargs_30382 = {}
    # Getting the type of 'len' (line 284)
    len_30379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 9), 'len', False)
    # Calling len(args, kwargs) (line 284)
    len_call_result_30383 = invoke(stypy.reporting.localization.Localization(__file__, 284, 9), len_30379, *[shape_30381], **kwargs_30382)
    
    # Assigning a type to the variable 'nd' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'nd', len_call_result_30383)
    
    # Assigning a Call to a Name (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to tupleset(...): (line 285)
    # Processing the call arguments (line 285)
    
    # Obtaining an instance of the builtin type 'tuple' (line 285)
    tuple_30385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 285)
    # Adding element type (line 285)
    
    # Call to slice(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'None' (line 285)
    None_30387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 29), 'None', False)
    # Processing the call keyword arguments (line 285)
    kwargs_30388 = {}
    # Getting the type of 'slice' (line 285)
    slice_30386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 285)
    slice_call_result_30389 = invoke(stypy.reporting.localization.Localization(__file__, 285, 23), slice_30386, *[None_30387], **kwargs_30388)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 23), tuple_30385, slice_call_result_30389)
    
    # Getting the type of 'nd' (line 285)
    nd_30390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 37), 'nd', False)
    # Applying the binary operator '*' (line 285)
    result_mul_30391 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 22), '*', tuple_30385, nd_30390)
    
    # Getting the type of 'axis' (line 285)
    axis_30392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'axis', False)
    
    # Call to slice(...): (line 285)
    # Processing the call arguments (line 285)
    int_30394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 53), 'int')
    # Getting the type of 'None' (line 285)
    None_30395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 56), 'None', False)
    # Processing the call keyword arguments (line 285)
    kwargs_30396 = {}
    # Getting the type of 'slice' (line 285)
    slice_30393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 47), 'slice', False)
    # Calling slice(args, kwargs) (line 285)
    slice_call_result_30397 = invoke(stypy.reporting.localization.Localization(__file__, 285, 47), slice_30393, *[int_30394, None_30395], **kwargs_30396)
    
    # Processing the call keyword arguments (line 285)
    kwargs_30398 = {}
    # Getting the type of 'tupleset' (line 285)
    tupleset_30384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 13), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 285)
    tupleset_call_result_30399 = invoke(stypy.reporting.localization.Localization(__file__, 285, 13), tupleset_30384, *[result_mul_30391, axis_30392, slice_call_result_30397], **kwargs_30398)
    
    # Assigning a type to the variable 'slice1' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'slice1', tupleset_call_result_30399)
    
    # Assigning a Call to a Name (line 286):
    
    # Assigning a Call to a Name (line 286):
    
    # Call to tupleset(...): (line 286)
    # Processing the call arguments (line 286)
    
    # Obtaining an instance of the builtin type 'tuple' (line 286)
    tuple_30401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 286)
    # Adding element type (line 286)
    
    # Call to slice(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'None' (line 286)
    None_30403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 'None', False)
    # Processing the call keyword arguments (line 286)
    kwargs_30404 = {}
    # Getting the type of 'slice' (line 286)
    slice_30402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 286)
    slice_call_result_30405 = invoke(stypy.reporting.localization.Localization(__file__, 286, 23), slice_30402, *[None_30403], **kwargs_30404)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), tuple_30401, slice_call_result_30405)
    
    # Getting the type of 'nd' (line 286)
    nd_30406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'nd', False)
    # Applying the binary operator '*' (line 286)
    result_mul_30407 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 22), '*', tuple_30401, nd_30406)
    
    # Getting the type of 'axis' (line 286)
    axis_30408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 41), 'axis', False)
    
    # Call to slice(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'None' (line 286)
    None_30410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 53), 'None', False)
    int_30411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 59), 'int')
    # Processing the call keyword arguments (line 286)
    kwargs_30412 = {}
    # Getting the type of 'slice' (line 286)
    slice_30409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 47), 'slice', False)
    # Calling slice(args, kwargs) (line 286)
    slice_call_result_30413 = invoke(stypy.reporting.localization.Localization(__file__, 286, 47), slice_30409, *[None_30410, int_30411], **kwargs_30412)
    
    # Processing the call keyword arguments (line 286)
    kwargs_30414 = {}
    # Getting the type of 'tupleset' (line 286)
    tupleset_30400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 286)
    tupleset_call_result_30415 = invoke(stypy.reporting.localization.Localization(__file__, 286, 13), tupleset_30400, *[result_mul_30407, axis_30408, slice_call_result_30413], **kwargs_30414)
    
    # Assigning a type to the variable 'slice2' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'slice2', tupleset_call_result_30415)
    
    # Assigning a Call to a Name (line 287):
    
    # Assigning a Call to a Name (line 287):
    
    # Call to cumsum(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'd' (line 287)
    d_30418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'd', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 287)
    slice1_30419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 27), 'slice1', False)
    # Getting the type of 'y' (line 287)
    y_30420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___30421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), y_30420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_30422 = invoke(stypy.reporting.localization.Localization(__file__, 287, 25), getitem___30421, slice1_30419)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 287)
    slice2_30423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 39), 'slice2', False)
    # Getting the type of 'y' (line 287)
    y_30424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 37), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___30425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 37), y_30424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_30426 = invoke(stypy.reporting.localization.Localization(__file__, 287, 37), getitem___30425, slice2_30423)
    
    # Applying the binary operator '+' (line 287)
    result_add_30427 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 25), '+', subscript_call_result_30422, subscript_call_result_30426)
    
    # Applying the binary operator '*' (line 287)
    result_mul_30428 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 20), '*', d_30418, result_add_30427)
    
    float_30429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 50), 'float')
    # Applying the binary operator 'div' (line 287)
    result_div_30430 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 48), 'div', result_mul_30428, float_30429)
    
    # Processing the call keyword arguments (line 287)
    # Getting the type of 'axis' (line 287)
    axis_30431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'axis', False)
    keyword_30432 = axis_30431
    kwargs_30433 = {'axis': keyword_30432}
    # Getting the type of 'np' (line 287)
    np_30416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 10), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 287)
    cumsum_30417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 10), np_30416, 'cumsum')
    # Calling cumsum(args, kwargs) (line 287)
    cumsum_call_result_30434 = invoke(stypy.reporting.localization.Localization(__file__, 287, 10), cumsum_30417, *[result_div_30430], **kwargs_30433)
    
    # Assigning a type to the variable 'res' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'res', cumsum_call_result_30434)
    
    # Type idiom detected: calculating its left and rigth part (line 289)
    # Getting the type of 'initial' (line 289)
    initial_30435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'initial')
    # Getting the type of 'None' (line 289)
    None_30436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'None')
    
    (may_be_30437, more_types_in_union_30438) = may_not_be_none(initial_30435, None_30436)

    if may_be_30437:

        if more_types_in_union_30438:
            # Runtime conditional SSA (line 289)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to isscalar(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'initial' (line 290)
        initial_30441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 27), 'initial', False)
        # Processing the call keyword arguments (line 290)
        kwargs_30442 = {}
        # Getting the type of 'np' (line 290)
        np_30439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 290)
        isscalar_30440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), np_30439, 'isscalar')
        # Calling isscalar(args, kwargs) (line 290)
        isscalar_call_result_30443 = invoke(stypy.reporting.localization.Localization(__file__, 290, 15), isscalar_30440, *[initial_30441], **kwargs_30442)
        
        # Applying the 'not' unary operator (line 290)
        result_not__30444 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'not', isscalar_call_result_30443)
        
        # Testing the type of an if condition (line 290)
        if_condition_30445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_not__30444)
        # Assigning a type to the variable 'if_condition_30445' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_30445', if_condition_30445)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 291)
        # Processing the call arguments (line 291)
        str_30447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', '`initial` parameter should be a scalar.')
        # Processing the call keyword arguments (line 291)
        kwargs_30448 = {}
        # Getting the type of 'ValueError' (line 291)
        ValueError_30446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 291)
        ValueError_call_result_30449 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), ValueError_30446, *[str_30447], **kwargs_30448)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), ValueError_call_result_30449, 'raise parameter', BaseException)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to list(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'res' (line 293)
        res_30451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'res', False)
        # Obtaining the member 'shape' of a type (line 293)
        shape_30452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), res_30451, 'shape')
        # Processing the call keyword arguments (line 293)
        kwargs_30453 = {}
        # Getting the type of 'list' (line 293)
        list_30450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'list', False)
        # Calling list(args, kwargs) (line 293)
        list_call_result_30454 = invoke(stypy.reporting.localization.Localization(__file__, 293, 16), list_30450, *[shape_30452], **kwargs_30453)
        
        # Assigning a type to the variable 'shape' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'shape', list_call_result_30454)
        
        # Assigning a Num to a Subscript (line 294):
        
        # Assigning a Num to a Subscript (line 294):
        int_30455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'int')
        # Getting the type of 'shape' (line 294)
        shape_30456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'shape')
        # Getting the type of 'axis' (line 294)
        axis_30457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'axis')
        # Storing an element on a container (line 294)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 8), shape_30456, (axis_30457, int_30455))
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to concatenate(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_30460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        
        # Call to ones(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'shape' (line 295)
        shape_30463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'shape', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'res' (line 295)
        res_30464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 51), 'res', False)
        # Obtaining the member 'dtype' of a type (line 295)
        dtype_30465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 51), res_30464, 'dtype')
        keyword_30466 = dtype_30465
        kwargs_30467 = {'dtype': keyword_30466}
        # Getting the type of 'np' (line 295)
        np_30461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'np', False)
        # Obtaining the member 'ones' of a type (line 295)
        ones_30462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 30), np_30461, 'ones')
        # Calling ones(args, kwargs) (line 295)
        ones_call_result_30468 = invoke(stypy.reporting.localization.Localization(__file__, 295, 30), ones_30462, *[shape_30463], **kwargs_30467)
        
        # Getting the type of 'initial' (line 295)
        initial_30469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 64), 'initial', False)
        # Applying the binary operator '*' (line 295)
        result_mul_30470 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 30), '*', ones_call_result_30468, initial_30469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 29), list_30460, result_mul_30470)
        # Adding element type (line 295)
        # Getting the type of 'res' (line 295)
        res_30471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 73), 'res', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 29), list_30460, res_30471)
        
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'axis' (line 296)
        axis_30472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'axis', False)
        keyword_30473 = axis_30472
        kwargs_30474 = {'axis': keyword_30473}
        # Getting the type of 'np' (line 295)
        np_30458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 295)
        concatenate_30459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 14), np_30458, 'concatenate')
        # Calling concatenate(args, kwargs) (line 295)
        concatenate_call_result_30475 = invoke(stypy.reporting.localization.Localization(__file__, 295, 14), concatenate_30459, *[list_30460], **kwargs_30474)
        
        # Assigning a type to the variable 'res' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'res', concatenate_call_result_30475)

        if more_types_in_union_30438:
            # SSA join for if statement (line 289)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'res' (line 298)
    res_30476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type', res_30476)
    
    # ################# End of 'cumtrapz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cumtrapz' in the type store
    # Getting the type of 'stypy_return_type' (line 209)
    stypy_return_type_30477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30477)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cumtrapz'
    return stypy_return_type_30477

# Assigning a type to the variable 'cumtrapz' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'cumtrapz', cumtrapz)

@norecursion
def _basic_simps(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_basic_simps'
    module_type_store = module_type_store.open_function_context('_basic_simps', 301, 0, False)
    
    # Passed parameters checking function
    _basic_simps.stypy_localization = localization
    _basic_simps.stypy_type_of_self = None
    _basic_simps.stypy_type_store = module_type_store
    _basic_simps.stypy_function_name = '_basic_simps'
    _basic_simps.stypy_param_names_list = ['y', 'start', 'stop', 'x', 'dx', 'axis']
    _basic_simps.stypy_varargs_param_name = None
    _basic_simps.stypy_kwargs_param_name = None
    _basic_simps.stypy_call_defaults = defaults
    _basic_simps.stypy_call_varargs = varargs
    _basic_simps.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_basic_simps', ['y', 'start', 'stop', 'x', 'dx', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_basic_simps', localization, ['y', 'start', 'stop', 'x', 'dx', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_basic_simps(...)' code ##################

    
    # Assigning a Call to a Name (line 302):
    
    # Assigning a Call to a Name (line 302):
    
    # Call to len(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'y' (line 302)
    y_30479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 13), 'y', False)
    # Obtaining the member 'shape' of a type (line 302)
    shape_30480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 13), y_30479, 'shape')
    # Processing the call keyword arguments (line 302)
    kwargs_30481 = {}
    # Getting the type of 'len' (line 302)
    len_30478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 9), 'len', False)
    # Calling len(args, kwargs) (line 302)
    len_call_result_30482 = invoke(stypy.reporting.localization.Localization(__file__, 302, 9), len_30478, *[shape_30480], **kwargs_30481)
    
    # Assigning a type to the variable 'nd' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'nd', len_call_result_30482)
    
    # Type idiom detected: calculating its left and rigth part (line 303)
    # Getting the type of 'start' (line 303)
    start_30483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 7), 'start')
    # Getting the type of 'None' (line 303)
    None_30484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'None')
    
    (may_be_30485, more_types_in_union_30486) = may_be_none(start_30483, None_30484)

    if may_be_30485:

        if more_types_in_union_30486:
            # Runtime conditional SSA (line 303)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 304):
        
        # Assigning a Num to a Name (line 304):
        int_30487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 16), 'int')
        # Assigning a type to the variable 'start' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'start', int_30487)

        if more_types_in_union_30486:
            # SSA join for if statement (line 303)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 305):
    
    # Assigning a Num to a Name (line 305):
    int_30488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 11), 'int')
    # Assigning a type to the variable 'step' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'step', int_30488)
    
    # Assigning a BinOp to a Name (line 306):
    
    # Assigning a BinOp to a Name (line 306):
    
    # Obtaining an instance of the builtin type 'tuple' (line 306)
    tuple_30489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 306)
    # Adding element type (line 306)
    
    # Call to slice(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'None' (line 306)
    None_30491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'None', False)
    # Processing the call keyword arguments (line 306)
    kwargs_30492 = {}
    # Getting the type of 'slice' (line 306)
    slice_30490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 306)
    slice_call_result_30493 = invoke(stypy.reporting.localization.Localization(__file__, 306, 17), slice_30490, *[None_30491], **kwargs_30492)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 17), tuple_30489, slice_call_result_30493)
    
    # Getting the type of 'nd' (line 306)
    nd_30494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 31), 'nd')
    # Applying the binary operator '*' (line 306)
    result_mul_30495 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 16), '*', tuple_30489, nd_30494)
    
    # Assigning a type to the variable 'slice_all' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'slice_all', result_mul_30495)
    
    # Assigning a Call to a Name (line 307):
    
    # Assigning a Call to a Name (line 307):
    
    # Call to tupleset(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'slice_all' (line 307)
    slice_all_30497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'slice_all', False)
    # Getting the type of 'axis' (line 307)
    axis_30498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 33), 'axis', False)
    
    # Call to slice(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'start' (line 307)
    start_30500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 45), 'start', False)
    # Getting the type of 'stop' (line 307)
    stop_30501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 52), 'stop', False)
    # Getting the type of 'step' (line 307)
    step_30502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 58), 'step', False)
    # Processing the call keyword arguments (line 307)
    kwargs_30503 = {}
    # Getting the type of 'slice' (line 307)
    slice_30499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'slice', False)
    # Calling slice(args, kwargs) (line 307)
    slice_call_result_30504 = invoke(stypy.reporting.localization.Localization(__file__, 307, 39), slice_30499, *[start_30500, stop_30501, step_30502], **kwargs_30503)
    
    # Processing the call keyword arguments (line 307)
    kwargs_30505 = {}
    # Getting the type of 'tupleset' (line 307)
    tupleset_30496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 307)
    tupleset_call_result_30506 = invoke(stypy.reporting.localization.Localization(__file__, 307, 13), tupleset_30496, *[slice_all_30497, axis_30498, slice_call_result_30504], **kwargs_30505)
    
    # Assigning a type to the variable 'slice0' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'slice0', tupleset_call_result_30506)
    
    # Assigning a Call to a Name (line 308):
    
    # Assigning a Call to a Name (line 308):
    
    # Call to tupleset(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'slice_all' (line 308)
    slice_all_30508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'slice_all', False)
    # Getting the type of 'axis' (line 308)
    axis_30509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 33), 'axis', False)
    
    # Call to slice(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'start' (line 308)
    start_30511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'start', False)
    int_30512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 51), 'int')
    # Applying the binary operator '+' (line 308)
    result_add_30513 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 45), '+', start_30511, int_30512)
    
    # Getting the type of 'stop' (line 308)
    stop_30514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 54), 'stop', False)
    int_30515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 59), 'int')
    # Applying the binary operator '+' (line 308)
    result_add_30516 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 54), '+', stop_30514, int_30515)
    
    # Getting the type of 'step' (line 308)
    step_30517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 62), 'step', False)
    # Processing the call keyword arguments (line 308)
    kwargs_30518 = {}
    # Getting the type of 'slice' (line 308)
    slice_30510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'slice', False)
    # Calling slice(args, kwargs) (line 308)
    slice_call_result_30519 = invoke(stypy.reporting.localization.Localization(__file__, 308, 39), slice_30510, *[result_add_30513, result_add_30516, step_30517], **kwargs_30518)
    
    # Processing the call keyword arguments (line 308)
    kwargs_30520 = {}
    # Getting the type of 'tupleset' (line 308)
    tupleset_30507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 13), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 308)
    tupleset_call_result_30521 = invoke(stypy.reporting.localization.Localization(__file__, 308, 13), tupleset_30507, *[slice_all_30508, axis_30509, slice_call_result_30519], **kwargs_30520)
    
    # Assigning a type to the variable 'slice1' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'slice1', tupleset_call_result_30521)
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to tupleset(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'slice_all' (line 309)
    slice_all_30523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 22), 'slice_all', False)
    # Getting the type of 'axis' (line 309)
    axis_30524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'axis', False)
    
    # Call to slice(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'start' (line 309)
    start_30526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 45), 'start', False)
    int_30527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 51), 'int')
    # Applying the binary operator '+' (line 309)
    result_add_30528 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 45), '+', start_30526, int_30527)
    
    # Getting the type of 'stop' (line 309)
    stop_30529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 54), 'stop', False)
    int_30530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 59), 'int')
    # Applying the binary operator '+' (line 309)
    result_add_30531 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 54), '+', stop_30529, int_30530)
    
    # Getting the type of 'step' (line 309)
    step_30532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 62), 'step', False)
    # Processing the call keyword arguments (line 309)
    kwargs_30533 = {}
    # Getting the type of 'slice' (line 309)
    slice_30525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 39), 'slice', False)
    # Calling slice(args, kwargs) (line 309)
    slice_call_result_30534 = invoke(stypy.reporting.localization.Localization(__file__, 309, 39), slice_30525, *[result_add_30528, result_add_30531, step_30532], **kwargs_30533)
    
    # Processing the call keyword arguments (line 309)
    kwargs_30535 = {}
    # Getting the type of 'tupleset' (line 309)
    tupleset_30522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 13), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 309)
    tupleset_call_result_30536 = invoke(stypy.reporting.localization.Localization(__file__, 309, 13), tupleset_30522, *[slice_all_30523, axis_30524, slice_call_result_30534], **kwargs_30535)
    
    # Assigning a type to the variable 'slice2' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'slice2', tupleset_call_result_30536)
    
    # Type idiom detected: calculating its left and rigth part (line 311)
    # Getting the type of 'x' (line 311)
    x_30537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 7), 'x')
    # Getting the type of 'None' (line 311)
    None_30538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'None')
    
    (may_be_30539, more_types_in_union_30540) = may_be_none(x_30537, None_30538)

    if may_be_30539:

        if more_types_in_union_30540:
            # Runtime conditional SSA (line 311)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to sum(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'dx' (line 312)
        dx_30543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'dx', False)
        float_30544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 27), 'float')
        # Applying the binary operator 'div' (line 312)
        result_div_30545 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 24), 'div', dx_30543, float_30544)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice0' (line 312)
        slice0_30546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 36), 'slice0', False)
        # Getting the type of 'y' (line 312)
        y_30547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___30548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 34), y_30547, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_30549 = invoke(stypy.reporting.localization.Localization(__file__, 312, 34), getitem___30548, slice0_30546)
        
        int_30550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 44), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice1' (line 312)
        slice1_30551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'slice1', False)
        # Getting the type of 'y' (line 312)
        y_30552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 46), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___30553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 46), y_30552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_30554 = invoke(stypy.reporting.localization.Localization(__file__, 312, 46), getitem___30553, slice1_30551)
        
        # Applying the binary operator '*' (line 312)
        result_mul_30555 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 44), '*', int_30550, subscript_call_result_30554)
        
        # Applying the binary operator '+' (line 312)
        result_add_30556 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 34), '+', subscript_call_result_30549, result_mul_30555)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice2' (line 312)
        slice2_30557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'slice2', False)
        # Getting the type of 'y' (line 312)
        y_30558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 56), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___30559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 56), y_30558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_30560 = invoke(stypy.reporting.localization.Localization(__file__, 312, 56), getitem___30559, slice2_30557)
        
        # Applying the binary operator '+' (line 312)
        result_add_30561 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 55), '+', result_add_30556, subscript_call_result_30560)
        
        # Applying the binary operator '*' (line 312)
        result_mul_30562 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 31), '*', result_div_30545, result_add_30561)
        
        # Processing the call keyword arguments (line 312)
        # Getting the type of 'axis' (line 313)
        axis_30563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), 'axis', False)
        keyword_30564 = axis_30563
        kwargs_30565 = {'axis': keyword_30564}
        # Getting the type of 'np' (line 312)
        np_30541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'np', False)
        # Obtaining the member 'sum' of a type (line 312)
        sum_30542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 17), np_30541, 'sum')
        # Calling sum(args, kwargs) (line 312)
        sum_call_result_30566 = invoke(stypy.reporting.localization.Localization(__file__, 312, 17), sum_30542, *[result_mul_30562], **kwargs_30565)
        
        # Assigning a type to the variable 'result' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'result', sum_call_result_30566)

        if more_types_in_union_30540:
            # Runtime conditional SSA for else branch (line 311)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_30539) or more_types_in_union_30540):
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to diff(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'x' (line 317)
        x_30569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'x', False)
        # Processing the call keyword arguments (line 317)
        # Getting the type of 'axis' (line 317)
        axis_30570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 28), 'axis', False)
        keyword_30571 = axis_30570
        kwargs_30572 = {'axis': keyword_30571}
        # Getting the type of 'np' (line 317)
        np_30567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'np', False)
        # Obtaining the member 'diff' of a type (line 317)
        diff_30568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), np_30567, 'diff')
        # Calling diff(args, kwargs) (line 317)
        diff_call_result_30573 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), diff_30568, *[x_30569], **kwargs_30572)
        
        # Assigning a type to the variable 'h' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'h', diff_call_result_30573)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to tupleset(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'slice_all' (line 318)
        slice_all_30575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'slice_all', False)
        # Getting the type of 'axis' (line 318)
        axis_30576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 34), 'axis', False)
        
        # Call to slice(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'start' (line 318)
        start_30578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 46), 'start', False)
        # Getting the type of 'stop' (line 318)
        stop_30579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 53), 'stop', False)
        # Getting the type of 'step' (line 318)
        step_30580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 59), 'step', False)
        # Processing the call keyword arguments (line 318)
        kwargs_30581 = {}
        # Getting the type of 'slice' (line 318)
        slice_30577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 40), 'slice', False)
        # Calling slice(args, kwargs) (line 318)
        slice_call_result_30582 = invoke(stypy.reporting.localization.Localization(__file__, 318, 40), slice_30577, *[start_30578, stop_30579, step_30580], **kwargs_30581)
        
        # Processing the call keyword arguments (line 318)
        kwargs_30583 = {}
        # Getting the type of 'tupleset' (line 318)
        tupleset_30574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 14), 'tupleset', False)
        # Calling tupleset(args, kwargs) (line 318)
        tupleset_call_result_30584 = invoke(stypy.reporting.localization.Localization(__file__, 318, 14), tupleset_30574, *[slice_all_30575, axis_30576, slice_call_result_30582], **kwargs_30583)
        
        # Assigning a type to the variable 'sl0' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'sl0', tupleset_call_result_30584)
        
        # Assigning a Call to a Name (line 319):
        
        # Assigning a Call to a Name (line 319):
        
        # Call to tupleset(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'slice_all' (line 319)
        slice_all_30586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 23), 'slice_all', False)
        # Getting the type of 'axis' (line 319)
        axis_30587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 34), 'axis', False)
        
        # Call to slice(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'start' (line 319)
        start_30589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 46), 'start', False)
        int_30590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 52), 'int')
        # Applying the binary operator '+' (line 319)
        result_add_30591 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 46), '+', start_30589, int_30590)
        
        # Getting the type of 'stop' (line 319)
        stop_30592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 55), 'stop', False)
        int_30593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 60), 'int')
        # Applying the binary operator '+' (line 319)
        result_add_30594 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 55), '+', stop_30592, int_30593)
        
        # Getting the type of 'step' (line 319)
        step_30595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 63), 'step', False)
        # Processing the call keyword arguments (line 319)
        kwargs_30596 = {}
        # Getting the type of 'slice' (line 319)
        slice_30588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 40), 'slice', False)
        # Calling slice(args, kwargs) (line 319)
        slice_call_result_30597 = invoke(stypy.reporting.localization.Localization(__file__, 319, 40), slice_30588, *[result_add_30591, result_add_30594, step_30595], **kwargs_30596)
        
        # Processing the call keyword arguments (line 319)
        kwargs_30598 = {}
        # Getting the type of 'tupleset' (line 319)
        tupleset_30585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 14), 'tupleset', False)
        # Calling tupleset(args, kwargs) (line 319)
        tupleset_call_result_30599 = invoke(stypy.reporting.localization.Localization(__file__, 319, 14), tupleset_30585, *[slice_all_30586, axis_30587, slice_call_result_30597], **kwargs_30598)
        
        # Assigning a type to the variable 'sl1' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'sl1', tupleset_call_result_30599)
        
        # Assigning a Subscript to a Name (line 320):
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        # Getting the type of 'sl0' (line 320)
        sl0_30600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'sl0')
        # Getting the type of 'h' (line 320)
        h_30601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'h')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___30602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 13), h_30601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_30603 = invoke(stypy.reporting.localization.Localization(__file__, 320, 13), getitem___30602, sl0_30600)
        
        # Assigning a type to the variable 'h0' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'h0', subscript_call_result_30603)
        
        # Assigning a Subscript to a Name (line 321):
        
        # Assigning a Subscript to a Name (line 321):
        
        # Obtaining the type of the subscript
        # Getting the type of 'sl1' (line 321)
        sl1_30604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 15), 'sl1')
        # Getting the type of 'h' (line 321)
        h_30605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 13), 'h')
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___30606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 13), h_30605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_30607 = invoke(stypy.reporting.localization.Localization(__file__, 321, 13), getitem___30606, sl1_30604)
        
        # Assigning a type to the variable 'h1' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'h1', subscript_call_result_30607)
        
        # Assigning a BinOp to a Name (line 322):
        
        # Assigning a BinOp to a Name (line 322):
        # Getting the type of 'h0' (line 322)
        h0_30608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'h0')
        # Getting the type of 'h1' (line 322)
        h1_30609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'h1')
        # Applying the binary operator '+' (line 322)
        result_add_30610 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), '+', h0_30608, h1_30609)
        
        # Assigning a type to the variable 'hsum' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'hsum', result_add_30610)
        
        # Assigning a BinOp to a Name (line 323):
        
        # Assigning a BinOp to a Name (line 323):
        # Getting the type of 'h0' (line 323)
        h0_30611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'h0')
        # Getting the type of 'h1' (line 323)
        h1_30612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'h1')
        # Applying the binary operator '*' (line 323)
        result_mul_30613 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 16), '*', h0_30611, h1_30612)
        
        # Assigning a type to the variable 'hprod' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'hprod', result_mul_30613)
        
        # Assigning a BinOp to a Name (line 324):
        
        # Assigning a BinOp to a Name (line 324):
        # Getting the type of 'h0' (line 324)
        h0_30614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'h0')
        # Getting the type of 'h1' (line 324)
        h1_30615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'h1')
        # Applying the binary operator 'div' (line 324)
        result_div_30616 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 18), 'div', h0_30614, h1_30615)
        
        # Assigning a type to the variable 'h0divh1' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'h0divh1', result_div_30616)
        
        # Assigning a BinOp to a Name (line 325):
        
        # Assigning a BinOp to a Name (line 325):
        # Getting the type of 'hsum' (line 325)
        hsum_30617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'hsum')
        float_30618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 19), 'float')
        # Applying the binary operator 'div' (line 325)
        result_div_30619 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 14), 'div', hsum_30617, float_30618)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice0' (line 325)
        slice0_30620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 28), 'slice0')
        # Getting the type of 'y' (line 325)
        y_30621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 26), 'y')
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___30622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 26), y_30621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_30623 = invoke(stypy.reporting.localization.Localization(__file__, 325, 26), getitem___30622, slice0_30620)
        
        int_30624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 37), 'int')
        float_30625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 39), 'float')
        # Getting the type of 'h0divh1' (line 325)
        h0divh1_30626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 43), 'h0divh1')
        # Applying the binary operator 'div' (line 325)
        result_div_30627 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 39), 'div', float_30625, h0divh1_30626)
        
        # Applying the binary operator '-' (line 325)
        result_sub_30628 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 37), '-', int_30624, result_div_30627)
        
        # Applying the binary operator '*' (line 325)
        result_mul_30629 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 26), '*', subscript_call_result_30623, result_sub_30628)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice1' (line 326)
        slice1_30630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 28), 'slice1')
        # Getting the type of 'y' (line 326)
        y_30631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'y')
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___30632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 26), y_30631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_30633 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), getitem___30632, slice1_30630)
        
        # Getting the type of 'hsum' (line 326)
        hsum_30634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'hsum')
        # Applying the binary operator '*' (line 326)
        result_mul_30635 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 26), '*', subscript_call_result_30633, hsum_30634)
        
        # Getting the type of 'hsum' (line 326)
        hsum_30636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'hsum')
        # Applying the binary operator '*' (line 326)
        result_mul_30637 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 40), '*', result_mul_30635, hsum_30636)
        
        # Getting the type of 'hprod' (line 326)
        hprod_30638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 46), 'hprod')
        # Applying the binary operator 'div' (line 326)
        result_div_30639 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 45), 'div', result_mul_30637, hprod_30638)
        
        # Applying the binary operator '+' (line 325)
        result_add_30640 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 26), '+', result_mul_30629, result_div_30639)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice2' (line 327)
        slice2_30641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 28), 'slice2')
        # Getting the type of 'y' (line 327)
        y_30642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'y')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___30643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 26), y_30642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_30644 = invoke(stypy.reporting.localization.Localization(__file__, 327, 26), getitem___30643, slice2_30641)
        
        int_30645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 37), 'int')
        # Getting the type of 'h0divh1' (line 327)
        h0divh1_30646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 39), 'h0divh1')
        # Applying the binary operator '-' (line 327)
        result_sub_30647 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 37), '-', int_30645, h0divh1_30646)
        
        # Applying the binary operator '*' (line 327)
        result_mul_30648 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 26), '*', subscript_call_result_30644, result_sub_30647)
        
        # Applying the binary operator '+' (line 326)
        result_add_30649 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 52), '+', result_add_30640, result_mul_30648)
        
        # Applying the binary operator '*' (line 325)
        result_mul_30650 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 23), '*', result_div_30619, result_add_30649)
        
        # Assigning a type to the variable 'tmp' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'tmp', result_mul_30650)
        
        # Assigning a Call to a Name (line 328):
        
        # Assigning a Call to a Name (line 328):
        
        # Call to sum(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'tmp' (line 328)
        tmp_30653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'tmp', False)
        # Processing the call keyword arguments (line 328)
        # Getting the type of 'axis' (line 328)
        axis_30654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'axis', False)
        keyword_30655 = axis_30654
        kwargs_30656 = {'axis': keyword_30655}
        # Getting the type of 'np' (line 328)
        np_30651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'np', False)
        # Obtaining the member 'sum' of a type (line 328)
        sum_30652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 17), np_30651, 'sum')
        # Calling sum(args, kwargs) (line 328)
        sum_call_result_30657 = invoke(stypy.reporting.localization.Localization(__file__, 328, 17), sum_30652, *[tmp_30653], **kwargs_30656)
        
        # Assigning a type to the variable 'result' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'result', sum_call_result_30657)

        if (may_be_30539 and more_types_in_union_30540):
            # SSA join for if statement (line 311)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 329)
    result_30658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type', result_30658)
    
    # ################# End of '_basic_simps(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_basic_simps' in the type store
    # Getting the type of 'stypy_return_type' (line 301)
    stypy_return_type_30659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_basic_simps'
    return stypy_return_type_30659

# Assigning a type to the variable '_basic_simps' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), '_basic_simps', _basic_simps)

@norecursion
def simps(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 332)
    None_30660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'None')
    int_30661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 24), 'int')
    int_30662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 32), 'int')
    str_30663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 41), 'str', 'avg')
    defaults = [None_30660, int_30661, int_30662, str_30663]
    # Create a new context for function 'simps'
    module_type_store = module_type_store.open_function_context('simps', 332, 0, False)
    
    # Passed parameters checking function
    simps.stypy_localization = localization
    simps.stypy_type_of_self = None
    simps.stypy_type_store = module_type_store
    simps.stypy_function_name = 'simps'
    simps.stypy_param_names_list = ['y', 'x', 'dx', 'axis', 'even']
    simps.stypy_varargs_param_name = None
    simps.stypy_kwargs_param_name = None
    simps.stypy_call_defaults = defaults
    simps.stypy_call_varargs = varargs
    simps.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simps', ['y', 'x', 'dx', 'axis', 'even'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simps', localization, ['y', 'x', 'dx', 'axis', 'even'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simps(...)' code ##################

    str_30664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, (-1)), 'str', "\n    Integrate y(x) using samples along the given axis and the composite\n    Simpson's rule.  If x is None, spacing of dx is assumed.\n\n    If there are an even number of samples, N, then there are an odd\n    number of intervals (N-1), but Simpson's rule requires an even number\n    of intervals.  The parameter 'even' controls how this is handled.\n\n    Parameters\n    ----------\n    y : array_like\n        Array to be integrated.\n    x : array_like, optional\n        If given, the points at which `y` is sampled.\n    dx : int, optional\n        Spacing of integration points along axis of `y`. Only used when\n        `x` is None. Default is 1.\n    axis : int, optional\n        Axis along which to integrate. Default is the last axis.\n    even : str {'avg', 'first', 'last'}, optional\n        'avg' : Average two results:1) use the first N-2 intervals with\n                  a trapezoidal rule on the last interval and 2) use the last\n                  N-2 intervals with a trapezoidal rule on the first interval.\n\n        'first' : Use Simpson's rule for the first N-2 intervals with\n                a trapezoidal rule on the last interval.\n\n        'last' : Use Simpson's rule for the last N-2 intervals with a\n               trapezoidal rule on the first interval.\n\n    See Also\n    --------\n    quad: adaptive quadrature using QUADPACK\n    romberg: adaptive Romberg quadrature\n    quadrature: adaptive Gaussian quadrature\n    fixed_quad: fixed-order Gaussian quadrature\n    dblquad: double integrals\n    tplquad: triple integrals\n    romb: integrators for sampled data\n    cumtrapz: cumulative integration for sampled data\n    ode: ODE integrators\n    odeint: ODE integrators\n\n    Notes\n    -----\n    For an odd number of samples that are equally spaced the result is\n    exact if the function is a polynomial of order 3 or less.  If\n    the samples are not equally spaced, then the result is exact only\n    if the function is a polynomial of order 2 or less.\n\n    ")
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to asarray(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'y' (line 384)
    y_30667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'y', False)
    # Processing the call keyword arguments (line 384)
    kwargs_30668 = {}
    # Getting the type of 'np' (line 384)
    np_30665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 384)
    asarray_30666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), np_30665, 'asarray')
    # Calling asarray(args, kwargs) (line 384)
    asarray_call_result_30669 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), asarray_30666, *[y_30667], **kwargs_30668)
    
    # Assigning a type to the variable 'y' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'y', asarray_call_result_30669)
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to len(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'y' (line 385)
    y_30671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'y', False)
    # Obtaining the member 'shape' of a type (line 385)
    shape_30672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 13), y_30671, 'shape')
    # Processing the call keyword arguments (line 385)
    kwargs_30673 = {}
    # Getting the type of 'len' (line 385)
    len_30670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 9), 'len', False)
    # Calling len(args, kwargs) (line 385)
    len_call_result_30674 = invoke(stypy.reporting.localization.Localization(__file__, 385, 9), len_30670, *[shape_30672], **kwargs_30673)
    
    # Assigning a type to the variable 'nd' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'nd', len_call_result_30674)
    
    # Assigning a Subscript to a Name (line 386):
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 386)
    axis_30675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'axis')
    # Getting the type of 'y' (line 386)
    y_30676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'y')
    # Obtaining the member 'shape' of a type (line 386)
    shape_30677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), y_30676, 'shape')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___30678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), shape_30677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_30679 = invoke(stypy.reporting.localization.Localization(__file__, 386, 8), getitem___30678, axis_30675)
    
    # Assigning a type to the variable 'N' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'N', subscript_call_result_30679)
    
    # Assigning a Name to a Name (line 387):
    
    # Assigning a Name to a Name (line 387):
    # Getting the type of 'dx' (line 387)
    dx_30680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 14), 'dx')
    # Assigning a type to the variable 'last_dx' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'last_dx', dx_30680)
    
    # Assigning a Name to a Name (line 388):
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'dx' (line 388)
    dx_30681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'dx')
    # Assigning a type to the variable 'first_dx' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'first_dx', dx_30681)
    
    # Assigning a Num to a Name (line 389):
    
    # Assigning a Num to a Name (line 389):
    int_30682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 18), 'int')
    # Assigning a type to the variable 'returnshape' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'returnshape', int_30682)
    
    # Type idiom detected: calculating its left and rigth part (line 390)
    # Getting the type of 'x' (line 390)
    x_30683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'x')
    # Getting the type of 'None' (line 390)
    None_30684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'None')
    
    (may_be_30685, more_types_in_union_30686) = may_not_be_none(x_30683, None_30684)

    if may_be_30685:

        if more_types_in_union_30686:
            # Runtime conditional SSA (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 391):
        
        # Assigning a Call to a Name (line 391):
        
        # Call to asarray(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'x' (line 391)
        x_30689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'x', False)
        # Processing the call keyword arguments (line 391)
        kwargs_30690 = {}
        # Getting the type of 'np' (line 391)
        np_30687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 391)
        asarray_30688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), np_30687, 'asarray')
        # Calling asarray(args, kwargs) (line 391)
        asarray_call_result_30691 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), asarray_30688, *[x_30689], **kwargs_30690)
        
        # Assigning a type to the variable 'x' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'x', asarray_call_result_30691)
        
        
        
        # Call to len(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'x' (line 392)
        x_30693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'x', False)
        # Obtaining the member 'shape' of a type (line 392)
        shape_30694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), x_30693, 'shape')
        # Processing the call keyword arguments (line 392)
        kwargs_30695 = {}
        # Getting the type of 'len' (line 392)
        len_30692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'len', False)
        # Calling len(args, kwargs) (line 392)
        len_call_result_30696 = invoke(stypy.reporting.localization.Localization(__file__, 392, 11), len_30692, *[shape_30694], **kwargs_30695)
        
        int_30697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 27), 'int')
        # Applying the binary operator '==' (line 392)
        result_eq_30698 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), '==', len_call_result_30696, int_30697)
        
        # Testing the type of an if condition (line 392)
        if_condition_30699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_eq_30698)
        # Assigning a type to the variable 'if_condition_30699' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_30699', if_condition_30699)
        # SSA begins for if statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 393):
        
        # Assigning a BinOp to a Name (line 393):
        
        # Obtaining an instance of the builtin type 'list' (line 393)
        list_30700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 393)
        # Adding element type (line 393)
        int_30701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 21), list_30700, int_30701)
        
        # Getting the type of 'nd' (line 393)
        nd_30702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 27), 'nd')
        # Applying the binary operator '*' (line 393)
        result_mul_30703 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 21), '*', list_30700, nd_30702)
        
        # Assigning a type to the variable 'shapex' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'shapex', result_mul_30703)
        
        # Assigning a Subscript to a Subscript (line 394):
        
        # Assigning a Subscript to a Subscript (line 394):
        
        # Obtaining the type of the subscript
        int_30704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 35), 'int')
        # Getting the type of 'x' (line 394)
        x_30705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 27), 'x')
        # Obtaining the member 'shape' of a type (line 394)
        shape_30706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 27), x_30705, 'shape')
        # Obtaining the member '__getitem__' of a type (line 394)
        getitem___30707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 27), shape_30706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 394)
        subscript_call_result_30708 = invoke(stypy.reporting.localization.Localization(__file__, 394, 27), getitem___30707, int_30704)
        
        # Getting the type of 'shapex' (line 394)
        shapex_30709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'shapex')
        # Getting the type of 'axis' (line 394)
        axis_30710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'axis')
        # Storing an element on a container (line 394)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 12), shapex_30709, (axis_30710, subscript_call_result_30708))
        
        # Assigning a Attribute to a Name (line 395):
        
        # Assigning a Attribute to a Name (line 395):
        # Getting the type of 'x' (line 395)
        x_30711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 24), 'x')
        # Obtaining the member 'shape' of a type (line 395)
        shape_30712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 24), x_30711, 'shape')
        # Assigning a type to the variable 'saveshape' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'saveshape', shape_30712)
        
        # Assigning a Num to a Name (line 396):
        
        # Assigning a Num to a Name (line 396):
        int_30713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 26), 'int')
        # Assigning a type to the variable 'returnshape' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'returnshape', int_30713)
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to reshape(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Call to tuple(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'shapex' (line 397)
        shapex_30717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 32), 'shapex', False)
        # Processing the call keyword arguments (line 397)
        kwargs_30718 = {}
        # Getting the type of 'tuple' (line 397)
        tuple_30716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'tuple', False)
        # Calling tuple(args, kwargs) (line 397)
        tuple_call_result_30719 = invoke(stypy.reporting.localization.Localization(__file__, 397, 26), tuple_30716, *[shapex_30717], **kwargs_30718)
        
        # Processing the call keyword arguments (line 397)
        kwargs_30720 = {}
        # Getting the type of 'x' (line 397)
        x_30714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'x', False)
        # Obtaining the member 'reshape' of a type (line 397)
        reshape_30715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 16), x_30714, 'reshape')
        # Calling reshape(args, kwargs) (line 397)
        reshape_call_result_30721 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), reshape_30715, *[tuple_call_result_30719], **kwargs_30720)
        
        # Assigning a type to the variable 'x' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'x', reshape_call_result_30721)
        # SSA branch for the else part of an if statement (line 392)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'x' (line 398)
        x_30723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'x', False)
        # Obtaining the member 'shape' of a type (line 398)
        shape_30724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 17), x_30723, 'shape')
        # Processing the call keyword arguments (line 398)
        kwargs_30725 = {}
        # Getting the type of 'len' (line 398)
        len_30722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'len', False)
        # Calling len(args, kwargs) (line 398)
        len_call_result_30726 = invoke(stypy.reporting.localization.Localization(__file__, 398, 13), len_30722, *[shape_30724], **kwargs_30725)
        
        
        # Call to len(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'y' (line 398)
        y_30728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 33), 'y', False)
        # Obtaining the member 'shape' of a type (line 398)
        shape_30729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 33), y_30728, 'shape')
        # Processing the call keyword arguments (line 398)
        kwargs_30730 = {}
        # Getting the type of 'len' (line 398)
        len_30727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'len', False)
        # Calling len(args, kwargs) (line 398)
        len_call_result_30731 = invoke(stypy.reporting.localization.Localization(__file__, 398, 29), len_30727, *[shape_30729], **kwargs_30730)
        
        # Applying the binary operator '!=' (line 398)
        result_ne_30732 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 13), '!=', len_call_result_30726, len_call_result_30731)
        
        # Testing the type of an if condition (line 398)
        if_condition_30733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 13), result_ne_30732)
        # Assigning a type to the variable 'if_condition_30733' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'if_condition_30733', if_condition_30733)
        # SSA begins for if statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 399)
        # Processing the call arguments (line 399)
        str_30735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 29), 'str', 'If given, shape of x must be 1-d or the same as y.')
        # Processing the call keyword arguments (line 399)
        kwargs_30736 = {}
        # Getting the type of 'ValueError' (line 399)
        ValueError_30734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 399)
        ValueError_call_result_30737 = invoke(stypy.reporting.localization.Localization(__file__, 399, 18), ValueError_30734, *[str_30735], **kwargs_30736)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 399, 12), ValueError_call_result_30737, 'raise parameter', BaseException)
        # SSA join for if statement (line 398)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 392)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 401)
        axis_30738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 19), 'axis')
        # Getting the type of 'x' (line 401)
        x_30739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'x')
        # Obtaining the member 'shape' of a type (line 401)
        shape_30740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 11), x_30739, 'shape')
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___30741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 11), shape_30740, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 401)
        subscript_call_result_30742 = invoke(stypy.reporting.localization.Localization(__file__, 401, 11), getitem___30741, axis_30738)
        
        # Getting the type of 'N' (line 401)
        N_30743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'N')
        # Applying the binary operator '!=' (line 401)
        result_ne_30744 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 11), '!=', subscript_call_result_30742, N_30743)
        
        # Testing the type of an if condition (line 401)
        if_condition_30745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 8), result_ne_30744)
        # Assigning a type to the variable 'if_condition_30745' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'if_condition_30745', if_condition_30745)
        # SSA begins for if statement (line 401)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 402)
        # Processing the call arguments (line 402)
        str_30747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 29), 'str', 'If given, length of x along axis must be the same as y.')
        # Processing the call keyword arguments (line 402)
        kwargs_30748 = {}
        # Getting the type of 'ValueError' (line 402)
        ValueError_30746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 402)
        ValueError_call_result_30749 = invoke(stypy.reporting.localization.Localization(__file__, 402, 18), ValueError_30746, *[str_30747], **kwargs_30748)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 402, 12), ValueError_call_result_30749, 'raise parameter', BaseException)
        # SSA join for if statement (line 401)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_30686:
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'N' (line 404)
    N_30750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 7), 'N')
    int_30751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 11), 'int')
    # Applying the binary operator '%' (line 404)
    result_mod_30752 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 7), '%', N_30750, int_30751)
    
    int_30753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 16), 'int')
    # Applying the binary operator '==' (line 404)
    result_eq_30754 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 7), '==', result_mod_30752, int_30753)
    
    # Testing the type of an if condition (line 404)
    if_condition_30755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 4), result_eq_30754)
    # Assigning a type to the variable 'if_condition_30755' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'if_condition_30755', if_condition_30755)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 405):
    
    # Assigning a Num to a Name (line 405):
    float_30756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 14), 'float')
    # Assigning a type to the variable 'val' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'val', float_30756)
    
    # Assigning a Num to a Name (line 406):
    
    # Assigning a Num to a Name (line 406):
    float_30757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 17), 'float')
    # Assigning a type to the variable 'result' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'result', float_30757)
    
    # Assigning a BinOp to a Name (line 407):
    
    # Assigning a BinOp to a Name (line 407):
    
    # Obtaining an instance of the builtin type 'tuple' (line 407)
    tuple_30758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 407)
    # Adding element type (line 407)
    
    # Call to slice(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'None' (line 407)
    None_30760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 24), 'None', False)
    # Processing the call keyword arguments (line 407)
    kwargs_30761 = {}
    # Getting the type of 'slice' (line 407)
    slice_30759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'slice', False)
    # Calling slice(args, kwargs) (line 407)
    slice_call_result_30762 = invoke(stypy.reporting.localization.Localization(__file__, 407, 18), slice_30759, *[None_30760], **kwargs_30761)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 18), tuple_30758, slice_call_result_30762)
    
    # Getting the type of 'nd' (line 407)
    nd_30763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 32), 'nd')
    # Applying the binary operator '*' (line 407)
    result_mul_30764 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 17), '*', tuple_30758, nd_30763)
    
    # Assigning a type to the variable 'slice1' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'slice1', result_mul_30764)
    
    # Assigning a BinOp to a Name (line 408):
    
    # Assigning a BinOp to a Name (line 408):
    
    # Obtaining an instance of the builtin type 'tuple' (line 408)
    tuple_30765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 408)
    # Adding element type (line 408)
    
    # Call to slice(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'None' (line 408)
    None_30767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'None', False)
    # Processing the call keyword arguments (line 408)
    kwargs_30768 = {}
    # Getting the type of 'slice' (line 408)
    slice_30766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'slice', False)
    # Calling slice(args, kwargs) (line 408)
    slice_call_result_30769 = invoke(stypy.reporting.localization.Localization(__file__, 408, 18), slice_30766, *[None_30767], **kwargs_30768)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 18), tuple_30765, slice_call_result_30769)
    
    # Getting the type of 'nd' (line 408)
    nd_30770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 32), 'nd')
    # Applying the binary operator '*' (line 408)
    result_mul_30771 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 17), '*', tuple_30765, nd_30770)
    
    # Assigning a type to the variable 'slice2' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'slice2', result_mul_30771)
    
    
    # Getting the type of 'even' (line 409)
    even_30772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'even')
    
    # Obtaining an instance of the builtin type 'list' (line 409)
    list_30773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 409)
    # Adding element type (line 409)
    str_30774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 24), 'str', 'avg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 23), list_30773, str_30774)
    # Adding element type (line 409)
    str_30775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 31), 'str', 'last')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 23), list_30773, str_30775)
    # Adding element type (line 409)
    str_30776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 39), 'str', 'first')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 23), list_30773, str_30776)
    
    # Applying the binary operator 'notin' (line 409)
    result_contains_30777 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 11), 'notin', even_30772, list_30773)
    
    # Testing the type of an if condition (line 409)
    if_condition_30778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), result_contains_30777)
    # Assigning a type to the variable 'if_condition_30778' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_30778', if_condition_30778)
    # SSA begins for if statement (line 409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 410)
    # Processing the call arguments (line 410)
    str_30780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 29), 'str', "Parameter 'even' must be 'avg', 'last', or 'first'.")
    # Processing the call keyword arguments (line 410)
    kwargs_30781 = {}
    # Getting the type of 'ValueError' (line 410)
    ValueError_30779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 410)
    ValueError_call_result_30782 = invoke(stypy.reporting.localization.Localization(__file__, 410, 18), ValueError_30779, *[str_30780], **kwargs_30781)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 410, 12), ValueError_call_result_30782, 'raise parameter', BaseException)
    # SSA join for if statement (line 409)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'even' (line 413)
    even_30783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'even')
    
    # Obtaining an instance of the builtin type 'list' (line 413)
    list_30784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 413)
    # Adding element type (line 413)
    str_30785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 20), 'str', 'avg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 19), list_30784, str_30785)
    # Adding element type (line 413)
    str_30786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 27), 'str', 'first')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 19), list_30784, str_30786)
    
    # Applying the binary operator 'in' (line 413)
    result_contains_30787 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 11), 'in', even_30783, list_30784)
    
    # Testing the type of an if condition (line 413)
    if_condition_30788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 8), result_contains_30787)
    # Assigning a type to the variable 'if_condition_30788' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'if_condition_30788', if_condition_30788)
    # SSA begins for if statement (line 413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to tupleset(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'slice1' (line 414)
    slice1_30790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 30), 'slice1', False)
    # Getting the type of 'axis' (line 414)
    axis_30791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 38), 'axis', False)
    int_30792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 44), 'int')
    # Processing the call keyword arguments (line 414)
    kwargs_30793 = {}
    # Getting the type of 'tupleset' (line 414)
    tupleset_30789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 21), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 414)
    tupleset_call_result_30794 = invoke(stypy.reporting.localization.Localization(__file__, 414, 21), tupleset_30789, *[slice1_30790, axis_30791, int_30792], **kwargs_30793)
    
    # Assigning a type to the variable 'slice1' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'slice1', tupleset_call_result_30794)
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to tupleset(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 'slice2' (line 415)
    slice2_30796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 30), 'slice2', False)
    # Getting the type of 'axis' (line 415)
    axis_30797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 38), 'axis', False)
    int_30798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 44), 'int')
    # Processing the call keyword arguments (line 415)
    kwargs_30799 = {}
    # Getting the type of 'tupleset' (line 415)
    tupleset_30795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 21), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 415)
    tupleset_call_result_30800 = invoke(stypy.reporting.localization.Localization(__file__, 415, 21), tupleset_30795, *[slice2_30796, axis_30797, int_30798], **kwargs_30799)
    
    # Assigning a type to the variable 'slice2' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'slice2', tupleset_call_result_30800)
    
    # Type idiom detected: calculating its left and rigth part (line 416)
    # Getting the type of 'x' (line 416)
    x_30801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'x')
    # Getting the type of 'None' (line 416)
    None_30802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'None')
    
    (may_be_30803, more_types_in_union_30804) = may_not_be_none(x_30801, None_30802)

    if may_be_30803:

        if more_types_in_union_30804:
            # Runtime conditional SSA (line 416)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 417):
        
        # Assigning a BinOp to a Name (line 417):
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice1' (line 417)
        slice1_30805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 28), 'slice1')
        # Getting the type of 'x' (line 417)
        x_30806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___30807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 26), x_30806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_30808 = invoke(stypy.reporting.localization.Localization(__file__, 417, 26), getitem___30807, slice1_30805)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice2' (line 417)
        slice2_30809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 40), 'slice2')
        # Getting the type of 'x' (line 417)
        x_30810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 38), 'x')
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___30811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 38), x_30810, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_30812 = invoke(stypy.reporting.localization.Localization(__file__, 417, 38), getitem___30811, slice2_30809)
        
        # Applying the binary operator '-' (line 417)
        result_sub_30813 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 26), '-', subscript_call_result_30808, subscript_call_result_30812)
        
        # Assigning a type to the variable 'last_dx' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'last_dx', result_sub_30813)

        if more_types_in_union_30804:
            # SSA join for if statement (line 416)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'val' (line 418)
    val_30814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'val')
    float_30815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 19), 'float')
    # Getting the type of 'last_dx' (line 418)
    last_dx_30816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'last_dx')
    # Applying the binary operator '*' (line 418)
    result_mul_30817 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 19), '*', float_30815, last_dx_30816)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 418)
    slice1_30818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 34), 'slice1')
    # Getting the type of 'y' (line 418)
    y_30819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 32), 'y')
    # Obtaining the member '__getitem__' of a type (line 418)
    getitem___30820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 32), y_30819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 418)
    subscript_call_result_30821 = invoke(stypy.reporting.localization.Localization(__file__, 418, 32), getitem___30820, slice1_30818)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 418)
    slice2_30822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 44), 'slice2')
    # Getting the type of 'y' (line 418)
    y_30823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 42), 'y')
    # Obtaining the member '__getitem__' of a type (line 418)
    getitem___30824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 42), y_30823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 418)
    subscript_call_result_30825 = invoke(stypy.reporting.localization.Localization(__file__, 418, 42), getitem___30824, slice2_30822)
    
    # Applying the binary operator '+' (line 418)
    result_add_30826 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 32), '+', subscript_call_result_30821, subscript_call_result_30825)
    
    # Applying the binary operator '*' (line 418)
    result_mul_30827 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 30), '*', result_mul_30817, result_add_30826)
    
    # Applying the binary operator '+=' (line 418)
    result_iadd_30828 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 12), '+=', val_30814, result_mul_30827)
    # Assigning a type to the variable 'val' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'val', result_iadd_30828)
    
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to _basic_simps(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'y' (line 419)
    y_30830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 34), 'y', False)
    int_30831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 37), 'int')
    # Getting the type of 'N' (line 419)
    N_30832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'N', False)
    int_30833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 42), 'int')
    # Applying the binary operator '-' (line 419)
    result_sub_30834 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 40), '-', N_30832, int_30833)
    
    # Getting the type of 'x' (line 419)
    x_30835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 45), 'x', False)
    # Getting the type of 'dx' (line 419)
    dx_30836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 48), 'dx', False)
    # Getting the type of 'axis' (line 419)
    axis_30837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 52), 'axis', False)
    # Processing the call keyword arguments (line 419)
    kwargs_30838 = {}
    # Getting the type of '_basic_simps' (line 419)
    _basic_simps_30829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 21), '_basic_simps', False)
    # Calling _basic_simps(args, kwargs) (line 419)
    _basic_simps_call_result_30839 = invoke(stypy.reporting.localization.Localization(__file__, 419, 21), _basic_simps_30829, *[y_30830, int_30831, result_sub_30834, x_30835, dx_30836, axis_30837], **kwargs_30838)
    
    # Assigning a type to the variable 'result' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'result', _basic_simps_call_result_30839)
    # SSA join for if statement (line 413)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'even' (line 421)
    even_30840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'even')
    
    # Obtaining an instance of the builtin type 'list' (line 421)
    list_30841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 421)
    # Adding element type (line 421)
    str_30842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 20), 'str', 'avg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 19), list_30841, str_30842)
    # Adding element type (line 421)
    str_30843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 27), 'str', 'last')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 19), list_30841, str_30843)
    
    # Applying the binary operator 'in' (line 421)
    result_contains_30844 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 11), 'in', even_30840, list_30841)
    
    # Testing the type of an if condition (line 421)
    if_condition_30845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 8), result_contains_30844)
    # Assigning a type to the variable 'if_condition_30845' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'if_condition_30845', if_condition_30845)
    # SSA begins for if statement (line 421)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to tupleset(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'slice1' (line 422)
    slice1_30847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 30), 'slice1', False)
    # Getting the type of 'axis' (line 422)
    axis_30848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 38), 'axis', False)
    int_30849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 44), 'int')
    # Processing the call keyword arguments (line 422)
    kwargs_30850 = {}
    # Getting the type of 'tupleset' (line 422)
    tupleset_30846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 21), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 422)
    tupleset_call_result_30851 = invoke(stypy.reporting.localization.Localization(__file__, 422, 21), tupleset_30846, *[slice1_30847, axis_30848, int_30849], **kwargs_30850)
    
    # Assigning a type to the variable 'slice1' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'slice1', tupleset_call_result_30851)
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to tupleset(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'slice2' (line 423)
    slice2_30853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 30), 'slice2', False)
    # Getting the type of 'axis' (line 423)
    axis_30854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 38), 'axis', False)
    int_30855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 44), 'int')
    # Processing the call keyword arguments (line 423)
    kwargs_30856 = {}
    # Getting the type of 'tupleset' (line 423)
    tupleset_30852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 423)
    tupleset_call_result_30857 = invoke(stypy.reporting.localization.Localization(__file__, 423, 21), tupleset_30852, *[slice2_30853, axis_30854, int_30855], **kwargs_30856)
    
    # Assigning a type to the variable 'slice2' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'slice2', tupleset_call_result_30857)
    
    # Type idiom detected: calculating its left and rigth part (line 424)
    # Getting the type of 'x' (line 424)
    x_30858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'x')
    # Getting the type of 'None' (line 424)
    None_30859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'None')
    
    (may_be_30860, more_types_in_union_30861) = may_not_be_none(x_30858, None_30859)

    if may_be_30860:

        if more_types_in_union_30861:
            # Runtime conditional SSA (line 424)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 425):
        
        # Assigning a BinOp to a Name (line 425):
        
        # Obtaining the type of the subscript
        
        # Call to tuple(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'slice2' (line 425)
        slice2_30863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 35), 'slice2', False)
        # Processing the call keyword arguments (line 425)
        kwargs_30864 = {}
        # Getting the type of 'tuple' (line 425)
        tuple_30862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 29), 'tuple', False)
        # Calling tuple(args, kwargs) (line 425)
        tuple_call_result_30865 = invoke(stypy.reporting.localization.Localization(__file__, 425, 29), tuple_30862, *[slice2_30863], **kwargs_30864)
        
        # Getting the type of 'x' (line 425)
        x_30866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'x')
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___30867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 27), x_30866, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 425)
        subscript_call_result_30868 = invoke(stypy.reporting.localization.Localization(__file__, 425, 27), getitem___30867, tuple_call_result_30865)
        
        
        # Obtaining the type of the subscript
        
        # Call to tuple(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'slice1' (line 425)
        slice1_30870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 54), 'slice1', False)
        # Processing the call keyword arguments (line 425)
        kwargs_30871 = {}
        # Getting the type of 'tuple' (line 425)
        tuple_30869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 48), 'tuple', False)
        # Calling tuple(args, kwargs) (line 425)
        tuple_call_result_30872 = invoke(stypy.reporting.localization.Localization(__file__, 425, 48), tuple_30869, *[slice1_30870], **kwargs_30871)
        
        # Getting the type of 'x' (line 425)
        x_30873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 46), 'x')
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___30874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 46), x_30873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 425)
        subscript_call_result_30875 = invoke(stypy.reporting.localization.Localization(__file__, 425, 46), getitem___30874, tuple_call_result_30872)
        
        # Applying the binary operator '-' (line 425)
        result_sub_30876 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 27), '-', subscript_call_result_30868, subscript_call_result_30875)
        
        # Assigning a type to the variable 'first_dx' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'first_dx', result_sub_30876)

        if more_types_in_union_30861:
            # SSA join for if statement (line 424)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'val' (line 426)
    val_30877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'val')
    float_30878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'float')
    # Getting the type of 'first_dx' (line 426)
    first_dx_30879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 23), 'first_dx')
    # Applying the binary operator '*' (line 426)
    result_mul_30880 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 19), '*', float_30878, first_dx_30879)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice2' (line 426)
    slice2_30881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 35), 'slice2')
    # Getting the type of 'y' (line 426)
    y_30882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 33), 'y')
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___30883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 33), y_30882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_30884 = invoke(stypy.reporting.localization.Localization(__file__, 426, 33), getitem___30883, slice2_30881)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice1' (line 426)
    slice1_30885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 45), 'slice1')
    # Getting the type of 'y' (line 426)
    y_30886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 43), 'y')
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___30887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 43), y_30886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_30888 = invoke(stypy.reporting.localization.Localization(__file__, 426, 43), getitem___30887, slice1_30885)
    
    # Applying the binary operator '+' (line 426)
    result_add_30889 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 33), '+', subscript_call_result_30884, subscript_call_result_30888)
    
    # Applying the binary operator '*' (line 426)
    result_mul_30890 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 31), '*', result_mul_30880, result_add_30889)
    
    # Applying the binary operator '+=' (line 426)
    result_iadd_30891 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 12), '+=', val_30877, result_mul_30890)
    # Assigning a type to the variable 'val' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'val', result_iadd_30891)
    
    
    # Getting the type of 'result' (line 427)
    result_30892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'result')
    
    # Call to _basic_simps(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'y' (line 427)
    y_30894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 35), 'y', False)
    int_30895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 38), 'int')
    # Getting the type of 'N' (line 427)
    N_30896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 41), 'N', False)
    int_30897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 43), 'int')
    # Applying the binary operator '-' (line 427)
    result_sub_30898 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 41), '-', N_30896, int_30897)
    
    # Getting the type of 'x' (line 427)
    x_30899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 46), 'x', False)
    # Getting the type of 'dx' (line 427)
    dx_30900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 49), 'dx', False)
    # Getting the type of 'axis' (line 427)
    axis_30901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 53), 'axis', False)
    # Processing the call keyword arguments (line 427)
    kwargs_30902 = {}
    # Getting the type of '_basic_simps' (line 427)
    _basic_simps_30893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), '_basic_simps', False)
    # Calling _basic_simps(args, kwargs) (line 427)
    _basic_simps_call_result_30903 = invoke(stypy.reporting.localization.Localization(__file__, 427, 22), _basic_simps_30893, *[y_30894, int_30895, result_sub_30898, x_30899, dx_30900, axis_30901], **kwargs_30902)
    
    # Applying the binary operator '+=' (line 427)
    result_iadd_30904 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 12), '+=', result_30892, _basic_simps_call_result_30903)
    # Assigning a type to the variable 'result' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'result', result_iadd_30904)
    
    # SSA join for if statement (line 421)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'even' (line 428)
    even_30905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'even')
    str_30906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 19), 'str', 'avg')
    # Applying the binary operator '==' (line 428)
    result_eq_30907 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), '==', even_30905, str_30906)
    
    # Testing the type of an if condition (line 428)
    if_condition_30908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_eq_30907)
    # Assigning a type to the variable 'if_condition_30908' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_30908', if_condition_30908)
    # SSA begins for if statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'val' (line 429)
    val_30909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'val')
    float_30910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 19), 'float')
    # Applying the binary operator 'div=' (line 429)
    result_div_30911 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 12), 'div=', val_30909, float_30910)
    # Assigning a type to the variable 'val' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'val', result_div_30911)
    
    
    # Getting the type of 'result' (line 430)
    result_30912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'result')
    float_30913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 22), 'float')
    # Applying the binary operator 'div=' (line 430)
    result_div_30914 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 12), 'div=', result_30912, float_30913)
    # Assigning a type to the variable 'result' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'result', result_div_30914)
    
    # SSA join for if statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 431):
    
    # Assigning a BinOp to a Name (line 431):
    # Getting the type of 'result' (line 431)
    result_30915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 17), 'result')
    # Getting the type of 'val' (line 431)
    val_30916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 26), 'val')
    # Applying the binary operator '+' (line 431)
    result_add_30917 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 17), '+', result_30915, val_30916)
    
    # Assigning a type to the variable 'result' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'result', result_add_30917)
    # SSA branch for the else part of an if statement (line 404)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 433):
    
    # Assigning a Call to a Name (line 433):
    
    # Call to _basic_simps(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'y' (line 433)
    y_30919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'y', False)
    int_30920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 33), 'int')
    # Getting the type of 'N' (line 433)
    N_30921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 36), 'N', False)
    int_30922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 38), 'int')
    # Applying the binary operator '-' (line 433)
    result_sub_30923 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 36), '-', N_30921, int_30922)
    
    # Getting the type of 'x' (line 433)
    x_30924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 41), 'x', False)
    # Getting the type of 'dx' (line 433)
    dx_30925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 44), 'dx', False)
    # Getting the type of 'axis' (line 433)
    axis_30926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 48), 'axis', False)
    # Processing the call keyword arguments (line 433)
    kwargs_30927 = {}
    # Getting the type of '_basic_simps' (line 433)
    _basic_simps_30918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), '_basic_simps', False)
    # Calling _basic_simps(args, kwargs) (line 433)
    _basic_simps_call_result_30928 = invoke(stypy.reporting.localization.Localization(__file__, 433, 17), _basic_simps_30918, *[y_30919, int_30920, result_sub_30923, x_30924, dx_30925, axis_30926], **kwargs_30927)
    
    # Assigning a type to the variable 'result' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'result', _basic_simps_call_result_30928)
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'returnshape' (line 434)
    returnshape_30929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 7), 'returnshape')
    # Testing the type of an if condition (line 434)
    if_condition_30930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 4), returnshape_30929)
    # Assigning a type to the variable 'if_condition_30930' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'if_condition_30930', if_condition_30930)
    # SSA begins for if statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to reshape(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'saveshape' (line 435)
    saveshape_30933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 22), 'saveshape', False)
    # Processing the call keyword arguments (line 435)
    kwargs_30934 = {}
    # Getting the type of 'x' (line 435)
    x_30931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'x', False)
    # Obtaining the member 'reshape' of a type (line 435)
    reshape_30932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 12), x_30931, 'reshape')
    # Calling reshape(args, kwargs) (line 435)
    reshape_call_result_30935 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), reshape_30932, *[saveshape_30933], **kwargs_30934)
    
    # Assigning a type to the variable 'x' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'x', reshape_call_result_30935)
    # SSA join for if statement (line 434)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 436)
    result_30936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type', result_30936)
    
    # ################# End of 'simps(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simps' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_30937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30937)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simps'
    return stypy_return_type_30937

# Assigning a type to the variable 'simps' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'simps', simps)

@norecursion
def romb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_30938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 15), 'float')
    int_30939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 25), 'int')
    # Getting the type of 'False' (line 439)
    False_30940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 34), 'False')
    defaults = [float_30938, int_30939, False_30940]
    # Create a new context for function 'romb'
    module_type_store = module_type_store.open_function_context('romb', 439, 0, False)
    
    # Passed parameters checking function
    romb.stypy_localization = localization
    romb.stypy_type_of_self = None
    romb.stypy_type_store = module_type_store
    romb.stypy_function_name = 'romb'
    romb.stypy_param_names_list = ['y', 'dx', 'axis', 'show']
    romb.stypy_varargs_param_name = None
    romb.stypy_kwargs_param_name = None
    romb.stypy_call_defaults = defaults
    romb.stypy_call_varargs = varargs
    romb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'romb', ['y', 'dx', 'axis', 'show'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'romb', localization, ['y', 'dx', 'axis', 'show'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'romb(...)' code ##################

    str_30941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, (-1)), 'str', '\n    Romberg integration using samples of a function.\n\n    Parameters\n    ----------\n    y : array_like\n        A vector of ``2**k + 1`` equally-spaced samples of a function.\n    dx : float, optional\n        The sample spacing. Default is 1.\n    axis : int, optional\n        The axis along which to integrate. Default is -1 (last axis).\n    show : bool, optional\n        When `y` is a single 1-D array, then if this argument is True\n        print the table showing Richardson extrapolation from the\n        samples. Default is False.\n\n    Returns\n    -------\n    romb : ndarray\n        The integrated result for `axis`.\n\n    See also\n    --------\n    quad : adaptive quadrature using QUADPACK\n    romberg : adaptive Romberg quadrature\n    quadrature : adaptive Gaussian quadrature\n    fixed_quad : fixed-order Gaussian quadrature\n    dblquad : double integrals\n    tplquad : triple integrals\n    simps : integrators for sampled data\n    cumtrapz : cumulative integration for sampled data\n    ode : ODE integrators\n    odeint : ODE integrators\n\n    ')
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to asarray(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'y' (line 475)
    y_30944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'y', False)
    # Processing the call keyword arguments (line 475)
    kwargs_30945 = {}
    # Getting the type of 'np' (line 475)
    np_30942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 475)
    asarray_30943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), np_30942, 'asarray')
    # Calling asarray(args, kwargs) (line 475)
    asarray_call_result_30946 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), asarray_30943, *[y_30944], **kwargs_30945)
    
    # Assigning a type to the variable 'y' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'y', asarray_call_result_30946)
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to len(...): (line 476)
    # Processing the call arguments (line 476)
    # Getting the type of 'y' (line 476)
    y_30948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 13), 'y', False)
    # Obtaining the member 'shape' of a type (line 476)
    shape_30949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 13), y_30948, 'shape')
    # Processing the call keyword arguments (line 476)
    kwargs_30950 = {}
    # Getting the type of 'len' (line 476)
    len_30947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 9), 'len', False)
    # Calling len(args, kwargs) (line 476)
    len_call_result_30951 = invoke(stypy.reporting.localization.Localization(__file__, 476, 9), len_30947, *[shape_30949], **kwargs_30950)
    
    # Assigning a type to the variable 'nd' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'nd', len_call_result_30951)
    
    # Assigning a Subscript to a Name (line 477):
    
    # Assigning a Subscript to a Name (line 477):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 477)
    axis_30952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 21), 'axis')
    # Getting the type of 'y' (line 477)
    y_30953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 13), 'y')
    # Obtaining the member 'shape' of a type (line 477)
    shape_30954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 13), y_30953, 'shape')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___30955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 13), shape_30954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_30956 = invoke(stypy.reporting.localization.Localization(__file__, 477, 13), getitem___30955, axis_30952)
    
    # Assigning a type to the variable 'Nsamps' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'Nsamps', subscript_call_result_30956)
    
    # Assigning a BinOp to a Name (line 478):
    
    # Assigning a BinOp to a Name (line 478):
    # Getting the type of 'Nsamps' (line 478)
    Nsamps_30957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 14), 'Nsamps')
    int_30958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 21), 'int')
    # Applying the binary operator '-' (line 478)
    result_sub_30959 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 14), '-', Nsamps_30957, int_30958)
    
    # Assigning a type to the variable 'Ninterv' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'Ninterv', result_sub_30959)
    
    # Assigning a Num to a Name (line 479):
    
    # Assigning a Num to a Name (line 479):
    int_30960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 8), 'int')
    # Assigning a type to the variable 'n' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'n', int_30960)
    
    # Assigning a Num to a Name (line 480):
    
    # Assigning a Num to a Name (line 480):
    int_30961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 8), 'int')
    # Assigning a type to the variable 'k' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'k', int_30961)
    
    
    # Getting the type of 'n' (line 481)
    n_30962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 10), 'n')
    # Getting the type of 'Ninterv' (line 481)
    Ninterv_30963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 14), 'Ninterv')
    # Applying the binary operator '<' (line 481)
    result_lt_30964 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 10), '<', n_30962, Ninterv_30963)
    
    # Testing the type of an if condition (line 481)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 4), result_lt_30964)
    # SSA begins for while statement (line 481)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'n' (line 482)
    n_30965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'n')
    int_30966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 14), 'int')
    # Applying the binary operator '<<=' (line 482)
    result_ilshift_30967 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 8), '<<=', n_30965, int_30966)
    # Assigning a type to the variable 'n' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'n', result_ilshift_30967)
    
    
    # Getting the type of 'k' (line 483)
    k_30968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'k')
    int_30969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 13), 'int')
    # Applying the binary operator '+=' (line 483)
    result_iadd_30970 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 8), '+=', k_30968, int_30969)
    # Assigning a type to the variable 'k' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'k', result_iadd_30970)
    
    # SSA join for while statement (line 481)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 484)
    n_30971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 7), 'n')
    # Getting the type of 'Ninterv' (line 484)
    Ninterv_30972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'Ninterv')
    # Applying the binary operator '!=' (line 484)
    result_ne_30973 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 7), '!=', n_30971, Ninterv_30972)
    
    # Testing the type of an if condition (line 484)
    if_condition_30974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 4), result_ne_30973)
    # Assigning a type to the variable 'if_condition_30974' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'if_condition_30974', if_condition_30974)
    # SSA begins for if statement (line 484)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 485)
    # Processing the call arguments (line 485)
    str_30976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 25), 'str', 'Number of samples must be one plus a non-negative power of 2.')
    # Processing the call keyword arguments (line 485)
    kwargs_30977 = {}
    # Getting the type of 'ValueError' (line 485)
    ValueError_30975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 485)
    ValueError_call_result_30978 = invoke(stypy.reporting.localization.Localization(__file__, 485, 14), ValueError_30975, *[str_30976], **kwargs_30977)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 485, 8), ValueError_call_result_30978, 'raise parameter', BaseException)
    # SSA join for if statement (line 484)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 488):
    
    # Assigning a Dict to a Name (line 488):
    
    # Obtaining an instance of the builtin type 'dict' (line 488)
    dict_30979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 488)
    
    # Assigning a type to the variable 'R' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'R', dict_30979)
    
    # Assigning a BinOp to a Name (line 489):
    
    # Assigning a BinOp to a Name (line 489):
    
    # Obtaining an instance of the builtin type 'tuple' (line 489)
    tuple_30980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 489)
    # Adding element type (line 489)
    
    # Call to slice(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'None' (line 489)
    None_30982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 23), 'None', False)
    # Processing the call keyword arguments (line 489)
    kwargs_30983 = {}
    # Getting the type of 'slice' (line 489)
    slice_30981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 489)
    slice_call_result_30984 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), slice_30981, *[None_30982], **kwargs_30983)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 17), tuple_30980, slice_call_result_30984)
    
    # Getting the type of 'nd' (line 489)
    nd_30985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 33), 'nd')
    # Applying the binary operator '*' (line 489)
    result_mul_30986 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 16), '*', tuple_30980, nd_30985)
    
    # Assigning a type to the variable 'slice_all' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'slice_all', result_mul_30986)
    
    # Assigning a Call to a Name (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Call to tupleset(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'slice_all' (line 490)
    slice_all_30988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'slice_all', False)
    # Getting the type of 'axis' (line 490)
    axis_30989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 33), 'axis', False)
    int_30990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 39), 'int')
    # Processing the call keyword arguments (line 490)
    kwargs_30991 = {}
    # Getting the type of 'tupleset' (line 490)
    tupleset_30987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 13), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 490)
    tupleset_call_result_30992 = invoke(stypy.reporting.localization.Localization(__file__, 490, 13), tupleset_30987, *[slice_all_30988, axis_30989, int_30990], **kwargs_30991)
    
    # Assigning a type to the variable 'slice0' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'slice0', tupleset_call_result_30992)
    
    # Assigning a Call to a Name (line 491):
    
    # Assigning a Call to a Name (line 491):
    
    # Call to tupleset(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'slice_all' (line 491)
    slice_all_30994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 23), 'slice_all', False)
    # Getting the type of 'axis' (line 491)
    axis_30995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 34), 'axis', False)
    int_30996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 40), 'int')
    # Processing the call keyword arguments (line 491)
    kwargs_30997 = {}
    # Getting the type of 'tupleset' (line 491)
    tupleset_30993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 14), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 491)
    tupleset_call_result_30998 = invoke(stypy.reporting.localization.Localization(__file__, 491, 14), tupleset_30993, *[slice_all_30994, axis_30995, int_30996], **kwargs_30997)
    
    # Assigning a type to the variable 'slicem1' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'slicem1', tupleset_call_result_30998)
    
    # Assigning a BinOp to a Name (line 492):
    
    # Assigning a BinOp to a Name (line 492):
    # Getting the type of 'Ninterv' (line 492)
    Ninterv_30999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'Ninterv')
    
    # Call to asarray(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'dx' (line 492)
    dx_31002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 29), 'dx', False)
    # Processing the call keyword arguments (line 492)
    # Getting the type of 'float' (line 492)
    float_31003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 39), 'float', False)
    keyword_31004 = float_31003
    kwargs_31005 = {'dtype': keyword_31004}
    # Getting the type of 'np' (line 492)
    np_31000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 18), 'np', False)
    # Obtaining the member 'asarray' of a type (line 492)
    asarray_31001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 18), np_31000, 'asarray')
    # Calling asarray(args, kwargs) (line 492)
    asarray_call_result_31006 = invoke(stypy.reporting.localization.Localization(__file__, 492, 18), asarray_31001, *[dx_31002], **kwargs_31005)
    
    # Applying the binary operator '*' (line 492)
    result_mul_31007 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 8), '*', Ninterv_30999, asarray_call_result_31006)
    
    # Assigning a type to the variable 'h' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'h', result_mul_31007)
    
    # Assigning a BinOp to a Subscript (line 493):
    
    # Assigning a BinOp to a Subscript (line 493):
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice0' (line 493)
    slice0_31008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 19), 'slice0')
    # Getting the type of 'y' (line 493)
    y_31009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 17), 'y')
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___31010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 17), y_31009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_31011 = invoke(stypy.reporting.localization.Localization(__file__, 493, 17), getitem___31010, slice0_31008)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'slicem1' (line 493)
    slicem1_31012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 31), 'slicem1')
    # Getting the type of 'y' (line 493)
    y_31013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 29), 'y')
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___31014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 29), y_31013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_31015 = invoke(stypy.reporting.localization.Localization(__file__, 493, 29), getitem___31014, slicem1_31012)
    
    # Applying the binary operator '+' (line 493)
    result_add_31016 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 17), '+', subscript_call_result_31011, subscript_call_result_31015)
    
    float_31017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 41), 'float')
    # Applying the binary operator 'div' (line 493)
    result_div_31018 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 16), 'div', result_add_31016, float_31017)
    
    # Getting the type of 'h' (line 493)
    h_31019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 45), 'h')
    # Applying the binary operator '*' (line 493)
    result_mul_31020 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 44), '*', result_div_31018, h_31019)
    
    # Getting the type of 'R' (line 493)
    R_31021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'R')
    
    # Obtaining an instance of the builtin type 'tuple' (line 493)
    tuple_31022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 7), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 493)
    # Adding element type (line 493)
    int_31023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 7), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 7), tuple_31022, int_31023)
    # Adding element type (line 493)
    int_31024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 7), tuple_31022, int_31024)
    
    # Storing an element on a container (line 493)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 4), R_31021, (tuple_31022, result_mul_31020))
    
    # Assigning a Name to a Name (line 494):
    
    # Assigning a Name to a Name (line 494):
    # Getting the type of 'slice_all' (line 494)
    slice_all_31025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 14), 'slice_all')
    # Assigning a type to the variable 'slice_R' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'slice_R', slice_all_31025)
    
    # Multiple assignment of 3 elements.
    
    # Assigning a Name to a Name (line 495):
    # Getting the type of 'Ninterv' (line 495)
    Ninterv_31026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 26), 'Ninterv')
    # Assigning a type to the variable 'step' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 19), 'step', Ninterv_31026)
    
    # Assigning a Name to a Name (line 495):
    # Getting the type of 'step' (line 495)
    step_31027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 19), 'step')
    # Assigning a type to the variable 'stop' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'stop', step_31027)
    
    # Assigning a Name to a Name (line 495):
    # Getting the type of 'stop' (line 495)
    stop_31028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'stop')
    # Assigning a type to the variable 'start' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'start', stop_31028)
    
    
    # Call to xrange(...): (line 496)
    # Processing the call arguments (line 496)
    int_31030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 20), 'int')
    # Getting the type of 'k' (line 496)
    k_31031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 23), 'k', False)
    int_31032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 25), 'int')
    # Applying the binary operator '+' (line 496)
    result_add_31033 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 23), '+', k_31031, int_31032)
    
    # Processing the call keyword arguments (line 496)
    kwargs_31034 = {}
    # Getting the type of 'xrange' (line 496)
    xrange_31029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 496)
    xrange_call_result_31035 = invoke(stypy.reporting.localization.Localization(__file__, 496, 13), xrange_31029, *[int_31030, result_add_31033], **kwargs_31034)
    
    # Testing the type of a for loop iterable (line 496)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 496, 4), xrange_call_result_31035)
    # Getting the type of the for loop variable (line 496)
    for_loop_var_31036 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 496, 4), xrange_call_result_31035)
    # Assigning a type to the variable 'i' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'i', for_loop_var_31036)
    # SSA begins for a for statement (line 496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'start' (line 497)
    start_31037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'start')
    int_31038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 18), 'int')
    # Applying the binary operator '>>=' (line 497)
    result_irshift_31039 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 8), '>>=', start_31037, int_31038)
    # Assigning a type to the variable 'start' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'start', result_irshift_31039)
    
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to tupleset(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'slice_R' (line 498)
    slice_R_31041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 27), 'slice_R', False)
    # Getting the type of 'axis' (line 498)
    axis_31042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'axis', False)
    
    # Call to slice(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'start' (line 498)
    start_31044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 48), 'start', False)
    # Getting the type of 'stop' (line 498)
    stop_31045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 55), 'stop', False)
    # Getting the type of 'step' (line 498)
    step_31046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 61), 'step', False)
    # Processing the call keyword arguments (line 498)
    kwargs_31047 = {}
    # Getting the type of 'slice' (line 498)
    slice_31043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 42), 'slice', False)
    # Calling slice(args, kwargs) (line 498)
    slice_call_result_31048 = invoke(stypy.reporting.localization.Localization(__file__, 498, 42), slice_31043, *[start_31044, stop_31045, step_31046], **kwargs_31047)
    
    # Processing the call keyword arguments (line 498)
    kwargs_31049 = {}
    # Getting the type of 'tupleset' (line 498)
    tupleset_31040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 18), 'tupleset', False)
    # Calling tupleset(args, kwargs) (line 498)
    tupleset_call_result_31050 = invoke(stypy.reporting.localization.Localization(__file__, 498, 18), tupleset_31040, *[slice_R_31041, axis_31042, slice_call_result_31048], **kwargs_31049)
    
    # Assigning a type to the variable 'slice_R' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'slice_R', tupleset_call_result_31050)
    
    # Getting the type of 'step' (line 499)
    step_31051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'step')
    int_31052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 17), 'int')
    # Applying the binary operator '>>=' (line 499)
    result_irshift_31053 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 8), '>>=', step_31051, int_31052)
    # Assigning a type to the variable 'step' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'step', result_irshift_31053)
    
    
    # Assigning a BinOp to a Subscript (line 500):
    
    # Assigning a BinOp to a Subscript (line 500):
    float_31054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 20), 'float')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 500)
    tuple_31055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 500)
    # Adding element type (line 500)
    # Getting the type of 'i' (line 500)
    i_31056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 28), 'i')
    int_31057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 30), 'int')
    # Applying the binary operator '-' (line 500)
    result_sub_31058 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 28), '-', i_31056, int_31057)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 28), tuple_31055, result_sub_31058)
    # Adding element type (line 500)
    int_31059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 28), tuple_31055, int_31059)
    
    # Getting the type of 'R' (line 500)
    R_31060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 25), 'R')
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___31061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 25), R_31060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_31062 = invoke(stypy.reporting.localization.Localization(__file__, 500, 25), getitem___31061, tuple_31055)
    
    # Getting the type of 'h' (line 500)
    h_31063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 39), 'h')
    
    # Call to sum(...): (line 500)
    # Processing the call keyword arguments (line 500)
    # Getting the type of 'axis' (line 500)
    axis_31069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 61), 'axis', False)
    keyword_31070 = axis_31069
    kwargs_31071 = {'axis': keyword_31070}
    
    # Obtaining the type of the subscript
    # Getting the type of 'slice_R' (line 500)
    slice_R_31064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 43), 'slice_R', False)
    # Getting the type of 'y' (line 500)
    y_31065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 41), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___31066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), y_31065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_31067 = invoke(stypy.reporting.localization.Localization(__file__, 500, 41), getitem___31066, slice_R_31064)
    
    # Obtaining the member 'sum' of a type (line 500)
    sum_31068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), subscript_call_result_31067, 'sum')
    # Calling sum(args, kwargs) (line 500)
    sum_call_result_31072 = invoke(stypy.reporting.localization.Localization(__file__, 500, 41), sum_31068, *[], **kwargs_31071)
    
    # Applying the binary operator '*' (line 500)
    result_mul_31073 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 39), '*', h_31063, sum_call_result_31072)
    
    # Applying the binary operator '+' (line 500)
    result_add_31074 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 25), '+', subscript_call_result_31062, result_mul_31073)
    
    # Applying the binary operator '*' (line 500)
    result_mul_31075 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 20), '*', float_31054, result_add_31074)
    
    # Getting the type of 'R' (line 500)
    R_31076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'R')
    
    # Obtaining an instance of the builtin type 'tuple' (line 500)
    tuple_31077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 500)
    # Adding element type (line 500)
    # Getting the type of 'i' (line 500)
    i_31078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 11), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 11), tuple_31077, i_31078)
    # Adding element type (line 500)
    int_31079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 11), tuple_31077, int_31079)
    
    # Storing an element on a container (line 500)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 8), R_31076, (tuple_31077, result_mul_31075))
    
    
    # Call to xrange(...): (line 501)
    # Processing the call arguments (line 501)
    int_31081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 24), 'int')
    # Getting the type of 'i' (line 501)
    i_31082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 27), 'i', False)
    int_31083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 29), 'int')
    # Applying the binary operator '+' (line 501)
    result_add_31084 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 27), '+', i_31082, int_31083)
    
    # Processing the call keyword arguments (line 501)
    kwargs_31085 = {}
    # Getting the type of 'xrange' (line 501)
    xrange_31080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 501)
    xrange_call_result_31086 = invoke(stypy.reporting.localization.Localization(__file__, 501, 17), xrange_31080, *[int_31081, result_add_31084], **kwargs_31085)
    
    # Testing the type of a for loop iterable (line 501)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 501, 8), xrange_call_result_31086)
    # Getting the type of the for loop variable (line 501)
    for_loop_var_31087 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 501, 8), xrange_call_result_31086)
    # Assigning a type to the variable 'j' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'j', for_loop_var_31087)
    # SSA begins for a for statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 502):
    
    # Assigning a Subscript to a Name (line 502):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 502)
    tuple_31088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 502)
    # Adding element type (line 502)
    # Getting the type of 'i' (line 502)
    i_31089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 22), tuple_31088, i_31089)
    # Adding element type (line 502)
    # Getting the type of 'j' (line 502)
    j_31090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'j')
    int_31091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 27), 'int')
    # Applying the binary operator '-' (line 502)
    result_sub_31092 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 25), '-', j_31090, int_31091)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 22), tuple_31088, result_sub_31092)
    
    # Getting the type of 'R' (line 502)
    R_31093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'R')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___31094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 19), R_31093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_31095 = invoke(stypy.reporting.localization.Localization(__file__, 502, 19), getitem___31094, tuple_31088)
    
    # Assigning a type to the variable 'prev' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'prev', subscript_call_result_31095)
    
    # Assigning a BinOp to a Subscript (line 503):
    
    # Assigning a BinOp to a Subscript (line 503):
    # Getting the type of 'prev' (line 503)
    prev_31096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'prev')
    # Getting the type of 'prev' (line 503)
    prev_31097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 32), 'prev')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 503)
    tuple_31098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 503)
    # Adding element type (line 503)
    # Getting the type of 'i' (line 503)
    i_31099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'i')
    int_31100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 42), 'int')
    # Applying the binary operator '-' (line 503)
    result_sub_31101 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 40), '-', i_31099, int_31100)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 40), tuple_31098, result_sub_31101)
    # Adding element type (line 503)
    # Getting the type of 'j' (line 503)
    j_31102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 45), 'j')
    int_31103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 47), 'int')
    # Applying the binary operator '-' (line 503)
    result_sub_31104 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 45), '-', j_31102, int_31103)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 40), tuple_31098, result_sub_31104)
    
    # Getting the type of 'R' (line 503)
    R_31105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 37), 'R')
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___31106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 37), R_31105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_31107 = invoke(stypy.reporting.localization.Localization(__file__, 503, 37), getitem___31106, tuple_31098)
    
    # Applying the binary operator '-' (line 503)
    result_sub_31108 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 32), '-', prev_31097, subscript_call_result_31107)
    
    int_31109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 56), 'int')
    int_31110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 62), 'int')
    # Getting the type of 'j' (line 503)
    j_31111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 64), 'j')
    # Applying the binary operator '*' (line 503)
    result_mul_31112 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 62), '*', int_31110, j_31111)
    
    # Applying the binary operator '<<' (line 503)
    result_lshift_31113 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 56), '<<', int_31109, result_mul_31112)
    
    int_31114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 68), 'int')
    # Applying the binary operator '-' (line 503)
    result_sub_31115 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 55), '-', result_lshift_31113, int_31114)
    
    # Applying the binary operator 'div' (line 503)
    result_div_31116 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 31), 'div', result_sub_31108, result_sub_31115)
    
    # Applying the binary operator '+' (line 503)
    result_add_31117 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 24), '+', prev_31096, result_div_31116)
    
    # Getting the type of 'R' (line 503)
    R_31118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'R')
    
    # Obtaining an instance of the builtin type 'tuple' (line 503)
    tuple_31119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 503)
    # Adding element type (line 503)
    # Getting the type of 'i' (line 503)
    i_31120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 15), tuple_31119, i_31120)
    # Adding element type (line 503)
    # Getting the type of 'j' (line 503)
    j_31121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 18), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 15), tuple_31119, j_31121)
    
    # Storing an element on a container (line 503)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 12), R_31118, (tuple_31119, result_add_31117))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'h' (line 504)
    h_31122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'h')
    float_31123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 13), 'float')
    # Applying the binary operator 'div=' (line 504)
    result_div_31124 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 8), 'div=', h_31122, float_31123)
    # Assigning a type to the variable 'h' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'h', result_div_31124)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 506)
    show_31125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 7), 'show')
    # Testing the type of an if condition (line 506)
    if_condition_31126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 506, 4), show_31125)
    # Assigning a type to the variable 'if_condition_31126' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'if_condition_31126', if_condition_31126)
    # SSA begins for if statement (line 506)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to isscalar(...): (line 507)
    # Processing the call arguments (line 507)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 507)
    tuple_31129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 507)
    # Adding element type (line 507)
    int_31130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 30), tuple_31129, int_31130)
    # Adding element type (line 507)
    int_31131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 30), tuple_31129, int_31131)
    
    # Getting the type of 'R' (line 507)
    R_31132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 27), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___31133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 27), R_31132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_31134 = invoke(stypy.reporting.localization.Localization(__file__, 507, 27), getitem___31133, tuple_31129)
    
    # Processing the call keyword arguments (line 507)
    kwargs_31135 = {}
    # Getting the type of 'np' (line 507)
    np_31127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 507)
    isscalar_31128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), np_31127, 'isscalar')
    # Calling isscalar(args, kwargs) (line 507)
    isscalar_call_result_31136 = invoke(stypy.reporting.localization.Localization(__file__, 507, 15), isscalar_31128, *[subscript_call_result_31134], **kwargs_31135)
    
    # Applying the 'not' unary operator (line 507)
    result_not__31137 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 11), 'not', isscalar_call_result_31136)
    
    # Testing the type of an if condition (line 507)
    if_condition_31138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 8), result_not__31137)
    # Assigning a type to the variable 'if_condition_31138' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'if_condition_31138', if_condition_31138)
    # SSA begins for if statement (line 507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 508)
    # Processing the call arguments (line 508)
    str_31140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 18), 'str', '*** Printing table only supported for integrals')
    str_31141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 18), 'str', ' of a single data set.')
    # Applying the binary operator '+' (line 508)
    result_add_31142 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 18), '+', str_31140, str_31141)
    
    # Processing the call keyword arguments (line 508)
    kwargs_31143 = {}
    # Getting the type of 'print' (line 508)
    print_31139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'print', False)
    # Calling print(args, kwargs) (line 508)
    print_call_result_31144 = invoke(stypy.reporting.localization.Localization(__file__, 508, 12), print_31139, *[result_add_31142], **kwargs_31143)
    
    # SSA branch for the else part of an if statement (line 507)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 511)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 512):
    
    # Assigning a Subscript to a Name (line 512):
    
    # Obtaining the type of the subscript
    int_31145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 30), 'int')
    # Getting the type of 'show' (line 512)
    show_31146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 25), 'show')
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___31147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 25), show_31146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 512)
    subscript_call_result_31148 = invoke(stypy.reporting.localization.Localization(__file__, 512, 25), getitem___31147, int_31145)
    
    # Assigning a type to the variable 'precis' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'precis', subscript_call_result_31148)
    # SSA branch for the except part of a try statement (line 511)
    # SSA branch for the except 'Tuple' branch of a try statement (line 511)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 514):
    
    # Assigning a Num to a Name (line 514):
    int_31149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 25), 'int')
    # Assigning a type to the variable 'precis' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'precis', int_31149)
    # SSA join for try-except statement (line 511)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 516):
    
    # Assigning a Subscript to a Name (line 516):
    
    # Obtaining the type of the subscript
    int_31150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 29), 'int')
    # Getting the type of 'show' (line 516)
    show_31151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 24), 'show')
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___31152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 24), show_31151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_31153 = invoke(stypy.reporting.localization.Localization(__file__, 516, 24), getitem___31152, int_31150)
    
    # Assigning a type to the variable 'width' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'width', subscript_call_result_31153)
    # SSA branch for the except part of a try statement (line 515)
    # SSA branch for the except 'Tuple' branch of a try statement (line 515)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 518):
    
    # Assigning a Num to a Name (line 518):
    int_31154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 24), 'int')
    # Assigning a type to the variable 'width' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'width', int_31154)
    # SSA join for try-except statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 519):
    
    # Assigning a BinOp to a Name (line 519):
    str_31155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 22), 'str', '%%%d.%df')
    
    # Obtaining an instance of the builtin type 'tuple' (line 519)
    tuple_31156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 519)
    # Adding element type (line 519)
    # Getting the type of 'width' (line 519)
    width_31157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 36), 'width')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 36), tuple_31156, width_31157)
    # Adding element type (line 519)
    # Getting the type of 'precis' (line 519)
    precis_31158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 43), 'precis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 36), tuple_31156, precis_31158)
    
    # Applying the binary operator '%' (line 519)
    result_mod_31159 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 22), '%', str_31155, tuple_31156)
    
    # Assigning a type to the variable 'formstr' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'formstr', result_mod_31159)
    
    # Assigning a Str to a Name (line 521):
    
    # Assigning a Str to a Name (line 521):
    str_31160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 20), 'str', 'Richardson Extrapolation Table for Romberg Integration')
    # Assigning a type to the variable 'title' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'title', str_31160)
    
    # Call to print(...): (line 522)
    # Processing the call arguments (line 522)
    str_31162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 18), 'str', '')
    
    # Call to center(...): (line 522)
    # Processing the call arguments (line 522)
    int_31165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 35), 'int')
    # Processing the call keyword arguments (line 522)
    kwargs_31166 = {}
    # Getting the type of 'title' (line 522)
    title_31163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 22), 'title', False)
    # Obtaining the member 'center' of a type (line 522)
    center_31164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 22), title_31163, 'center')
    # Calling center(args, kwargs) (line 522)
    center_call_result_31167 = invoke(stypy.reporting.localization.Localization(__file__, 522, 22), center_31164, *[int_31165], **kwargs_31166)
    
    str_31168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 40), 'str', '=')
    int_31169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 46), 'int')
    # Applying the binary operator '*' (line 522)
    result_mul_31170 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 40), '*', str_31168, int_31169)
    
    # Processing the call keyword arguments (line 522)
    str_31171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 54), 'str', '\n')
    keyword_31172 = str_31171
    str_31173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 64), 'str', '')
    keyword_31174 = str_31173
    kwargs_31175 = {'end': keyword_31174, 'sep': keyword_31172}
    # Getting the type of 'print' (line 522)
    print_31161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'print', False)
    # Calling print(args, kwargs) (line 522)
    print_call_result_31176 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), print_31161, *[str_31162, center_call_result_31167, result_mul_31170], **kwargs_31175)
    
    
    
    # Call to xrange(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'k' (line 523)
    k_31178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 28), 'k', False)
    int_31179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 30), 'int')
    # Applying the binary operator '+' (line 523)
    result_add_31180 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 28), '+', k_31178, int_31179)
    
    # Processing the call keyword arguments (line 523)
    kwargs_31181 = {}
    # Getting the type of 'xrange' (line 523)
    xrange_31177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 21), 'xrange', False)
    # Calling xrange(args, kwargs) (line 523)
    xrange_call_result_31182 = invoke(stypy.reporting.localization.Localization(__file__, 523, 21), xrange_31177, *[result_add_31180], **kwargs_31181)
    
    # Testing the type of a for loop iterable (line 523)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 523, 12), xrange_call_result_31182)
    # Getting the type of the for loop variable (line 523)
    for_loop_var_31183 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 523, 12), xrange_call_result_31182)
    # Assigning a type to the variable 'i' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'i', for_loop_var_31183)
    # SSA begins for a for statement (line 523)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'i' (line 524)
    i_31185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 32), 'i', False)
    int_31186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 34), 'int')
    # Applying the binary operator '+' (line 524)
    result_add_31187 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 32), '+', i_31185, int_31186)
    
    # Processing the call keyword arguments (line 524)
    kwargs_31188 = {}
    # Getting the type of 'xrange' (line 524)
    xrange_31184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 25), 'xrange', False)
    # Calling xrange(args, kwargs) (line 524)
    xrange_call_result_31189 = invoke(stypy.reporting.localization.Localization(__file__, 524, 25), xrange_31184, *[result_add_31187], **kwargs_31188)
    
    # Testing the type of a for loop iterable (line 524)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 524, 16), xrange_call_result_31189)
    # Getting the type of the for loop variable (line 524)
    for_loop_var_31190 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 524, 16), xrange_call_result_31189)
    # Assigning a type to the variable 'j' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'j', for_loop_var_31190)
    # SSA begins for a for statement (line 524)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'formstr' (line 525)
    formstr_31192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 26), 'formstr', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 525)
    tuple_31193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 525)
    # Adding element type (line 525)
    # Getting the type of 'i' (line 525)
    i_31194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 39), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 39), tuple_31193, i_31194)
    # Adding element type (line 525)
    # Getting the type of 'j' (line 525)
    j_31195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 42), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 39), tuple_31193, j_31195)
    
    # Getting the type of 'R' (line 525)
    R_31196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 36), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 525)
    getitem___31197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 36), R_31196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 525)
    subscript_call_result_31198 = invoke(stypy.reporting.localization.Localization(__file__, 525, 36), getitem___31197, tuple_31193)
    
    # Applying the binary operator '%' (line 525)
    result_mod_31199 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 26), '%', formstr_31192, subscript_call_result_31198)
    
    # Processing the call keyword arguments (line 525)
    str_31200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 51), 'str', ' ')
    keyword_31201 = str_31200
    kwargs_31202 = {'end': keyword_31201}
    # Getting the type of 'print' (line 525)
    print_31191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 20), 'print', False)
    # Calling print(args, kwargs) (line 525)
    print_call_result_31203 = invoke(stypy.reporting.localization.Localization(__file__, 525, 20), print_31191, *[result_mod_31199], **kwargs_31202)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 526)
    # Processing the call keyword arguments (line 526)
    kwargs_31205 = {}
    # Getting the type of 'print' (line 526)
    print_31204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'print', False)
    # Calling print(args, kwargs) (line 526)
    print_call_result_31206 = invoke(stypy.reporting.localization.Localization(__file__, 526, 16), print_31204, *[], **kwargs_31205)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 527)
    # Processing the call arguments (line 527)
    str_31208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 18), 'str', '=')
    int_31209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 24), 'int')
    # Applying the binary operator '*' (line 527)
    result_mul_31210 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 18), '*', str_31208, int_31209)
    
    # Processing the call keyword arguments (line 527)
    kwargs_31211 = {}
    # Getting the type of 'print' (line 527)
    print_31207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'print', False)
    # Calling print(args, kwargs) (line 527)
    print_call_result_31212 = invoke(stypy.reporting.localization.Localization(__file__, 527, 12), print_31207, *[result_mul_31210], **kwargs_31211)
    
    
    # Call to print(...): (line 528)
    # Processing the call keyword arguments (line 528)
    kwargs_31214 = {}
    # Getting the type of 'print' (line 528)
    print_31213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'print', False)
    # Calling print(args, kwargs) (line 528)
    print_call_result_31215 = invoke(stypy.reporting.localization.Localization(__file__, 528, 12), print_31213, *[], **kwargs_31214)
    
    # SSA join for if statement (line 507)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 506)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 530)
    tuple_31216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 530)
    # Adding element type (line 530)
    # Getting the type of 'k' (line 530)
    k_31217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 14), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 14), tuple_31216, k_31217)
    # Adding element type (line 530)
    # Getting the type of 'k' (line 530)
    k_31218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 17), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 14), tuple_31216, k_31218)
    
    # Getting the type of 'R' (line 530)
    R_31219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'R')
    # Obtaining the member '__getitem__' of a type (line 530)
    getitem___31220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 11), R_31219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 530)
    subscript_call_result_31221 = invoke(stypy.reporting.localization.Localization(__file__, 530, 11), getitem___31220, tuple_31216)
    
    # Assigning a type to the variable 'stypy_return_type' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'stypy_return_type', subscript_call_result_31221)
    
    # ################# End of 'romb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'romb' in the type store
    # Getting the type of 'stypy_return_type' (line 439)
    stypy_return_type_31222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31222)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'romb'
    return stypy_return_type_31222

# Assigning a type to the variable 'romb' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'romb', romb)

@norecursion
def _difftrap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_difftrap'
    module_type_store = module_type_store.open_function_context('_difftrap', 544, 0, False)
    
    # Passed parameters checking function
    _difftrap.stypy_localization = localization
    _difftrap.stypy_type_of_self = None
    _difftrap.stypy_type_store = module_type_store
    _difftrap.stypy_function_name = '_difftrap'
    _difftrap.stypy_param_names_list = ['function', 'interval', 'numtraps']
    _difftrap.stypy_varargs_param_name = None
    _difftrap.stypy_kwargs_param_name = None
    _difftrap.stypy_call_defaults = defaults
    _difftrap.stypy_call_varargs = varargs
    _difftrap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_difftrap', ['function', 'interval', 'numtraps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_difftrap', localization, ['function', 'interval', 'numtraps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_difftrap(...)' code ##################

    str_31223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, (-1)), 'str', "\n    Perform part of the trapezoidal rule to integrate a function.\n    Assume that we had called difftrap with all lower powers-of-2\n    starting with 1.  Calling difftrap only returns the summation\n    of the new ordinates.  It does _not_ multiply by the width\n    of the trapezoids.  This must be performed by the caller.\n        'function' is the function to evaluate (must accept vector arguments).\n        'interval' is a sequence with lower and upper limits\n                   of integration.\n        'numtraps' is the number of trapezoids to use (must be a\n                   power-of-2).\n    ")
    
    
    # Getting the type of 'numtraps' (line 557)
    numtraps_31224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 7), 'numtraps')
    int_31225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 19), 'int')
    # Applying the binary operator '<=' (line 557)
    result_le_31226 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 7), '<=', numtraps_31224, int_31225)
    
    # Testing the type of an if condition (line 557)
    if_condition_31227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 4), result_le_31226)
    # Assigning a type to the variable 'if_condition_31227' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'if_condition_31227', if_condition_31227)
    # SSA begins for if statement (line 557)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 558)
    # Processing the call arguments (line 558)
    str_31229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 25), 'str', 'numtraps must be > 0 in difftrap().')
    # Processing the call keyword arguments (line 558)
    kwargs_31230 = {}
    # Getting the type of 'ValueError' (line 558)
    ValueError_31228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 558)
    ValueError_call_result_31231 = invoke(stypy.reporting.localization.Localization(__file__, 558, 14), ValueError_31228, *[str_31229], **kwargs_31230)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 558, 8), ValueError_call_result_31231, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 557)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'numtraps' (line 559)
    numtraps_31232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'numtraps')
    int_31233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 21), 'int')
    # Applying the binary operator '==' (line 559)
    result_eq_31234 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 9), '==', numtraps_31232, int_31233)
    
    # Testing the type of an if condition (line 559)
    if_condition_31235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 9), result_eq_31234)
    # Assigning a type to the variable 'if_condition_31235' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'if_condition_31235', if_condition_31235)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    float_31236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 15), 'float')
    
    # Call to function(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Obtaining the type of the subscript
    int_31238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 38), 'int')
    # Getting the type of 'interval' (line 560)
    interval_31239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 29), 'interval', False)
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___31240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 29), interval_31239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_31241 = invoke(stypy.reporting.localization.Localization(__file__, 560, 29), getitem___31240, int_31238)
    
    # Processing the call keyword arguments (line 560)
    kwargs_31242 = {}
    # Getting the type of 'function' (line 560)
    function_31237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'function', False)
    # Calling function(args, kwargs) (line 560)
    function_call_result_31243 = invoke(stypy.reporting.localization.Localization(__file__, 560, 20), function_31237, *[subscript_call_result_31241], **kwargs_31242)
    
    
    # Call to function(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Obtaining the type of the subscript
    int_31245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 60), 'int')
    # Getting the type of 'interval' (line 560)
    interval_31246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 51), 'interval', False)
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___31247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 51), interval_31246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_31248 = invoke(stypy.reporting.localization.Localization(__file__, 560, 51), getitem___31247, int_31245)
    
    # Processing the call keyword arguments (line 560)
    kwargs_31249 = {}
    # Getting the type of 'function' (line 560)
    function_31244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'function', False)
    # Calling function(args, kwargs) (line 560)
    function_call_result_31250 = invoke(stypy.reporting.localization.Localization(__file__, 560, 42), function_31244, *[subscript_call_result_31248], **kwargs_31249)
    
    # Applying the binary operator '+' (line 560)
    result_add_31251 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 20), '+', function_call_result_31243, function_call_result_31250)
    
    # Applying the binary operator '*' (line 560)
    result_mul_31252 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 15), '*', float_31236, result_add_31251)
    
    # Assigning a type to the variable 'stypy_return_type' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'stypy_return_type', result_mul_31252)
    # SSA branch for the else part of an if statement (line 559)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 562):
    
    # Assigning a BinOp to a Name (line 562):
    # Getting the type of 'numtraps' (line 562)
    numtraps_31253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 19), 'numtraps')
    int_31254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 28), 'int')
    # Applying the binary operator 'div' (line 562)
    result_div_31255 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 19), 'div', numtraps_31253, int_31254)
    
    # Assigning a type to the variable 'numtosum' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'numtosum', result_div_31255)
    
    # Assigning a BinOp to a Name (line 563):
    
    # Assigning a BinOp to a Name (line 563):
    
    # Call to float(...): (line 563)
    # Processing the call arguments (line 563)
    
    # Obtaining the type of the subscript
    int_31257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 27), 'int')
    # Getting the type of 'interval' (line 563)
    interval_31258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 18), 'interval', False)
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___31259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 18), interval_31258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_31260 = invoke(stypy.reporting.localization.Localization(__file__, 563, 18), getitem___31259, int_31257)
    
    
    # Obtaining the type of the subscript
    int_31261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 39), 'int')
    # Getting the type of 'interval' (line 563)
    interval_31262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 30), 'interval', False)
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___31263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 30), interval_31262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_31264 = invoke(stypy.reporting.localization.Localization(__file__, 563, 30), getitem___31263, int_31261)
    
    # Applying the binary operator '-' (line 563)
    result_sub_31265 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 18), '-', subscript_call_result_31260, subscript_call_result_31264)
    
    # Processing the call keyword arguments (line 563)
    kwargs_31266 = {}
    # Getting the type of 'float' (line 563)
    float_31256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'float', False)
    # Calling float(args, kwargs) (line 563)
    float_call_result_31267 = invoke(stypy.reporting.localization.Localization(__file__, 563, 12), float_31256, *[result_sub_31265], **kwargs_31266)
    
    # Getting the type of 'numtosum' (line 563)
    numtosum_31268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 43), 'numtosum')
    # Applying the binary operator 'div' (line 563)
    result_div_31269 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 12), 'div', float_call_result_31267, numtosum_31268)
    
    # Assigning a type to the variable 'h' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'h', result_div_31269)
    
    # Assigning a BinOp to a Name (line 564):
    
    # Assigning a BinOp to a Name (line 564):
    
    # Obtaining the type of the subscript
    int_31270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 23), 'int')
    # Getting the type of 'interval' (line 564)
    interval_31271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'interval')
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___31272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 14), interval_31271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_31273 = invoke(stypy.reporting.localization.Localization(__file__, 564, 14), getitem___31272, int_31270)
    
    float_31274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 28), 'float')
    # Getting the type of 'h' (line 564)
    h_31275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 34), 'h')
    # Applying the binary operator '*' (line 564)
    result_mul_31276 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 28), '*', float_31274, h_31275)
    
    # Applying the binary operator '+' (line 564)
    result_add_31277 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 14), '+', subscript_call_result_31273, result_mul_31276)
    
    # Assigning a type to the variable 'lox' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'lox', result_add_31277)
    
    # Assigning a BinOp to a Name (line 565):
    
    # Assigning a BinOp to a Name (line 565):
    # Getting the type of 'lox' (line 565)
    lox_31278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 17), 'lox')
    # Getting the type of 'h' (line 565)
    h_31279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 23), 'h')
    
    # Call to arange(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'numtosum' (line 565)
    numtosum_31282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 37), 'numtosum', False)
    # Processing the call keyword arguments (line 565)
    kwargs_31283 = {}
    # Getting the type of 'np' (line 565)
    np_31280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 27), 'np', False)
    # Obtaining the member 'arange' of a type (line 565)
    arange_31281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 27), np_31280, 'arange')
    # Calling arange(args, kwargs) (line 565)
    arange_call_result_31284 = invoke(stypy.reporting.localization.Localization(__file__, 565, 27), arange_31281, *[numtosum_31282], **kwargs_31283)
    
    # Applying the binary operator '*' (line 565)
    result_mul_31285 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 23), '*', h_31279, arange_call_result_31284)
    
    # Applying the binary operator '+' (line 565)
    result_add_31286 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 17), '+', lox_31278, result_mul_31285)
    
    # Assigning a type to the variable 'points' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'points', result_add_31286)
    
    # Assigning a Call to a Name (line 566):
    
    # Assigning a Call to a Name (line 566):
    
    # Call to sum(...): (line 566)
    # Processing the call arguments (line 566)
    
    # Call to function(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'points' (line 566)
    points_31290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 28), 'points', False)
    # Processing the call keyword arguments (line 566)
    kwargs_31291 = {}
    # Getting the type of 'function' (line 566)
    function_31289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'function', False)
    # Calling function(args, kwargs) (line 566)
    function_call_result_31292 = invoke(stypy.reporting.localization.Localization(__file__, 566, 19), function_31289, *[points_31290], **kwargs_31291)
    
    # Processing the call keyword arguments (line 566)
    int_31293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 42), 'int')
    keyword_31294 = int_31293
    kwargs_31295 = {'axis': keyword_31294}
    # Getting the type of 'np' (line 566)
    np_31287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 566)
    sum_31288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), np_31287, 'sum')
    # Calling sum(args, kwargs) (line 566)
    sum_call_result_31296 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), sum_31288, *[function_call_result_31292], **kwargs_31295)
    
    # Assigning a type to the variable 's' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 's', sum_call_result_31296)
    # Getting the type of 's' (line 567)
    s_31297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'stypy_return_type', s_31297)
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 557)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_difftrap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_difftrap' in the type store
    # Getting the type of 'stypy_return_type' (line 544)
    stypy_return_type_31298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_difftrap'
    return stypy_return_type_31298

# Assigning a type to the variable '_difftrap' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), '_difftrap', _difftrap)

@norecursion
def _romberg_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_romberg_diff'
    module_type_store = module_type_store.open_function_context('_romberg_diff', 570, 0, False)
    
    # Passed parameters checking function
    _romberg_diff.stypy_localization = localization
    _romberg_diff.stypy_type_of_self = None
    _romberg_diff.stypy_type_store = module_type_store
    _romberg_diff.stypy_function_name = '_romberg_diff'
    _romberg_diff.stypy_param_names_list = ['b', 'c', 'k']
    _romberg_diff.stypy_varargs_param_name = None
    _romberg_diff.stypy_kwargs_param_name = None
    _romberg_diff.stypy_call_defaults = defaults
    _romberg_diff.stypy_call_varargs = varargs
    _romberg_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_romberg_diff', ['b', 'c', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_romberg_diff', localization, ['b', 'c', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_romberg_diff(...)' code ##################

    str_31299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, (-1)), 'str', '\n    Compute the differences for the Romberg quadrature corrections.\n    See Forman Acton\'s "Real Computing Made Real," p 143.\n    ')
    
    # Assigning a BinOp to a Name (line 575):
    
    # Assigning a BinOp to a Name (line 575):
    float_31300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 10), 'float')
    # Getting the type of 'k' (line 575)
    k_31301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 15), 'k')
    # Applying the binary operator '**' (line 575)
    result_pow_31302 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 10), '**', float_31300, k_31301)
    
    # Assigning a type to the variable 'tmp' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'tmp', result_pow_31302)
    # Getting the type of 'tmp' (line 576)
    tmp_31303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'tmp')
    # Getting the type of 'c' (line 576)
    c_31304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 18), 'c')
    # Applying the binary operator '*' (line 576)
    result_mul_31305 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 12), '*', tmp_31303, c_31304)
    
    # Getting the type of 'b' (line 576)
    b_31306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 22), 'b')
    # Applying the binary operator '-' (line 576)
    result_sub_31307 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 12), '-', result_mul_31305, b_31306)
    
    # Getting the type of 'tmp' (line 576)
    tmp_31308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 26), 'tmp')
    float_31309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 32), 'float')
    # Applying the binary operator '-' (line 576)
    result_sub_31310 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 26), '-', tmp_31308, float_31309)
    
    # Applying the binary operator 'div' (line 576)
    result_div_31311 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 11), 'div', result_sub_31307, result_sub_31310)
    
    # Assigning a type to the variable 'stypy_return_type' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'stypy_return_type', result_div_31311)
    
    # ################# End of '_romberg_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_romberg_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 570)
    stypy_return_type_31312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31312)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_romberg_diff'
    return stypy_return_type_31312

# Assigning a type to the variable '_romberg_diff' (line 570)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), '_romberg_diff', _romberg_diff)

@norecursion
def _printresmat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_printresmat'
    module_type_store = module_type_store.open_function_context('_printresmat', 579, 0, False)
    
    # Passed parameters checking function
    _printresmat.stypy_localization = localization
    _printresmat.stypy_type_of_self = None
    _printresmat.stypy_type_store = module_type_store
    _printresmat.stypy_function_name = '_printresmat'
    _printresmat.stypy_param_names_list = ['function', 'interval', 'resmat']
    _printresmat.stypy_varargs_param_name = None
    _printresmat.stypy_kwargs_param_name = None
    _printresmat.stypy_call_defaults = defaults
    _printresmat.stypy_call_varargs = varargs
    _printresmat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_printresmat', ['function', 'interval', 'resmat'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_printresmat', localization, ['function', 'interval', 'resmat'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_printresmat(...)' code ##################

    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 581):
    int_31313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 12), 'int')
    # Assigning a type to the variable 'j' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'j', int_31313)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'j' (line 581)
    j_31314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'j')
    # Assigning a type to the variable 'i' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'i', j_31314)
    
    # Call to print(...): (line 582)
    # Processing the call arguments (line 582)
    str_31316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 10), 'str', 'Romberg integration of')
    
    # Call to repr(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'function' (line 582)
    function_31318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 41), 'function', False)
    # Processing the call keyword arguments (line 582)
    kwargs_31319 = {}
    # Getting the type of 'repr' (line 582)
    repr_31317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 36), 'repr', False)
    # Calling repr(args, kwargs) (line 582)
    repr_call_result_31320 = invoke(stypy.reporting.localization.Localization(__file__, 582, 36), repr_31317, *[function_31318], **kwargs_31319)
    
    # Processing the call keyword arguments (line 582)
    str_31321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 56), 'str', ' ')
    keyword_31322 = str_31321
    kwargs_31323 = {'end': keyword_31322}
    # Getting the type of 'print' (line 582)
    print_31315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'print', False)
    # Calling print(args, kwargs) (line 582)
    print_call_result_31324 = invoke(stypy.reporting.localization.Localization(__file__, 582, 4), print_31315, *[str_31316, repr_call_result_31320], **kwargs_31323)
    
    
    # Call to print(...): (line 583)
    # Processing the call arguments (line 583)
    str_31326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 10), 'str', 'from')
    # Getting the type of 'interval' (line 583)
    interval_31327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 18), 'interval', False)
    # Processing the call keyword arguments (line 583)
    kwargs_31328 = {}
    # Getting the type of 'print' (line 583)
    print_31325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'print', False)
    # Calling print(args, kwargs) (line 583)
    print_call_result_31329 = invoke(stypy.reporting.localization.Localization(__file__, 583, 4), print_31325, *[str_31326, interval_31327], **kwargs_31328)
    
    
    # Call to print(...): (line 584)
    # Processing the call arguments (line 584)
    str_31331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 10), 'str', '')
    # Processing the call keyword arguments (line 584)
    kwargs_31332 = {}
    # Getting the type of 'print' (line 584)
    print_31330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'print', False)
    # Calling print(args, kwargs) (line 584)
    print_call_result_31333 = invoke(stypy.reporting.localization.Localization(__file__, 584, 4), print_31330, *[str_31331], **kwargs_31332)
    
    
    # Call to print(...): (line 585)
    # Processing the call arguments (line 585)
    str_31335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 10), 'str', '%6s %9s %9s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 585)
    tuple_31336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 585)
    # Adding element type (line 585)
    str_31337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 27), 'str', 'Steps')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 27), tuple_31336, str_31337)
    # Adding element type (line 585)
    str_31338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 36), 'str', 'StepSize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 27), tuple_31336, str_31338)
    # Adding element type (line 585)
    str_31339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 48), 'str', 'Results')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 27), tuple_31336, str_31339)
    
    # Applying the binary operator '%' (line 585)
    result_mod_31340 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 10), '%', str_31335, tuple_31336)
    
    # Processing the call keyword arguments (line 585)
    kwargs_31341 = {}
    # Getting the type of 'print' (line 585)
    print_31334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'print', False)
    # Calling print(args, kwargs) (line 585)
    print_call_result_31342 = invoke(stypy.reporting.localization.Localization(__file__, 585, 4), print_31334, *[result_mod_31340], **kwargs_31341)
    
    
    
    # Call to xrange(...): (line 586)
    # Processing the call arguments (line 586)
    
    # Call to len(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'resmat' (line 586)
    resmat_31345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), 'resmat', False)
    # Processing the call keyword arguments (line 586)
    kwargs_31346 = {}
    # Getting the type of 'len' (line 586)
    len_31344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'len', False)
    # Calling len(args, kwargs) (line 586)
    len_call_result_31347 = invoke(stypy.reporting.localization.Localization(__file__, 586, 20), len_31344, *[resmat_31345], **kwargs_31346)
    
    # Processing the call keyword arguments (line 586)
    kwargs_31348 = {}
    # Getting the type of 'xrange' (line 586)
    xrange_31343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 586)
    xrange_call_result_31349 = invoke(stypy.reporting.localization.Localization(__file__, 586, 13), xrange_31343, *[len_call_result_31347], **kwargs_31348)
    
    # Testing the type of a for loop iterable (line 586)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 586, 4), xrange_call_result_31349)
    # Getting the type of the for loop variable (line 586)
    for_loop_var_31350 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 586, 4), xrange_call_result_31349)
    # Assigning a type to the variable 'i' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'i', for_loop_var_31350)
    # SSA begins for a for statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 587)
    # Processing the call arguments (line 587)
    str_31352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 14), 'str', '%6d %9f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 587)
    tuple_31353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 587)
    # Adding element type (line 587)
    int_31354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 27), 'int')
    # Getting the type of 'i' (line 587)
    i_31355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'i', False)
    # Applying the binary operator '**' (line 587)
    result_pow_31356 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 27), '**', int_31354, i_31355)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 27), tuple_31353, result_pow_31356)
    # Adding element type (line 587)
    
    # Obtaining the type of the subscript
    int_31357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 43), 'int')
    # Getting the type of 'interval' (line 587)
    interval_31358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 34), 'interval', False)
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___31359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 34), interval_31358, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_31360 = invoke(stypy.reporting.localization.Localization(__file__, 587, 34), getitem___31359, int_31357)
    
    
    # Obtaining the type of the subscript
    int_31361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 55), 'int')
    # Getting the type of 'interval' (line 587)
    interval_31362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 46), 'interval', False)
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___31363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 46), interval_31362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_31364 = invoke(stypy.reporting.localization.Localization(__file__, 587, 46), getitem___31363, int_31361)
    
    # Applying the binary operator '-' (line 587)
    result_sub_31365 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 34), '-', subscript_call_result_31360, subscript_call_result_31364)
    
    float_31366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 60), 'float')
    # Getting the type of 'i' (line 587)
    i_31367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 64), 'i', False)
    # Applying the binary operator '**' (line 587)
    result_pow_31368 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 60), '**', float_31366, i_31367)
    
    # Applying the binary operator 'div' (line 587)
    result_div_31369 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 33), 'div', result_sub_31365, result_pow_31368)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 27), tuple_31353, result_div_31369)
    
    # Applying the binary operator '%' (line 587)
    result_mod_31370 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 14), '%', str_31352, tuple_31353)
    
    # Processing the call keyword arguments (line 587)
    str_31371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 73), 'str', ' ')
    keyword_31372 = str_31371
    kwargs_31373 = {'end': keyword_31372}
    # Getting the type of 'print' (line 587)
    print_31351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'print', False)
    # Calling print(args, kwargs) (line 587)
    print_call_result_31374 = invoke(stypy.reporting.localization.Localization(__file__, 587, 8), print_31351, *[result_mod_31370], **kwargs_31373)
    
    
    
    # Call to xrange(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'i' (line 588)
    i_31376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 24), 'i', False)
    int_31377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 26), 'int')
    # Applying the binary operator '+' (line 588)
    result_add_31378 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 24), '+', i_31376, int_31377)
    
    # Processing the call keyword arguments (line 588)
    kwargs_31379 = {}
    # Getting the type of 'xrange' (line 588)
    xrange_31375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 588)
    xrange_call_result_31380 = invoke(stypy.reporting.localization.Localization(__file__, 588, 17), xrange_31375, *[result_add_31378], **kwargs_31379)
    
    # Testing the type of a for loop iterable (line 588)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 588, 8), xrange_call_result_31380)
    # Getting the type of the for loop variable (line 588)
    for_loop_var_31381 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 588, 8), xrange_call_result_31380)
    # Assigning a type to the variable 'j' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'j', for_loop_var_31381)
    # SSA begins for a for statement (line 588)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 589)
    # Processing the call arguments (line 589)
    str_31383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 18), 'str', '%9f')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 589)
    j_31384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'j', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 589)
    i_31385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), 'i', False)
    # Getting the type of 'resmat' (line 589)
    resmat_31386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 27), 'resmat', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___31387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 27), resmat_31386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 589)
    subscript_call_result_31388 = invoke(stypy.reporting.localization.Localization(__file__, 589, 27), getitem___31387, i_31385)
    
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___31389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 27), subscript_call_result_31388, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 589)
    subscript_call_result_31390 = invoke(stypy.reporting.localization.Localization(__file__, 589, 27), getitem___31389, j_31384)
    
    # Applying the binary operator '%' (line 589)
    result_mod_31391 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 18), '%', str_31383, subscript_call_result_31390)
    
    # Processing the call keyword arguments (line 589)
    str_31392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 46), 'str', ' ')
    keyword_31393 = str_31392
    kwargs_31394 = {'end': keyword_31393}
    # Getting the type of 'print' (line 589)
    print_31382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'print', False)
    # Calling print(args, kwargs) (line 589)
    print_call_result_31395 = invoke(stypy.reporting.localization.Localization(__file__, 589, 12), print_31382, *[result_mod_31391], **kwargs_31394)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 590)
    # Processing the call arguments (line 590)
    str_31397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 14), 'str', '')
    # Processing the call keyword arguments (line 590)
    kwargs_31398 = {}
    # Getting the type of 'print' (line 590)
    print_31396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'print', False)
    # Calling print(args, kwargs) (line 590)
    print_call_result_31399 = invoke(stypy.reporting.localization.Localization(__file__, 590, 8), print_31396, *[str_31397], **kwargs_31398)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 591)
    # Processing the call arguments (line 591)
    str_31401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 10), 'str', '')
    # Processing the call keyword arguments (line 591)
    kwargs_31402 = {}
    # Getting the type of 'print' (line 591)
    print_31400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'print', False)
    # Calling print(args, kwargs) (line 591)
    print_call_result_31403 = invoke(stypy.reporting.localization.Localization(__file__, 591, 4), print_31400, *[str_31401], **kwargs_31402)
    
    
    # Call to print(...): (line 592)
    # Processing the call arguments (line 592)
    str_31405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 10), 'str', 'The final result is')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 592)
    j_31406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 43), 'j', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 592)
    i_31407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 40), 'i', False)
    # Getting the type of 'resmat' (line 592)
    resmat_31408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 33), 'resmat', False)
    # Obtaining the member '__getitem__' of a type (line 592)
    getitem___31409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 33), resmat_31408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 592)
    subscript_call_result_31410 = invoke(stypy.reporting.localization.Localization(__file__, 592, 33), getitem___31409, i_31407)
    
    # Obtaining the member '__getitem__' of a type (line 592)
    getitem___31411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 33), subscript_call_result_31410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 592)
    subscript_call_result_31412 = invoke(stypy.reporting.localization.Localization(__file__, 592, 33), getitem___31411, j_31406)
    
    # Processing the call keyword arguments (line 592)
    str_31413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 51), 'str', ' ')
    keyword_31414 = str_31413
    kwargs_31415 = {'end': keyword_31414}
    # Getting the type of 'print' (line 592)
    print_31404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'print', False)
    # Calling print(args, kwargs) (line 592)
    print_call_result_31416 = invoke(stypy.reporting.localization.Localization(__file__, 592, 4), print_31404, *[str_31405, subscript_call_result_31412], **kwargs_31415)
    
    
    # Call to print(...): (line 593)
    # Processing the call arguments (line 593)
    str_31418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 10), 'str', 'after')
    int_31419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 19), 'int')
    
    # Call to len(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'resmat' (line 593)
    resmat_31421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'resmat', False)
    # Processing the call keyword arguments (line 593)
    kwargs_31422 = {}
    # Getting the type of 'len' (line 593)
    len_31420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'len', False)
    # Calling len(args, kwargs) (line 593)
    len_call_result_31423 = invoke(stypy.reporting.localization.Localization(__file__, 593, 23), len_31420, *[resmat_31421], **kwargs_31422)
    
    int_31424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 35), 'int')
    # Applying the binary operator '-' (line 593)
    result_sub_31425 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 23), '-', len_call_result_31423, int_31424)
    
    # Applying the binary operator '**' (line 593)
    result_pow_31426 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 19), '**', int_31419, result_sub_31425)
    
    int_31427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 38), 'int')
    # Applying the binary operator '+' (line 593)
    result_add_31428 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 19), '+', result_pow_31426, int_31427)
    
    str_31429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 41), 'str', 'function evaluations.')
    # Processing the call keyword arguments (line 593)
    kwargs_31430 = {}
    # Getting the type of 'print' (line 593)
    print_31417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'print', False)
    # Calling print(args, kwargs) (line 593)
    print_call_result_31431 = invoke(stypy.reporting.localization.Localization(__file__, 593, 4), print_31417, *[str_31418, result_add_31428, str_31429], **kwargs_31430)
    
    
    # ################# End of '_printresmat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_printresmat' in the type store
    # Getting the type of 'stypy_return_type' (line 579)
    stypy_return_type_31432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31432)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_printresmat'
    return stypy_return_type_31432

# Assigning a type to the variable '_printresmat' (line 579)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), '_printresmat', _printresmat)

@norecursion
def romberg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 596)
    tuple_31433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 596)
    
    float_31434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 41), 'float')
    float_31435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 55), 'float')
    # Getting the type of 'False' (line 596)
    False_31436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 69), 'False')
    int_31437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 19), 'int')
    # Getting the type of 'False' (line 597)
    False_31438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 32), 'False')
    defaults = [tuple_31433, float_31434, float_31435, False_31436, int_31437, False_31438]
    # Create a new context for function 'romberg'
    module_type_store = module_type_store.open_function_context('romberg', 596, 0, False)
    
    # Passed parameters checking function
    romberg.stypy_localization = localization
    romberg.stypy_type_of_self = None
    romberg.stypy_type_store = module_type_store
    romberg.stypy_function_name = 'romberg'
    romberg.stypy_param_names_list = ['function', 'a', 'b', 'args', 'tol', 'rtol', 'show', 'divmax', 'vec_func']
    romberg.stypy_varargs_param_name = None
    romberg.stypy_kwargs_param_name = None
    romberg.stypy_call_defaults = defaults
    romberg.stypy_call_varargs = varargs
    romberg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'romberg', ['function', 'a', 'b', 'args', 'tol', 'rtol', 'show', 'divmax', 'vec_func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'romberg', localization, ['function', 'a', 'b', 'args', 'tol', 'rtol', 'show', 'divmax', 'vec_func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'romberg(...)' code ##################

    str_31439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, (-1)), 'str', '\n    Romberg integration of a callable function or method.\n\n    Returns the integral of `function` (a function of one variable)\n    over the interval (`a`, `b`).\n\n    If `show` is 1, the triangular array of the intermediate results\n    will be printed.  If `vec_func` is True (default is False), then\n    `function` is assumed to support vector arguments.\n\n    Parameters\n    ----------\n    function : callable\n        Function to be integrated.\n    a : float\n        Lower limit of integration.\n    b : float\n        Upper limit of integration.\n\n    Returns\n    -------\n    results  : float\n        Result of the integration.\n\n    Other Parameters\n    ----------------\n    args : tuple, optional\n        Extra arguments to pass to function. Each element of `args` will\n        be passed as a single argument to `func`. Default is to pass no\n        extra arguments.\n    tol, rtol : float, optional\n        The desired absolute and relative tolerances. Defaults are 1.48e-8.\n    show : bool, optional\n        Whether to print the results. Default is False.\n    divmax : int, optional\n        Maximum order of extrapolation. Default is 10.\n    vec_func : bool, optional\n        Whether `func` handles arrays as arguments (i.e whether it is a\n        "vector" function). Default is False.\n\n    See Also\n    --------\n    fixed_quad : Fixed-order Gaussian quadrature.\n    quad : Adaptive quadrature using QUADPACK.\n    dblquad : Double integrals.\n    tplquad : Triple integrals.\n    romb : Integrators for sampled data.\n    simps : Integrators for sampled data.\n    cumtrapz : Cumulative integration for sampled data.\n    ode : ODE integrator.\n    odeint : ODE integrator.\n\n    References\n    ----------\n    .. [1] \'Romberg\'s method\' http://en.wikipedia.org/wiki/Romberg%27s_method\n\n    Examples\n    --------\n    Integrate a gaussian from 0 to 1 and compare to the error function.\n\n    >>> from scipy import integrate\n    >>> from scipy.special import erf\n    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)\n    >>> result = integrate.romberg(gaussian, 0, 1, show=True)\n    Romberg integration of <function vfunc at ...> from [0, 1]\n\n    ::\n\n       Steps  StepSize  Results\n           1  1.000000  0.385872\n           2  0.500000  0.412631  0.421551\n           4  0.250000  0.419184  0.421368  0.421356\n           8  0.125000  0.420810  0.421352  0.421350  0.421350\n          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350\n          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350\n\n    The final result is 0.421350396475 after 33 function evaluations.\n\n    >>> print("%g %g" % (2*result, erf(1)))\n    0.842701 0.842701\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Call to isinf(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'a' (line 680)
    a_31442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'a', False)
    # Processing the call keyword arguments (line 680)
    kwargs_31443 = {}
    # Getting the type of 'np' (line 680)
    np_31440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 7), 'np', False)
    # Obtaining the member 'isinf' of a type (line 680)
    isinf_31441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 7), np_31440, 'isinf')
    # Calling isinf(args, kwargs) (line 680)
    isinf_call_result_31444 = invoke(stypy.reporting.localization.Localization(__file__, 680, 7), isinf_31441, *[a_31442], **kwargs_31443)
    
    
    # Call to isinf(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'b' (line 680)
    b_31447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 31), 'b', False)
    # Processing the call keyword arguments (line 680)
    kwargs_31448 = {}
    # Getting the type of 'np' (line 680)
    np_31445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 22), 'np', False)
    # Obtaining the member 'isinf' of a type (line 680)
    isinf_31446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 22), np_31445, 'isinf')
    # Calling isinf(args, kwargs) (line 680)
    isinf_call_result_31449 = invoke(stypy.reporting.localization.Localization(__file__, 680, 22), isinf_31446, *[b_31447], **kwargs_31448)
    
    # Applying the binary operator 'or' (line 680)
    result_or_keyword_31450 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 7), 'or', isinf_call_result_31444, isinf_call_result_31449)
    
    # Testing the type of an if condition (line 680)
    if_condition_31451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 680, 4), result_or_keyword_31450)
    # Assigning a type to the variable 'if_condition_31451' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'if_condition_31451', if_condition_31451)
    # SSA begins for if statement (line 680)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 681)
    # Processing the call arguments (line 681)
    str_31453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 25), 'str', 'Romberg integration only available for finite limits.')
    # Processing the call keyword arguments (line 681)
    kwargs_31454 = {}
    # Getting the type of 'ValueError' (line 681)
    ValueError_31452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 681)
    ValueError_call_result_31455 = invoke(stypy.reporting.localization.Localization(__file__, 681, 14), ValueError_31452, *[str_31453], **kwargs_31454)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 681, 8), ValueError_call_result_31455, 'raise parameter', BaseException)
    # SSA join for if statement (line 680)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 683):
    
    # Assigning a Call to a Name (line 683):
    
    # Call to vectorize1(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'function' (line 683)
    function_31457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 23), 'function', False)
    # Getting the type of 'args' (line 683)
    args_31458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 33), 'args', False)
    # Processing the call keyword arguments (line 683)
    # Getting the type of 'vec_func' (line 683)
    vec_func_31459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 48), 'vec_func', False)
    keyword_31460 = vec_func_31459
    kwargs_31461 = {'vec_func': keyword_31460}
    # Getting the type of 'vectorize1' (line 683)
    vectorize1_31456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'vectorize1', False)
    # Calling vectorize1(args, kwargs) (line 683)
    vectorize1_call_result_31462 = invoke(stypy.reporting.localization.Localization(__file__, 683, 12), vectorize1_31456, *[function_31457, args_31458], **kwargs_31461)
    
    # Assigning a type to the variable 'vfunc' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'vfunc', vectorize1_call_result_31462)
    
    # Assigning a Num to a Name (line 684):
    
    # Assigning a Num to a Name (line 684):
    int_31463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 8), 'int')
    # Assigning a type to the variable 'n' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'n', int_31463)
    
    # Assigning a List to a Name (line 685):
    
    # Assigning a List to a Name (line 685):
    
    # Obtaining an instance of the builtin type 'list' (line 685)
    list_31464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 685)
    # Adding element type (line 685)
    # Getting the type of 'a' (line 685)
    a_31465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 15), list_31464, a_31465)
    # Adding element type (line 685)
    # Getting the type of 'b' (line 685)
    b_31466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 19), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 15), list_31464, b_31466)
    
    # Assigning a type to the variable 'interval' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'interval', list_31464)
    
    # Assigning a BinOp to a Name (line 686):
    
    # Assigning a BinOp to a Name (line 686):
    # Getting the type of 'b' (line 686)
    b_31467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 15), 'b')
    # Getting the type of 'a' (line 686)
    a_31468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 19), 'a')
    # Applying the binary operator '-' (line 686)
    result_sub_31469 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 15), '-', b_31467, a_31468)
    
    # Assigning a type to the variable 'intrange' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'intrange', result_sub_31469)
    
    # Assigning a Call to a Name (line 687):
    
    # Assigning a Call to a Name (line 687):
    
    # Call to _difftrap(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'vfunc' (line 687)
    vfunc_31471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 23), 'vfunc', False)
    # Getting the type of 'interval' (line 687)
    interval_31472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 30), 'interval', False)
    # Getting the type of 'n' (line 687)
    n_31473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 40), 'n', False)
    # Processing the call keyword arguments (line 687)
    kwargs_31474 = {}
    # Getting the type of '_difftrap' (line 687)
    _difftrap_31470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 13), '_difftrap', False)
    # Calling _difftrap(args, kwargs) (line 687)
    _difftrap_call_result_31475 = invoke(stypy.reporting.localization.Localization(__file__, 687, 13), _difftrap_31470, *[vfunc_31471, interval_31472, n_31473], **kwargs_31474)
    
    # Assigning a type to the variable 'ordsum' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'ordsum', _difftrap_call_result_31475)
    
    # Assigning a BinOp to a Name (line 688):
    
    # Assigning a BinOp to a Name (line 688):
    # Getting the type of 'intrange' (line 688)
    intrange_31476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 13), 'intrange')
    # Getting the type of 'ordsum' (line 688)
    ordsum_31477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 24), 'ordsum')
    # Applying the binary operator '*' (line 688)
    result_mul_31478 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 13), '*', intrange_31476, ordsum_31477)
    
    # Assigning a type to the variable 'result' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'result', result_mul_31478)
    
    # Assigning a List to a Name (line 689):
    
    # Assigning a List to a Name (line 689):
    
    # Obtaining an instance of the builtin type 'list' (line 689)
    list_31479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 689)
    # Adding element type (line 689)
    
    # Obtaining an instance of the builtin type 'list' (line 689)
    list_31480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 689)
    # Adding element type (line 689)
    # Getting the type of 'result' (line 689)
    result_31481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 15), 'result')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 14), list_31480, result_31481)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 13), list_31479, list_31480)
    
    # Assigning a type to the variable 'resmat' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'resmat', list_31479)
    
    # Assigning a Attribute to a Name (line 690):
    
    # Assigning a Attribute to a Name (line 690):
    # Getting the type of 'np' (line 690)
    np_31482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 10), 'np')
    # Obtaining the member 'inf' of a type (line 690)
    inf_31483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 10), np_31482, 'inf')
    # Assigning a type to the variable 'err' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'err', inf_31483)
    
    # Assigning a Subscript to a Name (line 691):
    
    # Assigning a Subscript to a Name (line 691):
    
    # Obtaining the type of the subscript
    int_31484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 22), 'int')
    # Getting the type of 'resmat' (line 691)
    resmat_31485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 15), 'resmat')
    # Obtaining the member '__getitem__' of a type (line 691)
    getitem___31486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 15), resmat_31485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 691)
    subscript_call_result_31487 = invoke(stypy.reporting.localization.Localization(__file__, 691, 15), getitem___31486, int_31484)
    
    # Assigning a type to the variable 'last_row' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'last_row', subscript_call_result_31487)
    
    
    # Call to xrange(...): (line 692)
    # Processing the call arguments (line 692)
    int_31489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 20), 'int')
    # Getting the type of 'divmax' (line 692)
    divmax_31490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 23), 'divmax', False)
    int_31491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 30), 'int')
    # Applying the binary operator '+' (line 692)
    result_add_31492 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 23), '+', divmax_31490, int_31491)
    
    # Processing the call keyword arguments (line 692)
    kwargs_31493 = {}
    # Getting the type of 'xrange' (line 692)
    xrange_31488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 692)
    xrange_call_result_31494 = invoke(stypy.reporting.localization.Localization(__file__, 692, 13), xrange_31488, *[int_31489, result_add_31492], **kwargs_31493)
    
    # Testing the type of a for loop iterable (line 692)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 692, 4), xrange_call_result_31494)
    # Getting the type of the for loop variable (line 692)
    for_loop_var_31495 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 692, 4), xrange_call_result_31494)
    # Assigning a type to the variable 'i' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'i', for_loop_var_31495)
    # SSA begins for a for statement (line 692)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'n' (line 693)
    n_31496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'n')
    int_31497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 13), 'int')
    # Applying the binary operator '*=' (line 693)
    result_imul_31498 = python_operator(stypy.reporting.localization.Localization(__file__, 693, 8), '*=', n_31496, int_31497)
    # Assigning a type to the variable 'n' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'n', result_imul_31498)
    
    
    # Getting the type of 'ordsum' (line 694)
    ordsum_31499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'ordsum')
    
    # Call to _difftrap(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'vfunc' (line 694)
    vfunc_31501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), 'vfunc', False)
    # Getting the type of 'interval' (line 694)
    interval_31502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 35), 'interval', False)
    # Getting the type of 'n' (line 694)
    n_31503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 45), 'n', False)
    # Processing the call keyword arguments (line 694)
    kwargs_31504 = {}
    # Getting the type of '_difftrap' (line 694)
    _difftrap_31500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 18), '_difftrap', False)
    # Calling _difftrap(args, kwargs) (line 694)
    _difftrap_call_result_31505 = invoke(stypy.reporting.localization.Localization(__file__, 694, 18), _difftrap_31500, *[vfunc_31501, interval_31502, n_31503], **kwargs_31504)
    
    # Applying the binary operator '+=' (line 694)
    result_iadd_31506 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 8), '+=', ordsum_31499, _difftrap_call_result_31505)
    # Assigning a type to the variable 'ordsum' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'ordsum', result_iadd_31506)
    
    
    # Assigning a List to a Name (line 695):
    
    # Assigning a List to a Name (line 695):
    
    # Obtaining an instance of the builtin type 'list' (line 695)
    list_31507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 695)
    # Adding element type (line 695)
    # Getting the type of 'intrange' (line 695)
    intrange_31508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 15), 'intrange')
    # Getting the type of 'ordsum' (line 695)
    ordsum_31509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 26), 'ordsum')
    # Applying the binary operator '*' (line 695)
    result_mul_31510 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 15), '*', intrange_31508, ordsum_31509)
    
    # Getting the type of 'n' (line 695)
    n_31511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'n')
    # Applying the binary operator 'div' (line 695)
    result_div_31512 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 33), 'div', result_mul_31510, n_31511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 14), list_31507, result_div_31512)
    
    # Assigning a type to the variable 'row' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'row', list_31507)
    
    
    # Call to xrange(...): (line 696)
    # Processing the call arguments (line 696)
    # Getting the type of 'i' (line 696)
    i_31514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 24), 'i', False)
    # Processing the call keyword arguments (line 696)
    kwargs_31515 = {}
    # Getting the type of 'xrange' (line 696)
    xrange_31513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 696)
    xrange_call_result_31516 = invoke(stypy.reporting.localization.Localization(__file__, 696, 17), xrange_31513, *[i_31514], **kwargs_31515)
    
    # Testing the type of a for loop iterable (line 696)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 696, 8), xrange_call_result_31516)
    # Getting the type of the for loop variable (line 696)
    for_loop_var_31517 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 696, 8), xrange_call_result_31516)
    # Assigning a type to the variable 'k' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'k', for_loop_var_31517)
    # SSA begins for a for statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 697)
    # Processing the call arguments (line 697)
    
    # Call to _romberg_diff(...): (line 697)
    # Processing the call arguments (line 697)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 697)
    k_31521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 46), 'k', False)
    # Getting the type of 'last_row' (line 697)
    last_row_31522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 37), 'last_row', False)
    # Obtaining the member '__getitem__' of a type (line 697)
    getitem___31523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 37), last_row_31522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 697)
    subscript_call_result_31524 = invoke(stypy.reporting.localization.Localization(__file__, 697, 37), getitem___31523, k_31521)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 697)
    k_31525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 54), 'k', False)
    # Getting the type of 'row' (line 697)
    row_31526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 50), 'row', False)
    # Obtaining the member '__getitem__' of a type (line 697)
    getitem___31527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 50), row_31526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 697)
    subscript_call_result_31528 = invoke(stypy.reporting.localization.Localization(__file__, 697, 50), getitem___31527, k_31525)
    
    # Getting the type of 'k' (line 697)
    k_31529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 58), 'k', False)
    int_31530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 60), 'int')
    # Applying the binary operator '+' (line 697)
    result_add_31531 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 58), '+', k_31529, int_31530)
    
    # Processing the call keyword arguments (line 697)
    kwargs_31532 = {}
    # Getting the type of '_romberg_diff' (line 697)
    _romberg_diff_31520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 23), '_romberg_diff', False)
    # Calling _romberg_diff(args, kwargs) (line 697)
    _romberg_diff_call_result_31533 = invoke(stypy.reporting.localization.Localization(__file__, 697, 23), _romberg_diff_31520, *[subscript_call_result_31524, subscript_call_result_31528, result_add_31531], **kwargs_31532)
    
    # Processing the call keyword arguments (line 697)
    kwargs_31534 = {}
    # Getting the type of 'row' (line 697)
    row_31518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'row', False)
    # Obtaining the member 'append' of a type (line 697)
    append_31519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 12), row_31518, 'append')
    # Calling append(args, kwargs) (line 697)
    append_call_result_31535 = invoke(stypy.reporting.localization.Localization(__file__, 697, 12), append_31519, *[_romberg_diff_call_result_31533], **kwargs_31534)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 698):
    
    # Assigning a Subscript to a Name (line 698):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 698)
    i_31536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 21), 'i')
    # Getting the type of 'row' (line 698)
    row_31537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 17), 'row')
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___31538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 17), row_31537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_31539 = invoke(stypy.reporting.localization.Localization(__file__, 698, 17), getitem___31538, i_31536)
    
    # Assigning a type to the variable 'result' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'result', subscript_call_result_31539)
    
    # Assigning a Subscript to a Name (line 699):
    
    # Assigning a Subscript to a Name (line 699):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 699)
    i_31540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 30), 'i')
    int_31541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 32), 'int')
    # Applying the binary operator '-' (line 699)
    result_sub_31542 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 30), '-', i_31540, int_31541)
    
    # Getting the type of 'last_row' (line 699)
    last_row_31543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 21), 'last_row')
    # Obtaining the member '__getitem__' of a type (line 699)
    getitem___31544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 21), last_row_31543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 699)
    subscript_call_result_31545 = invoke(stypy.reporting.localization.Localization(__file__, 699, 21), getitem___31544, result_sub_31542)
    
    # Assigning a type to the variable 'lastresult' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'lastresult', subscript_call_result_31545)
    
    # Getting the type of 'show' (line 700)
    show_31546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 11), 'show')
    # Testing the type of an if condition (line 700)
    if_condition_31547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 8), show_31546)
    # Assigning a type to the variable 'if_condition_31547' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'if_condition_31547', if_condition_31547)
    # SSA begins for if statement (line 700)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'row' (line 701)
    row_31550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 26), 'row', False)
    # Processing the call keyword arguments (line 701)
    kwargs_31551 = {}
    # Getting the type of 'resmat' (line 701)
    resmat_31548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'resmat', False)
    # Obtaining the member 'append' of a type (line 701)
    append_31549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 12), resmat_31548, 'append')
    # Calling append(args, kwargs) (line 701)
    append_call_result_31552 = invoke(stypy.reporting.localization.Localization(__file__, 701, 12), append_31549, *[row_31550], **kwargs_31551)
    
    # SSA join for if statement (line 700)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 702):
    
    # Assigning a Call to a Name (line 702):
    
    # Call to abs(...): (line 702)
    # Processing the call arguments (line 702)
    # Getting the type of 'result' (line 702)
    result_31554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 18), 'result', False)
    # Getting the type of 'lastresult' (line 702)
    lastresult_31555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 27), 'lastresult', False)
    # Applying the binary operator '-' (line 702)
    result_sub_31556 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 18), '-', result_31554, lastresult_31555)
    
    # Processing the call keyword arguments (line 702)
    kwargs_31557 = {}
    # Getting the type of 'abs' (line 702)
    abs_31553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 14), 'abs', False)
    # Calling abs(args, kwargs) (line 702)
    abs_call_result_31558 = invoke(stypy.reporting.localization.Localization(__file__, 702, 14), abs_31553, *[result_sub_31556], **kwargs_31557)
    
    # Assigning a type to the variable 'err' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'err', abs_call_result_31558)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'err' (line 703)
    err_31559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 11), 'err')
    # Getting the type of 'tol' (line 703)
    tol_31560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 17), 'tol')
    # Applying the binary operator '<' (line 703)
    result_lt_31561 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), '<', err_31559, tol_31560)
    
    
    # Getting the type of 'err' (line 703)
    err_31562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 24), 'err')
    # Getting the type of 'rtol' (line 703)
    rtol_31563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 30), 'rtol')
    
    # Call to abs(...): (line 703)
    # Processing the call arguments (line 703)
    # Getting the type of 'result' (line 703)
    result_31565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 41), 'result', False)
    # Processing the call keyword arguments (line 703)
    kwargs_31566 = {}
    # Getting the type of 'abs' (line 703)
    abs_31564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 37), 'abs', False)
    # Calling abs(args, kwargs) (line 703)
    abs_call_result_31567 = invoke(stypy.reporting.localization.Localization(__file__, 703, 37), abs_31564, *[result_31565], **kwargs_31566)
    
    # Applying the binary operator '*' (line 703)
    result_mul_31568 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 30), '*', rtol_31563, abs_call_result_31567)
    
    # Applying the binary operator '<' (line 703)
    result_lt_31569 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 24), '<', err_31562, result_mul_31568)
    
    # Applying the binary operator 'or' (line 703)
    result_or_keyword_31570 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), 'or', result_lt_31561, result_lt_31569)
    
    # Testing the type of an if condition (line 703)
    if_condition_31571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 703, 8), result_or_keyword_31570)
    # Assigning a type to the variable 'if_condition_31571' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'if_condition_31571', if_condition_31571)
    # SSA begins for if statement (line 703)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 703)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 705):
    
    # Assigning a Name to a Name (line 705):
    # Getting the type of 'row' (line 705)
    row_31572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'row')
    # Assigning a type to the variable 'last_row' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'last_row', row_31572)
    # SSA branch for the else part of a for statement (line 692)
    module_type_store.open_ssa_branch('for loop else')
    
    # Call to warn(...): (line 707)
    # Processing the call arguments (line 707)
    str_31575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 12), 'str', 'divmax (%d) exceeded. Latest difference = %e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 708)
    tuple_31576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 708)
    # Adding element type (line 708)
    # Getting the type of 'divmax' (line 708)
    divmax_31577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 62), 'divmax', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 62), tuple_31576, divmax_31577)
    # Adding element type (line 708)
    # Getting the type of 'err' (line 708)
    err_31578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 70), 'err', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 62), tuple_31576, err_31578)
    
    # Applying the binary operator '%' (line 708)
    result_mod_31579 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 12), '%', str_31575, tuple_31576)
    
    # Getting the type of 'AccuracyWarning' (line 709)
    AccuracyWarning_31580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 12), 'AccuracyWarning', False)
    # Processing the call keyword arguments (line 707)
    kwargs_31581 = {}
    # Getting the type of 'warnings' (line 707)
    warnings_31573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 707)
    warn_31574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), warnings_31573, 'warn')
    # Calling warn(args, kwargs) (line 707)
    warn_call_result_31582 = invoke(stypy.reporting.localization.Localization(__file__, 707, 8), warn_31574, *[result_mod_31579, AccuracyWarning_31580], **kwargs_31581)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 711)
    show_31583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 7), 'show')
    # Testing the type of an if condition (line 711)
    if_condition_31584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 4), show_31583)
    # Assigning a type to the variable 'if_condition_31584' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'if_condition_31584', if_condition_31584)
    # SSA begins for if statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _printresmat(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'vfunc' (line 712)
    vfunc_31586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 21), 'vfunc', False)
    # Getting the type of 'interval' (line 712)
    interval_31587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 28), 'interval', False)
    # Getting the type of 'resmat' (line 712)
    resmat_31588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 38), 'resmat', False)
    # Processing the call keyword arguments (line 712)
    kwargs_31589 = {}
    # Getting the type of '_printresmat' (line 712)
    _printresmat_31585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), '_printresmat', False)
    # Calling _printresmat(args, kwargs) (line 712)
    _printresmat_call_result_31590 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), _printresmat_31585, *[vfunc_31586, interval_31587, resmat_31588], **kwargs_31589)
    
    # SSA join for if statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 713)
    result_31591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'stypy_return_type', result_31591)
    
    # ################# End of 'romberg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'romberg' in the type store
    # Getting the type of 'stypy_return_type' (line 596)
    stypy_return_type_31592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31592)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'romberg'
    return stypy_return_type_31592

# Assigning a type to the variable 'romberg' (line 596)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 0), 'romberg', romberg)

# Assigning a Dict to a Name (line 742):

# Assigning a Dict to a Name (line 742):

# Obtaining an instance of the builtin type 'dict' (line 742)
dict_31593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 742)
# Adding element type (key, value) (line 742)
int_31594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 743)
tuple_31595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 743)
# Adding element type (line 743)
int_31596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 8), tuple_31595, int_31596)
# Adding element type (line 743)
int_31597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 8), tuple_31595, int_31597)
# Adding element type (line 743)

# Obtaining an instance of the builtin type 'list' (line 743)
list_31598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 743)
# Adding element type (line 743)
int_31599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 12), list_31598, int_31599)
# Adding element type (line 743)
int_31600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 12), list_31598, int_31600)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 8), tuple_31595, list_31598)
# Adding element type (line 743)
int_31601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 8), tuple_31595, int_31601)
# Adding element type (line 743)
int_31602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 8), tuple_31595, int_31602)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31594, tuple_31595))
# Adding element type (key, value) (line 742)
int_31603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 744)
tuple_31604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 744)
# Adding element type (line 744)
int_31605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 8), tuple_31604, int_31605)
# Adding element type (line 744)
int_31606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 8), tuple_31604, int_31606)
# Adding element type (line 744)

# Obtaining an instance of the builtin type 'list' (line 744)
list_31607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 744)
# Adding element type (line 744)
int_31608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 12), list_31607, int_31608)
# Adding element type (line 744)
int_31609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 12), list_31607, int_31609)
# Adding element type (line 744)
int_31610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 12), list_31607, int_31610)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 8), tuple_31604, list_31607)
# Adding element type (line 744)
int_31611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 8), tuple_31604, int_31611)
# Adding element type (line 744)
int_31612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 8), tuple_31604, int_31612)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31603, tuple_31604))
# Adding element type (key, value) (line 742)
int_31613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 745)
tuple_31614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 745)
# Adding element type (line 745)
int_31615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 8), tuple_31614, int_31615)
# Adding element type (line 745)
int_31616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 8), tuple_31614, int_31616)
# Adding element type (line 745)

# Obtaining an instance of the builtin type 'list' (line 745)
list_31617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 745)
# Adding element type (line 745)
int_31618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 12), list_31617, int_31618)
# Adding element type (line 745)
int_31619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 12), list_31617, int_31619)
# Adding element type (line 745)
int_31620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 12), list_31617, int_31620)
# Adding element type (line 745)
int_31621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 12), list_31617, int_31621)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 8), tuple_31614, list_31617)
# Adding element type (line 745)
int_31622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 8), tuple_31614, int_31622)
# Adding element type (line 745)
int_31623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 8), tuple_31614, int_31623)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31613, tuple_31614))
# Adding element type (key, value) (line 742)
int_31624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 746)
tuple_31625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 746)
# Adding element type (line 746)
int_31626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 8), tuple_31625, int_31626)
# Adding element type (line 746)
int_31627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 8), tuple_31625, int_31627)
# Adding element type (line 746)

# Obtaining an instance of the builtin type 'list' (line 746)
list_31628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 746)
# Adding element type (line 746)
int_31629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 13), list_31628, int_31629)
# Adding element type (line 746)
int_31630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 13), list_31628, int_31630)
# Adding element type (line 746)
int_31631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 13), list_31628, int_31631)
# Adding element type (line 746)
int_31632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 13), list_31628, int_31632)
# Adding element type (line 746)
int_31633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 13), list_31628, int_31633)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 8), tuple_31625, list_31628)
# Adding element type (line 746)
int_31634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 8), tuple_31625, int_31634)
# Adding element type (line 746)
int_31635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 8), tuple_31625, int_31635)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31624, tuple_31625))
# Adding element type (key, value) (line 742)
int_31636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 747)
tuple_31637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 747)
# Adding element type (line 747)
int_31638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 8), tuple_31637, int_31638)
# Adding element type (line 747)
int_31639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 8), tuple_31637, int_31639)
# Adding element type (line 747)

# Obtaining an instance of the builtin type 'list' (line 747)
list_31640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 747)
# Adding element type (line 747)
int_31641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 14), list_31640, int_31641)
# Adding element type (line 747)
int_31642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 14), list_31640, int_31642)
# Adding element type (line 747)
int_31643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 14), list_31640, int_31643)
# Adding element type (line 747)
int_31644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 14), list_31640, int_31644)
# Adding element type (line 747)
int_31645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 14), list_31640, int_31645)
# Adding element type (line 747)
int_31646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 14), list_31640, int_31646)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 8), tuple_31637, list_31640)
# Adding element type (line 747)
int_31647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 8), tuple_31637, int_31647)
# Adding element type (line 747)
int_31648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 8), tuple_31637, int_31648)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31636, tuple_31637))
# Adding element type (key, value) (line 742)
int_31649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 748)
tuple_31650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 748)
# Adding element type (line 748)
int_31651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 8), tuple_31650, int_31651)
# Adding element type (line 748)
int_31652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 8), tuple_31650, int_31652)
# Adding element type (line 748)

# Obtaining an instance of the builtin type 'list' (line 748)
list_31653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 748)
# Adding element type (line 748)
int_31654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31654)
# Adding element type (line 748)
int_31655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31655)
# Adding element type (line 748)
int_31656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31656)
# Adding element type (line 748)
int_31657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31657)
# Adding element type (line 748)
int_31658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31658)
# Adding element type (line 748)
int_31659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31659)
# Adding element type (line 748)
int_31660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 14), list_31653, int_31660)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 8), tuple_31650, list_31653)
# Adding element type (line 748)
int_31661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 8), tuple_31650, int_31661)
# Adding element type (line 748)
int_31662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 8), tuple_31650, int_31662)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31649, tuple_31650))
# Adding element type (key, value) (line 742)
int_31663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 749)
tuple_31664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 749)
# Adding element type (line 749)
int_31665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 8), tuple_31664, int_31665)
# Adding element type (line 749)
int_31666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 8), tuple_31664, int_31666)
# Adding element type (line 749)

# Obtaining an instance of the builtin type 'list' (line 749)
list_31667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 749)
# Adding element type (line 749)
int_31668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31668)
# Adding element type (line 749)
int_31669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31669)
# Adding element type (line 749)
int_31670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31670)
# Adding element type (line 749)
int_31671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31671)
# Adding element type (line 749)
int_31672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31672)
# Adding element type (line 749)
int_31673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31673)
# Adding element type (line 749)
int_31674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31674)
# Adding element type (line 749)
int_31675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_31667, int_31675)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 8), tuple_31664, list_31667)
# Adding element type (line 749)
int_31676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 8), tuple_31664, int_31676)
# Adding element type (line 749)
int_31677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 8), tuple_31664, int_31677)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31663, tuple_31664))
# Adding element type (key, value) (line 742)
int_31678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 750)
tuple_31679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 750)
# Adding element type (line 750)
int_31680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 8), tuple_31679, int_31680)
# Adding element type (line 750)
int_31681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 8), tuple_31679, int_31681)
# Adding element type (line 750)

# Obtaining an instance of the builtin type 'list' (line 750)
list_31682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 750)
# Adding element type (line 750)
int_31683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31683)
# Adding element type (line 750)
int_31684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31684)
# Adding element type (line 750)
int_31685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31685)
# Adding element type (line 750)
int_31686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31686)
# Adding element type (line 750)
int_31687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31687)
# Adding element type (line 750)
int_31688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31688)
# Adding element type (line 750)
int_31689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31689)
# Adding element type (line 750)
int_31690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31690)
# Adding element type (line 750)
int_31691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 16), list_31682, int_31691)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 8), tuple_31679, list_31682)
# Adding element type (line 750)
int_31692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 8), tuple_31679, int_31692)
# Adding element type (line 750)
int_31693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 8), tuple_31679, int_31693)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31678, tuple_31679))
# Adding element type (key, value) (line 742)
int_31694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 752)
tuple_31695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 752)
# Adding element type (line 752)
int_31696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 8), tuple_31695, int_31696)
# Adding element type (line 752)
int_31697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 8), tuple_31695, int_31697)
# Adding element type (line 752)

# Obtaining an instance of the builtin type 'list' (line 752)
list_31698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 752)
# Adding element type (line 752)
int_31699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31699)
# Adding element type (line 752)
int_31700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31700)
# Adding element type (line 752)
int_31701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31701)
# Adding element type (line 752)
int_31702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31702)
# Adding element type (line 752)
int_31703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31703)
# Adding element type (line 752)
int_31704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31704)
# Adding element type (line 752)
int_31705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31705)
# Adding element type (line 752)
int_31706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31706)
# Adding element type (line 752)
int_31707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31707)
# Adding element type (line 752)
int_31708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 16), list_31698, int_31708)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 8), tuple_31695, list_31698)
# Adding element type (line 752)
int_31709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 8), tuple_31695, int_31709)
# Adding element type (line 752)
int_31710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 8), tuple_31695, int_31710)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31694, tuple_31695))
# Adding element type (key, value) (line 742)
int_31711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 754)
tuple_31712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 754)
# Adding element type (line 754)
int_31713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 9), tuple_31712, int_31713)
# Adding element type (line 754)
int_31714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 9), tuple_31712, int_31714)
# Adding element type (line 754)

# Obtaining an instance of the builtin type 'list' (line 754)
list_31715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 754)
# Adding element type (line 754)
int_31716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31716)
# Adding element type (line 754)
int_31717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31717)
# Adding element type (line 754)
int_31718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31718)
# Adding element type (line 754)
int_31719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31719)
# Adding element type (line 754)
int_31720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31720)
# Adding element type (line 754)
int_31721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31721)
# Adding element type (line 754)
int_31722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31722)
# Adding element type (line 754)
int_31723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31723)
# Adding element type (line 754)
int_31724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31724)
# Adding element type (line 754)
int_31725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31725)
# Adding element type (line 754)
int_31726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 18), list_31715, int_31726)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 9), tuple_31712, list_31715)
# Adding element type (line 754)
int_31727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 9), tuple_31712, int_31727)
# Adding element type (line 754)
int_31728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 9), tuple_31712, int_31728)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31711, tuple_31712))
# Adding element type (key, value) (line 742)
int_31729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 757)
tuple_31730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 757)
# Adding element type (line 757)
int_31731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 9), tuple_31730, int_31731)
# Adding element type (line 757)
int_31732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 9), tuple_31730, int_31732)
# Adding element type (line 757)

# Obtaining an instance of the builtin type 'list' (line 757)
list_31733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 757)
# Adding element type (line 757)
int_31734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31734)
# Adding element type (line 757)
int_31735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31735)
# Adding element type (line 757)
int_31736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31736)
# Adding element type (line 757)
int_31737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31737)
# Adding element type (line 757)
int_31738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31738)
# Adding element type (line 757)
int_31739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31739)
# Adding element type (line 757)
int_31740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31740)
# Adding element type (line 757)
int_31741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31741)
# Adding element type (line 757)
int_31742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31742)
# Adding element type (line 757)
int_31743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31743)
# Adding element type (line 757)
int_31744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31744)
# Adding element type (line 757)
int_31745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 21), list_31733, int_31745)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 9), tuple_31730, list_31733)
# Adding element type (line 757)
long_31746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 41), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 9), tuple_31730, long_31746)
# Adding element type (line 757)
long_31747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 54), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 9), tuple_31730, long_31747)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31729, tuple_31730))
# Adding element type (key, value) (line 742)
int_31748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 760)
tuple_31749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 760)
# Adding element type (line 760)
int_31750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 9), tuple_31749, int_31750)
# Adding element type (line 760)
int_31751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 9), tuple_31749, int_31751)
# Adding element type (line 760)

# Obtaining an instance of the builtin type 'list' (line 760)
list_31752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 760)
# Adding element type (line 760)
int_31753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31753)
# Adding element type (line 760)
int_31754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31754)
# Adding element type (line 760)
int_31755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31755)
# Adding element type (line 760)
int_31756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31756)
# Adding element type (line 760)
int_31757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31757)
# Adding element type (line 760)
int_31758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31758)
# Adding element type (line 760)
int_31759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31759)
# Adding element type (line 760)
int_31760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31760)
# Adding element type (line 760)
int_31761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31761)
# Adding element type (line 760)
int_31762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31762)
# Adding element type (line 760)
int_31763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31763)
# Adding element type (line 760)
int_31764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31764)
# Adding element type (line 760)
int_31765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 21), list_31752, int_31765)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 9), tuple_31749, list_31752)
# Adding element type (line 760)
int_31766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 9), tuple_31749, int_31766)
# Adding element type (line 760)
int_31767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 9), tuple_31749, int_31767)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31748, tuple_31749))
# Adding element type (key, value) (line 742)
int_31768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 763)
tuple_31769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 763)
# Adding element type (line 763)
int_31770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 9), tuple_31769, int_31770)
# Adding element type (line 763)
long_31771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 13), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 9), tuple_31769, long_31771)
# Adding element type (line 763)

# Obtaining an instance of the builtin type 'list' (line 763)
list_31772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 763)
# Adding element type (line 763)
long_31773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 27), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31773)
# Adding element type (line 763)
long_31774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 39), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31774)
# Adding element type (line 763)
long_31775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 52), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31775)
# Adding element type (line 763)
long_31776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 27), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31776)
# Adding element type (line 763)
long_31777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 40), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31777)
# Adding element type (line 763)
long_31778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 54), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31778)
# Adding element type (line 763)
long_31779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 27), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31779)
# Adding element type (line 763)
long_31780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 40), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31780)
# Adding element type (line 763)
long_31781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 53), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31781)
# Adding element type (line 763)
long_31782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 27), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31782)
# Adding element type (line 763)
long_31783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 41), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31783)
# Adding element type (line 763)
long_31784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 54), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31784)
# Adding element type (line 763)
long_31785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 27), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31785)
# Adding element type (line 763)
long_31786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 39), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 26), list_31772, long_31786)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 9), tuple_31769, list_31772)
# Adding element type (line 763)
long_31787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 52), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 9), tuple_31769, long_31787)
# Adding element type (line 763)
long_31788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 9), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 9), tuple_31769, long_31788)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31768, tuple_31769))
# Adding element type (key, value) (line 742)
int_31789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 769)
tuple_31790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 769)
# Adding element type (line 769)
int_31791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 9), tuple_31790, int_31791)
# Adding element type (line 769)
long_31792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 12), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 9), tuple_31790, long_31792)
# Adding element type (line 769)

# Obtaining an instance of the builtin type 'list' (line 769)
list_31793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 769)
# Adding element type (line 769)
int_31794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, int_31794)
# Adding element type (line 769)
int_31795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, int_31795)
# Adding element type (line 769)
int_31796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, int_31796)
# Adding element type (line 769)
long_31797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 55), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31797)
# Adding element type (line 769)
long_31798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 25), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31798)
# Adding element type (line 769)
long_31799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 37), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31799)
# Adding element type (line 769)
long_31800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 49), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31800)
# Adding element type (line 769)
long_31801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 62), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31801)
# Adding element type (line 769)
long_31802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 25), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31802)
# Adding element type (line 769)
long_31803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 38), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31803)
# Adding element type (line 769)
long_31804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 50), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31804)
# Adding element type (line 769)
long_31805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 62), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, long_31805)
# Adding element type (line 769)
int_31806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, int_31806)
# Adding element type (line 769)
int_31807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, int_31807)
# Adding element type (line 769)
int_31808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_31793, int_31808)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 9), tuple_31790, list_31793)
# Adding element type (line 769)
long_31809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 57), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 9), tuple_31790, long_31809)
# Adding element type (line 769)
long_31810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 9), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 9), tuple_31790, long_31810)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 17), dict_31593, (int_31789, tuple_31790))

# Assigning a type to the variable '_builtincoeffs' (line 742)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 0), '_builtincoeffs', dict_31593)

@norecursion
def newton_cotes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_31811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 27), 'int')
    defaults = [int_31811]
    # Create a new context for function 'newton_cotes'
    module_type_store = module_type_store.open_function_context('newton_cotes', 777, 0, False)
    
    # Passed parameters checking function
    newton_cotes.stypy_localization = localization
    newton_cotes.stypy_type_of_self = None
    newton_cotes.stypy_type_store = module_type_store
    newton_cotes.stypy_function_name = 'newton_cotes'
    newton_cotes.stypy_param_names_list = ['rn', 'equal']
    newton_cotes.stypy_varargs_param_name = None
    newton_cotes.stypy_kwargs_param_name = None
    newton_cotes.stypy_call_defaults = defaults
    newton_cotes.stypy_call_varargs = varargs
    newton_cotes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'newton_cotes', ['rn', 'equal'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'newton_cotes', localization, ['rn', 'equal'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'newton_cotes(...)' code ##################

    str_31812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, (-1)), 'str', '\n    Return weights and error coefficient for Newton-Cotes integration.\n\n    Suppose we have (N+1) samples of f at the positions\n    x_0, x_1, ..., x_N.  Then an N-point Newton-Cotes formula for the\n    integral between x_0 and x_N is:\n\n    :math:`\\int_{x_0}^{x_N} f(x)dx = \\Delta x \\sum_{i=0}^{N} a_i f(x_i)\n    + B_N (\\Delta x)^{N+2} f^{N+1} (\\xi)`\n\n    where :math:`\\xi \\in [x_0,x_N]`\n    and :math:`\\Delta x = \\frac{x_N-x_0}{N}` is the average samples spacing.\n\n    If the samples are equally-spaced and N is even, then the error\n    term is :math:`B_N (\\Delta x)^{N+3} f^{N+2}(\\xi)`.\n\n    Parameters\n    ----------\n    rn : int\n        The integer order for equally-spaced data or the relative positions of\n        the samples with the first sample at 0 and the last at N, where N+1 is\n        the length of `rn`.  N is the order of the Newton-Cotes integration.\n    equal : int, optional\n        Set to 1 to enforce equally spaced data.\n\n    Returns\n    -------\n    an : ndarray\n        1-D array of weights to apply to the function at the provided sample\n        positions.\n    B : float\n        Error coefficient.\n\n    Notes\n    -----\n    Normally, the Newton-Cotes rules are used on smaller integration\n    regions and a composite rule is used to return the total integral.\n\n    ')
    
    
    # SSA begins for try-except statement (line 817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a BinOp to a Name (line 818):
    
    # Assigning a BinOp to a Name (line 818):
    
    # Call to len(...): (line 818)
    # Processing the call arguments (line 818)
    # Getting the type of 'rn' (line 818)
    rn_31814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 16), 'rn', False)
    # Processing the call keyword arguments (line 818)
    kwargs_31815 = {}
    # Getting the type of 'len' (line 818)
    len_31813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'len', False)
    # Calling len(args, kwargs) (line 818)
    len_call_result_31816 = invoke(stypy.reporting.localization.Localization(__file__, 818, 12), len_31813, *[rn_31814], **kwargs_31815)
    
    int_31817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 20), 'int')
    # Applying the binary operator '-' (line 818)
    result_sub_31818 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 12), '-', len_call_result_31816, int_31817)
    
    # Assigning a type to the variable 'N' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), 'N', result_sub_31818)
    
    # Getting the type of 'equal' (line 819)
    equal_31819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 11), 'equal')
    # Testing the type of an if condition (line 819)
    if_condition_31820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 819, 8), equal_31819)
    # Assigning a type to the variable 'if_condition_31820' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'if_condition_31820', if_condition_31820)
    # SSA begins for if statement (line 819)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 820):
    
    # Assigning a Call to a Name (line 820):
    
    # Call to arange(...): (line 820)
    # Processing the call arguments (line 820)
    # Getting the type of 'N' (line 820)
    N_31823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 27), 'N', False)
    int_31824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 29), 'int')
    # Applying the binary operator '+' (line 820)
    result_add_31825 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 27), '+', N_31823, int_31824)
    
    # Processing the call keyword arguments (line 820)
    kwargs_31826 = {}
    # Getting the type of 'np' (line 820)
    np_31821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 17), 'np', False)
    # Obtaining the member 'arange' of a type (line 820)
    arange_31822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 17), np_31821, 'arange')
    # Calling arange(args, kwargs) (line 820)
    arange_call_result_31827 = invoke(stypy.reporting.localization.Localization(__file__, 820, 17), arange_31822, *[result_add_31825], **kwargs_31826)
    
    # Assigning a type to the variable 'rn' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'rn', arange_call_result_31827)
    # SSA branch for the else part of an if statement (line 819)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to all(...): (line 821)
    # Processing the call arguments (line 821)
    
    
    # Call to diff(...): (line 821)
    # Processing the call arguments (line 821)
    # Getting the type of 'rn' (line 821)
    rn_31832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 28), 'rn', False)
    # Processing the call keyword arguments (line 821)
    kwargs_31833 = {}
    # Getting the type of 'np' (line 821)
    np_31830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 20), 'np', False)
    # Obtaining the member 'diff' of a type (line 821)
    diff_31831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 20), np_31830, 'diff')
    # Calling diff(args, kwargs) (line 821)
    diff_call_result_31834 = invoke(stypy.reporting.localization.Localization(__file__, 821, 20), diff_31831, *[rn_31832], **kwargs_31833)
    
    int_31835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 35), 'int')
    # Applying the binary operator '==' (line 821)
    result_eq_31836 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 20), '==', diff_call_result_31834, int_31835)
    
    # Processing the call keyword arguments (line 821)
    kwargs_31837 = {}
    # Getting the type of 'np' (line 821)
    np_31828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 13), 'np', False)
    # Obtaining the member 'all' of a type (line 821)
    all_31829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 13), np_31828, 'all')
    # Calling all(args, kwargs) (line 821)
    all_call_result_31838 = invoke(stypy.reporting.localization.Localization(__file__, 821, 13), all_31829, *[result_eq_31836], **kwargs_31837)
    
    # Testing the type of an if condition (line 821)
    if_condition_31839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 13), all_call_result_31838)
    # Assigning a type to the variable 'if_condition_31839' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 13), 'if_condition_31839', if_condition_31839)
    # SSA begins for if statement (line 821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 822):
    
    # Assigning a Num to a Name (line 822):
    int_31840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 20), 'int')
    # Assigning a type to the variable 'equal' (line 822)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 12), 'equal', int_31840)
    # SSA join for if statement (line 821)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 819)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 817)
    # SSA branch for the except '<any exception>' branch of a try statement (line 817)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 824):
    
    # Assigning a Name to a Name (line 824):
    # Getting the type of 'rn' (line 824)
    rn_31841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 12), 'rn')
    # Assigning a type to the variable 'N' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'N', rn_31841)
    
    # Assigning a Call to a Name (line 825):
    
    # Assigning a Call to a Name (line 825):
    
    # Call to arange(...): (line 825)
    # Processing the call arguments (line 825)
    # Getting the type of 'N' (line 825)
    N_31844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 23), 'N', False)
    int_31845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 25), 'int')
    # Applying the binary operator '+' (line 825)
    result_add_31846 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 23), '+', N_31844, int_31845)
    
    # Processing the call keyword arguments (line 825)
    kwargs_31847 = {}
    # Getting the type of 'np' (line 825)
    np_31842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 13), 'np', False)
    # Obtaining the member 'arange' of a type (line 825)
    arange_31843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 13), np_31842, 'arange')
    # Calling arange(args, kwargs) (line 825)
    arange_call_result_31848 = invoke(stypy.reporting.localization.Localization(__file__, 825, 13), arange_31843, *[result_add_31846], **kwargs_31847)
    
    # Assigning a type to the variable 'rn' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 8), 'rn', arange_call_result_31848)
    
    # Assigning a Num to a Name (line 826):
    
    # Assigning a Num to a Name (line 826):
    int_31849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 16), 'int')
    # Assigning a type to the variable 'equal' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'equal', int_31849)
    # SSA join for try-except statement (line 817)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'equal' (line 828)
    equal_31850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 7), 'equal')
    
    # Getting the type of 'N' (line 828)
    N_31851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 17), 'N')
    # Getting the type of '_builtincoeffs' (line 828)
    _builtincoeffs_31852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 22), '_builtincoeffs')
    # Applying the binary operator 'in' (line 828)
    result_contains_31853 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 17), 'in', N_31851, _builtincoeffs_31852)
    
    # Applying the binary operator 'and' (line 828)
    result_and_keyword_31854 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 7), 'and', equal_31850, result_contains_31853)
    
    # Testing the type of an if condition (line 828)
    if_condition_31855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 4), result_and_keyword_31854)
    # Assigning a type to the variable 'if_condition_31855' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'if_condition_31855', if_condition_31855)
    # SSA begins for if statement (line 828)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 829):
    
    # Assigning a Subscript to a Name (line 829):
    
    # Obtaining the type of the subscript
    int_31856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 829)
    N_31857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 44), 'N')
    # Getting the type of '_builtincoeffs' (line 829)
    _builtincoeffs_31858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 29), '_builtincoeffs')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 29), _builtincoeffs_31858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31860 = invoke(stypy.reporting.localization.Localization(__file__, 829, 29), getitem___31859, N_31857)
    
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 8), subscript_call_result_31860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31862 = invoke(stypy.reporting.localization.Localization(__file__, 829, 8), getitem___31861, int_31856)
    
    # Assigning a type to the variable 'tuple_var_assignment_29994' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29994', subscript_call_result_31862)
    
    # Assigning a Subscript to a Name (line 829):
    
    # Obtaining the type of the subscript
    int_31863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 829)
    N_31864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 44), 'N')
    # Getting the type of '_builtincoeffs' (line 829)
    _builtincoeffs_31865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 29), '_builtincoeffs')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 29), _builtincoeffs_31865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31867 = invoke(stypy.reporting.localization.Localization(__file__, 829, 29), getitem___31866, N_31864)
    
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 8), subscript_call_result_31867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31869 = invoke(stypy.reporting.localization.Localization(__file__, 829, 8), getitem___31868, int_31863)
    
    # Assigning a type to the variable 'tuple_var_assignment_29995' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29995', subscript_call_result_31869)
    
    # Assigning a Subscript to a Name (line 829):
    
    # Obtaining the type of the subscript
    int_31870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 829)
    N_31871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 44), 'N')
    # Getting the type of '_builtincoeffs' (line 829)
    _builtincoeffs_31872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 29), '_builtincoeffs')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 29), _builtincoeffs_31872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31874 = invoke(stypy.reporting.localization.Localization(__file__, 829, 29), getitem___31873, N_31871)
    
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 8), subscript_call_result_31874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31876 = invoke(stypy.reporting.localization.Localization(__file__, 829, 8), getitem___31875, int_31870)
    
    # Assigning a type to the variable 'tuple_var_assignment_29996' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29996', subscript_call_result_31876)
    
    # Assigning a Subscript to a Name (line 829):
    
    # Obtaining the type of the subscript
    int_31877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 829)
    N_31878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 44), 'N')
    # Getting the type of '_builtincoeffs' (line 829)
    _builtincoeffs_31879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 29), '_builtincoeffs')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 29), _builtincoeffs_31879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31881 = invoke(stypy.reporting.localization.Localization(__file__, 829, 29), getitem___31880, N_31878)
    
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 8), subscript_call_result_31881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31883 = invoke(stypy.reporting.localization.Localization(__file__, 829, 8), getitem___31882, int_31877)
    
    # Assigning a type to the variable 'tuple_var_assignment_29997' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29997', subscript_call_result_31883)
    
    # Assigning a Subscript to a Name (line 829):
    
    # Obtaining the type of the subscript
    int_31884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 829)
    N_31885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 44), 'N')
    # Getting the type of '_builtincoeffs' (line 829)
    _builtincoeffs_31886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 29), '_builtincoeffs')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 29), _builtincoeffs_31886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31888 = invoke(stypy.reporting.localization.Localization(__file__, 829, 29), getitem___31887, N_31885)
    
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___31889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 8), subscript_call_result_31888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_31890 = invoke(stypy.reporting.localization.Localization(__file__, 829, 8), getitem___31889, int_31884)
    
    # Assigning a type to the variable 'tuple_var_assignment_29998' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29998', subscript_call_result_31890)
    
    # Assigning a Name to a Name (line 829):
    # Getting the type of 'tuple_var_assignment_29994' (line 829)
    tuple_var_assignment_29994_31891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29994')
    # Assigning a type to the variable 'na' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'na', tuple_var_assignment_29994_31891)
    
    # Assigning a Name to a Name (line 829):
    # Getting the type of 'tuple_var_assignment_29995' (line 829)
    tuple_var_assignment_29995_31892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29995')
    # Assigning a type to the variable 'da' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 12), 'da', tuple_var_assignment_29995_31892)
    
    # Assigning a Name to a Name (line 829):
    # Getting the type of 'tuple_var_assignment_29996' (line 829)
    tuple_var_assignment_29996_31893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29996')
    # Assigning a type to the variable 'vi' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 16), 'vi', tuple_var_assignment_29996_31893)
    
    # Assigning a Name to a Name (line 829):
    # Getting the type of 'tuple_var_assignment_29997' (line 829)
    tuple_var_assignment_29997_31894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29997')
    # Assigning a type to the variable 'nb' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 20), 'nb', tuple_var_assignment_29997_31894)
    
    # Assigning a Name to a Name (line 829):
    # Getting the type of 'tuple_var_assignment_29998' (line 829)
    tuple_var_assignment_29998_31895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'tuple_var_assignment_29998')
    # Assigning a type to the variable 'db' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 24), 'db', tuple_var_assignment_29998_31895)
    
    # Assigning a BinOp to a Name (line 830):
    
    # Assigning a BinOp to a Name (line 830):
    # Getting the type of 'na' (line 830)
    na_31896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 13), 'na')
    
    # Call to array(...): (line 830)
    # Processing the call arguments (line 830)
    # Getting the type of 'vi' (line 830)
    vi_31899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 27), 'vi', False)
    # Processing the call keyword arguments (line 830)
    # Getting the type of 'float' (line 830)
    float_31900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 37), 'float', False)
    keyword_31901 = float_31900
    kwargs_31902 = {'dtype': keyword_31901}
    # Getting the type of 'np' (line 830)
    np_31897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 830)
    array_31898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 18), np_31897, 'array')
    # Calling array(args, kwargs) (line 830)
    array_call_result_31903 = invoke(stypy.reporting.localization.Localization(__file__, 830, 18), array_31898, *[vi_31899], **kwargs_31902)
    
    # Applying the binary operator '*' (line 830)
    result_mul_31904 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 13), '*', na_31896, array_call_result_31903)
    
    # Getting the type of 'da' (line 830)
    da_31905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 46), 'da')
    # Applying the binary operator 'div' (line 830)
    result_div_31906 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 44), 'div', result_mul_31904, da_31905)
    
    # Assigning a type to the variable 'an' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'an', result_div_31906)
    
    # Obtaining an instance of the builtin type 'tuple' (line 831)
    tuple_31907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 831)
    # Adding element type (line 831)
    # Getting the type of 'an' (line 831)
    an_31908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'an')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 15), tuple_31907, an_31908)
    # Adding element type (line 831)
    
    # Call to float(...): (line 831)
    # Processing the call arguments (line 831)
    # Getting the type of 'nb' (line 831)
    nb_31910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 25), 'nb', False)
    # Processing the call keyword arguments (line 831)
    kwargs_31911 = {}
    # Getting the type of 'float' (line 831)
    float_31909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 19), 'float', False)
    # Calling float(args, kwargs) (line 831)
    float_call_result_31912 = invoke(stypy.reporting.localization.Localization(__file__, 831, 19), float_31909, *[nb_31910], **kwargs_31911)
    
    # Getting the type of 'db' (line 831)
    db_31913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 29), 'db')
    # Applying the binary operator 'div' (line 831)
    result_div_31914 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 19), 'div', float_call_result_31912, db_31913)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 15), tuple_31907, result_div_31914)
    
    # Assigning a type to the variable 'stypy_return_type' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'stypy_return_type', tuple_31907)
    # SSA join for if statement (line 828)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_31915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 11), 'int')
    # Getting the type of 'rn' (line 833)
    rn_31916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'rn')
    # Obtaining the member '__getitem__' of a type (line 833)
    getitem___31917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 8), rn_31916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 833)
    subscript_call_result_31918 = invoke(stypy.reporting.localization.Localization(__file__, 833, 8), getitem___31917, int_31915)
    
    int_31919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 17), 'int')
    # Applying the binary operator '!=' (line 833)
    result_ne_31920 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 8), '!=', subscript_call_result_31918, int_31919)
    
    
    
    # Obtaining the type of the subscript
    int_31921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 27), 'int')
    # Getting the type of 'rn' (line 833)
    rn_31922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 24), 'rn')
    # Obtaining the member '__getitem__' of a type (line 833)
    getitem___31923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 24), rn_31922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 833)
    subscript_call_result_31924 = invoke(stypy.reporting.localization.Localization(__file__, 833, 24), getitem___31923, int_31921)
    
    # Getting the type of 'N' (line 833)
    N_31925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 34), 'N')
    # Applying the binary operator '!=' (line 833)
    result_ne_31926 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 24), '!=', subscript_call_result_31924, N_31925)
    
    # Applying the binary operator 'or' (line 833)
    result_or_keyword_31927 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 7), 'or', result_ne_31920, result_ne_31926)
    
    # Testing the type of an if condition (line 833)
    if_condition_31928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 833, 4), result_or_keyword_31927)
    # Assigning a type to the variable 'if_condition_31928' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'if_condition_31928', if_condition_31928)
    # SSA begins for if statement (line 833)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 834)
    # Processing the call arguments (line 834)
    str_31930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 25), 'str', 'The sample positions must start at 0 and end at N')
    # Processing the call keyword arguments (line 834)
    kwargs_31931 = {}
    # Getting the type of 'ValueError' (line 834)
    ValueError_31929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 834)
    ValueError_call_result_31932 = invoke(stypy.reporting.localization.Localization(__file__, 834, 14), ValueError_31929, *[str_31930], **kwargs_31931)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 834, 8), ValueError_call_result_31932, 'raise parameter', BaseException)
    # SSA join for if statement (line 833)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 836):
    
    # Assigning a BinOp to a Name (line 836):
    # Getting the type of 'rn' (line 836)
    rn_31933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 9), 'rn')
    
    # Call to float(...): (line 836)
    # Processing the call arguments (line 836)
    # Getting the type of 'N' (line 836)
    N_31935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 20), 'N', False)
    # Processing the call keyword arguments (line 836)
    kwargs_31936 = {}
    # Getting the type of 'float' (line 836)
    float_31934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 14), 'float', False)
    # Calling float(args, kwargs) (line 836)
    float_call_result_31937 = invoke(stypy.reporting.localization.Localization(__file__, 836, 14), float_31934, *[N_31935], **kwargs_31936)
    
    # Applying the binary operator 'div' (line 836)
    result_div_31938 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 9), 'div', rn_31933, float_call_result_31937)
    
    # Assigning a type to the variable 'yi' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 4), 'yi', result_div_31938)
    
    # Assigning a BinOp to a Name (line 837):
    
    # Assigning a BinOp to a Name (line 837):
    int_31939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 9), 'int')
    # Getting the type of 'yi' (line 837)
    yi_31940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 13), 'yi')
    # Applying the binary operator '*' (line 837)
    result_mul_31941 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 9), '*', int_31939, yi_31940)
    
    int_31942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 18), 'int')
    # Applying the binary operator '-' (line 837)
    result_sub_31943 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 9), '-', result_mul_31941, int_31942)
    
    # Assigning a type to the variable 'ti' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'ti', result_sub_31943)
    
    # Assigning a Call to a Name (line 838):
    
    # Assigning a Call to a Name (line 838):
    
    # Call to arange(...): (line 838)
    # Processing the call arguments (line 838)
    # Getting the type of 'N' (line 838)
    N_31946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 21), 'N', False)
    int_31947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 23), 'int')
    # Applying the binary operator '+' (line 838)
    result_add_31948 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 21), '+', N_31946, int_31947)
    
    # Processing the call keyword arguments (line 838)
    kwargs_31949 = {}
    # Getting the type of 'np' (line 838)
    np_31944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 838)
    arange_31945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 11), np_31944, 'arange')
    # Calling arange(args, kwargs) (line 838)
    arange_call_result_31950 = invoke(stypy.reporting.localization.Localization(__file__, 838, 11), arange_31945, *[result_add_31948], **kwargs_31949)
    
    # Assigning a type to the variable 'nvec' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'nvec', arange_call_result_31950)
    
    # Assigning a BinOp to a Name (line 839):
    
    # Assigning a BinOp to a Name (line 839):
    # Getting the type of 'ti' (line 839)
    ti_31951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'ti')
    
    # Obtaining the type of the subscript
    slice_31952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 839, 14), None, None, None)
    # Getting the type of 'np' (line 839)
    np_31953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 22), 'np')
    # Obtaining the member 'newaxis' of a type (line 839)
    newaxis_31954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 22), np_31953, 'newaxis')
    # Getting the type of 'nvec' (line 839)
    nvec_31955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 14), 'nvec')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___31956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 14), nvec_31955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_31957 = invoke(stypy.reporting.localization.Localization(__file__, 839, 14), getitem___31956, (slice_31952, newaxis_31954))
    
    # Applying the binary operator '**' (line 839)
    result_pow_31958 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 8), '**', ti_31951, subscript_call_result_31957)
    
    # Assigning a type to the variable 'C' (line 839)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 4), 'C', result_pow_31958)
    
    # Assigning a Call to a Name (line 840):
    
    # Assigning a Call to a Name (line 840):
    
    # Call to inv(...): (line 840)
    # Processing the call arguments (line 840)
    # Getting the type of 'C' (line 840)
    C_31962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 25), 'C', False)
    # Processing the call keyword arguments (line 840)
    kwargs_31963 = {}
    # Getting the type of 'np' (line 840)
    np_31959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 11), 'np', False)
    # Obtaining the member 'linalg' of a type (line 840)
    linalg_31960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 11), np_31959, 'linalg')
    # Obtaining the member 'inv' of a type (line 840)
    inv_31961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 11), linalg_31960, 'inv')
    # Calling inv(args, kwargs) (line 840)
    inv_call_result_31964 = invoke(stypy.reporting.localization.Localization(__file__, 840, 11), inv_31961, *[C_31962], **kwargs_31963)
    
    # Assigning a type to the variable 'Cinv' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 4), 'Cinv', inv_call_result_31964)
    
    
    # Call to range(...): (line 842)
    # Processing the call arguments (line 842)
    int_31966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 19), 'int')
    # Processing the call keyword arguments (line 842)
    kwargs_31967 = {}
    # Getting the type of 'range' (line 842)
    range_31965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 13), 'range', False)
    # Calling range(args, kwargs) (line 842)
    range_call_result_31968 = invoke(stypy.reporting.localization.Localization(__file__, 842, 13), range_31965, *[int_31966], **kwargs_31967)
    
    # Testing the type of a for loop iterable (line 842)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 842, 4), range_call_result_31968)
    # Getting the type of the for loop variable (line 842)
    for_loop_var_31969 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 842, 4), range_call_result_31968)
    # Assigning a type to the variable 'i' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'i', for_loop_var_31969)
    # SSA begins for a for statement (line 842)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 843):
    
    # Assigning a BinOp to a Name (line 843):
    int_31970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 15), 'int')
    # Getting the type of 'Cinv' (line 843)
    Cinv_31971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 17), 'Cinv')
    # Applying the binary operator '*' (line 843)
    result_mul_31972 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 15), '*', int_31970, Cinv_31971)
    
    
    # Call to dot(...): (line 843)
    # Processing the call arguments (line 843)
    # Getting the type of 'Cinv' (line 843)
    Cinv_31979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 40), 'Cinv', False)
    # Processing the call keyword arguments (line 843)
    kwargs_31980 = {}
    
    # Call to dot(...): (line 843)
    # Processing the call arguments (line 843)
    # Getting the type of 'C' (line 843)
    C_31975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 33), 'C', False)
    # Processing the call keyword arguments (line 843)
    kwargs_31976 = {}
    # Getting the type of 'Cinv' (line 843)
    Cinv_31973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 24), 'Cinv', False)
    # Obtaining the member 'dot' of a type (line 843)
    dot_31974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 24), Cinv_31973, 'dot')
    # Calling dot(args, kwargs) (line 843)
    dot_call_result_31977 = invoke(stypy.reporting.localization.Localization(__file__, 843, 24), dot_31974, *[C_31975], **kwargs_31976)
    
    # Obtaining the member 'dot' of a type (line 843)
    dot_31978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 24), dot_call_result_31977, 'dot')
    # Calling dot(args, kwargs) (line 843)
    dot_call_result_31981 = invoke(stypy.reporting.localization.Localization(__file__, 843, 24), dot_31978, *[Cinv_31979], **kwargs_31980)
    
    # Applying the binary operator '-' (line 843)
    result_sub_31982 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 15), '-', result_mul_31972, dot_call_result_31981)
    
    # Assigning a type to the variable 'Cinv' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'Cinv', result_sub_31982)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 844):
    
    # Assigning a BinOp to a Name (line 844):
    float_31983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 10), 'float')
    
    # Obtaining the type of the subscript
    int_31984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 24), 'int')
    slice_31985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 844, 17), None, None, int_31984)
    # Getting the type of 'nvec' (line 844)
    nvec_31986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 17), 'nvec')
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___31987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 17), nvec_31986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_31988 = invoke(stypy.reporting.localization.Localization(__file__, 844, 17), getitem___31987, slice_31985)
    
    int_31989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 27), 'int')
    # Applying the binary operator '+' (line 844)
    result_add_31990 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 17), '+', subscript_call_result_31988, int_31989)
    
    # Applying the binary operator 'div' (line 844)
    result_div_31991 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 10), 'div', float_31983, result_add_31990)
    
    # Assigning a type to the variable 'vec' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 4), 'vec', result_div_31991)
    
    # Assigning a BinOp to a Name (line 845):
    
    # Assigning a BinOp to a Name (line 845):
    
    # Call to dot(...): (line 845)
    # Processing the call arguments (line 845)
    # Getting the type of 'vec' (line 845)
    vec_31999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 26), 'vec', False)
    # Processing the call keyword arguments (line 845)
    kwargs_32000 = {}
    
    # Obtaining the type of the subscript
    slice_31992 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 845, 9), None, None, None)
    int_31993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 19), 'int')
    slice_31994 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 845, 9), None, None, int_31993)
    # Getting the type of 'Cinv' (line 845)
    Cinv_31995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 9), 'Cinv', False)
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___31996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 9), Cinv_31995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_31997 = invoke(stypy.reporting.localization.Localization(__file__, 845, 9), getitem___31996, (slice_31992, slice_31994))
    
    # Obtaining the member 'dot' of a type (line 845)
    dot_31998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 9), subscript_call_result_31997, 'dot')
    # Calling dot(args, kwargs) (line 845)
    dot_call_result_32001 = invoke(stypy.reporting.localization.Localization(__file__, 845, 9), dot_31998, *[vec_31999], **kwargs_32000)
    
    # Getting the type of 'N' (line 845)
    N_32002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 34), 'N')
    float_32003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 38), 'float')
    # Applying the binary operator 'div' (line 845)
    result_div_32004 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 34), 'div', N_32002, float_32003)
    
    # Applying the binary operator '*' (line 845)
    result_mul_32005 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 9), '*', dot_call_result_32001, result_div_32004)
    
    # Assigning a type to the variable 'ai' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'ai', result_mul_32005)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'N' (line 847)
    N_32006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 8), 'N')
    int_32007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 12), 'int')
    # Applying the binary operator '%' (line 847)
    result_mod_32008 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 8), '%', N_32006, int_32007)
    
    int_32009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 17), 'int')
    # Applying the binary operator '==' (line 847)
    result_eq_32010 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 8), '==', result_mod_32008, int_32009)
    
    # Getting the type of 'equal' (line 847)
    equal_32011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 24), 'equal')
    # Applying the binary operator 'and' (line 847)
    result_and_keyword_32012 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 7), 'and', result_eq_32010, equal_32011)
    
    # Testing the type of an if condition (line 847)
    if_condition_32013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 847, 4), result_and_keyword_32012)
    # Assigning a type to the variable 'if_condition_32013' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'if_condition_32013', if_condition_32013)
    # SSA begins for if statement (line 847)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 848):
    
    # Assigning a BinOp to a Name (line 848):
    # Getting the type of 'N' (line 848)
    N_32014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 13), 'N')
    # Getting the type of 'N' (line 848)
    N_32015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 16), 'N')
    float_32016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 18), 'float')
    # Applying the binary operator '+' (line 848)
    result_add_32017 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 16), '+', N_32015, float_32016)
    
    # Applying the binary operator 'div' (line 848)
    result_div_32018 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 13), 'div', N_32014, result_add_32017)
    
    # Assigning a type to the variable 'BN' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'BN', result_div_32018)
    
    # Assigning a BinOp to a Name (line 849):
    
    # Assigning a BinOp to a Name (line 849):
    # Getting the type of 'N' (line 849)
    N_32019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 16), 'N')
    int_32020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 18), 'int')
    # Applying the binary operator '+' (line 849)
    result_add_32021 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 16), '+', N_32019, int_32020)
    
    # Assigning a type to the variable 'power' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'power', result_add_32021)
    # SSA branch for the else part of an if statement (line 847)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 851):
    
    # Assigning a BinOp to a Name (line 851):
    # Getting the type of 'N' (line 851)
    N_32022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 13), 'N')
    # Getting the type of 'N' (line 851)
    N_32023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 16), 'N')
    float_32024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 18), 'float')
    # Applying the binary operator '+' (line 851)
    result_add_32025 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 16), '+', N_32023, float_32024)
    
    # Applying the binary operator 'div' (line 851)
    result_div_32026 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 13), 'div', N_32022, result_add_32025)
    
    # Assigning a type to the variable 'BN' (line 851)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'BN', result_div_32026)
    
    # Assigning a BinOp to a Name (line 852):
    
    # Assigning a BinOp to a Name (line 852):
    # Getting the type of 'N' (line 852)
    N_32027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 16), 'N')
    int_32028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 18), 'int')
    # Applying the binary operator '+' (line 852)
    result_add_32029 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 16), '+', N_32027, int_32028)
    
    # Assigning a type to the variable 'power' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'power', result_add_32029)
    # SSA join for if statement (line 847)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 854):
    
    # Assigning a BinOp to a Name (line 854):
    # Getting the type of 'BN' (line 854)
    BN_32030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 9), 'BN')
    
    # Call to dot(...): (line 854)
    # Processing the call arguments (line 854)
    # Getting the type of 'yi' (line 854)
    yi_32033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 21), 'yi', False)
    # Getting the type of 'power' (line 854)
    power_32034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 25), 'power', False)
    # Applying the binary operator '**' (line 854)
    result_pow_32035 = python_operator(stypy.reporting.localization.Localization(__file__, 854, 21), '**', yi_32033, power_32034)
    
    # Getting the type of 'ai' (line 854)
    ai_32036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 32), 'ai', False)
    # Processing the call keyword arguments (line 854)
    kwargs_32037 = {}
    # Getting the type of 'np' (line 854)
    np_32031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 854)
    dot_32032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 14), np_32031, 'dot')
    # Calling dot(args, kwargs) (line 854)
    dot_call_result_32038 = invoke(stypy.reporting.localization.Localization(__file__, 854, 14), dot_32032, *[result_pow_32035, ai_32036], **kwargs_32037)
    
    # Applying the binary operator '-' (line 854)
    result_sub_32039 = python_operator(stypy.reporting.localization.Localization(__file__, 854, 9), '-', BN_32030, dot_call_result_32038)
    
    # Assigning a type to the variable 'BN' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 4), 'BN', result_sub_32039)
    
    # Assigning a BinOp to a Name (line 855):
    
    # Assigning a BinOp to a Name (line 855):
    # Getting the type of 'power' (line 855)
    power_32040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 9), 'power')
    int_32041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 15), 'int')
    # Applying the binary operator '+' (line 855)
    result_add_32042 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 9), '+', power_32040, int_32041)
    
    # Assigning a type to the variable 'p1' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 4), 'p1', result_add_32042)
    
    # Assigning a BinOp to a Name (line 856):
    
    # Assigning a BinOp to a Name (line 856):
    # Getting the type of 'power' (line 856)
    power_32043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 10), 'power')
    
    # Call to log(...): (line 856)
    # Processing the call arguments (line 856)
    # Getting the type of 'N' (line 856)
    N_32046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 25), 'N', False)
    # Processing the call keyword arguments (line 856)
    kwargs_32047 = {}
    # Getting the type of 'math' (line 856)
    math_32044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 16), 'math', False)
    # Obtaining the member 'log' of a type (line 856)
    log_32045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 16), math_32044, 'log')
    # Calling log(args, kwargs) (line 856)
    log_call_result_32048 = invoke(stypy.reporting.localization.Localization(__file__, 856, 16), log_32045, *[N_32046], **kwargs_32047)
    
    # Applying the binary operator '*' (line 856)
    result_mul_32049 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 10), '*', power_32043, log_call_result_32048)
    
    
    # Call to gammaln(...): (line 856)
    # Processing the call arguments (line 856)
    # Getting the type of 'p1' (line 856)
    p1_32051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 38), 'p1', False)
    # Processing the call keyword arguments (line 856)
    kwargs_32052 = {}
    # Getting the type of 'gammaln' (line 856)
    gammaln_32050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 30), 'gammaln', False)
    # Calling gammaln(args, kwargs) (line 856)
    gammaln_call_result_32053 = invoke(stypy.reporting.localization.Localization(__file__, 856, 30), gammaln_32050, *[p1_32051], **kwargs_32052)
    
    # Applying the binary operator '-' (line 856)
    result_sub_32054 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 10), '-', result_mul_32049, gammaln_call_result_32053)
    
    # Assigning a type to the variable 'fac' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'fac', result_sub_32054)
    
    # Assigning a Call to a Name (line 857):
    
    # Assigning a Call to a Name (line 857):
    
    # Call to exp(...): (line 857)
    # Processing the call arguments (line 857)
    # Getting the type of 'fac' (line 857)
    fac_32057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 19), 'fac', False)
    # Processing the call keyword arguments (line 857)
    kwargs_32058 = {}
    # Getting the type of 'math' (line 857)
    math_32055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 10), 'math', False)
    # Obtaining the member 'exp' of a type (line 857)
    exp_32056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 10), math_32055, 'exp')
    # Calling exp(args, kwargs) (line 857)
    exp_call_result_32059 = invoke(stypy.reporting.localization.Localization(__file__, 857, 10), exp_32056, *[fac_32057], **kwargs_32058)
    
    # Assigning a type to the variable 'fac' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'fac', exp_call_result_32059)
    
    # Obtaining an instance of the builtin type 'tuple' (line 858)
    tuple_32060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 858)
    # Adding element type (line 858)
    # Getting the type of 'ai' (line 858)
    ai_32061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 11), 'ai')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 858, 11), tuple_32060, ai_32061)
    # Adding element type (line 858)
    # Getting the type of 'BN' (line 858)
    BN_32062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 15), 'BN')
    # Getting the type of 'fac' (line 858)
    fac_32063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 18), 'fac')
    # Applying the binary operator '*' (line 858)
    result_mul_32064 = python_operator(stypy.reporting.localization.Localization(__file__, 858, 15), '*', BN_32062, fac_32063)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 858, 11), tuple_32060, result_mul_32064)
    
    # Assigning a type to the variable 'stypy_return_type' (line 858)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 858, 4), 'stypy_return_type', tuple_32060)
    
    # ################# End of 'newton_cotes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'newton_cotes' in the type store
    # Getting the type of 'stypy_return_type' (line 777)
    stypy_return_type_32065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'newton_cotes'
    return stypy_return_type_32065

# Assigning a type to the variable 'newton_cotes' (line 777)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 0), 'newton_cotes', newton_cotes)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
