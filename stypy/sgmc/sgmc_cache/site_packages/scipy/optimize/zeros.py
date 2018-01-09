
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import warnings
4: 
5: from . import _zeros
6: from numpy import finfo, sign, sqrt
7: 
8: _iter = 100
9: _xtol = 2e-12
10: _rtol = 4*finfo(float).eps
11: 
12: __all__ = ['newton', 'bisect', 'ridder', 'brentq', 'brenth']
13: 
14: CONVERGED = 'converged'
15: SIGNERR = 'sign error'
16: CONVERR = 'convergence error'
17: flag_map = {0: CONVERGED, -1: SIGNERR, -2: CONVERR}
18: 
19: 
20: class RootResults(object):
21:     ''' Represents the root finding result.
22:     Attributes
23:     ----------
24:     root : float
25:         Estimated root location.
26:     iterations : int
27:         Number of iterations needed to find the root.
28:     function_calls : int
29:         Number of times the function was called.
30:     converged : bool
31:         True if the routine converged.
32:     flag : str
33:         Description of the cause of termination.
34:     '''
35:     def __init__(self, root, iterations, function_calls, flag):
36:         self.root = root
37:         self.iterations = iterations
38:         self.function_calls = function_calls
39:         self.converged = flag == 0
40:         try:
41:             self.flag = flag_map[flag]
42:         except KeyError:
43:             self.flag = 'unknown error %d' % (flag,)
44: 
45:     def __repr__(self):
46:         attrs = ['converged', 'flag', 'function_calls',
47:                  'iterations', 'root']
48:         m = max(map(len, attrs)) + 1
49:         return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self, a))
50:                           for a in attrs])
51: 
52: 
53: def results_c(full_output, r):
54:     if full_output:
55:         x, funcalls, iterations, flag = r
56:         results = RootResults(root=x,
57:                               iterations=iterations,
58:                               function_calls=funcalls,
59:                               flag=flag)
60:         return x, results
61:     else:
62:         return r
63: 
64: 
65: # Newton-Raphson method
66: def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
67:            fprime2=None):
68:     '''
69:     Find a zero using the Newton-Raphson or secant method.
70: 
71:     Find a zero of the function `func` given a nearby starting point `x0`.
72:     The Newton-Raphson method is used if the derivative `fprime` of `func`
73:     is provided, otherwise the secant method is used.  If the second order
74:     derivate `fprime2` of `func` is provided, parabolic Halley's method
75:     is used.
76: 
77:     Parameters
78:     ----------
79:     func : function
80:         The function whose zero is wanted. It must be a function of a
81:         single variable of the form f(x,a,b,c...), where a,b,c... are extra
82:         arguments that can be passed in the `args` parameter.
83:     x0 : float
84:         An initial estimate of the zero that should be somewhere near the
85:         actual zero.
86:     fprime : function, optional
87:         The derivative of the function when available and convenient. If it
88:         is None (default), then the secant method is used.
89:     args : tuple, optional
90:         Extra arguments to be used in the function call.
91:     tol : float, optional
92:         The allowable error of the zero value.
93:     maxiter : int, optional
94:         Maximum number of iterations.
95:     fprime2 : function, optional
96:         The second order derivative of the function when available and
97:         convenient. If it is None (default), then the normal Newton-Raphson
98:         or the secant method is used. If it is given, parabolic Halley's
99:         method is used.
100: 
101:     Returns
102:     -------
103:     zero : float
104:         Estimated location where function is zero.
105: 
106:     See Also
107:     --------
108:     brentq, brenth, ridder, bisect
109:     fsolve : find zeroes in n dimensions.
110: 
111:     Notes
112:     -----
113:     The convergence rate of the Newton-Raphson method is quadratic,
114:     the Halley method is cubic, and the secant method is
115:     sub-quadratic.  This means that if the function is well behaved
116:     the actual error in the estimated zero is approximately the square
117:     (cube for Halley) of the requested tolerance up to roundoff
118:     error. However, the stopping criterion used here is the step size
119:     and there is no guarantee that a zero has been found. Consequently
120:     the result should be verified. Safer algorithms are brentq,
121:     brenth, ridder, and bisect, but they all require that the root
122:     first be bracketed in an interval where the function changes
123:     sign. The brentq algorithm is recommended for general use in one
124:     dimensional problems when such an interval has been found.
125: 
126:     Examples
127:     --------
128: 
129:     >>> def f(x):
130:     ...     return (x**3 - 1)  # only one real root at x = 1
131:     
132:     >>> from scipy import optimize
133: 
134:     ``fprime`` and ``fprime2`` not provided, use secant method
135:     
136:     >>> root = optimize.newton(f, 1.5)
137:     >>> root
138:     1.0000000000000016
139: 
140:     Only ``fprime`` provided, use Newton Raphson method
141:     
142:     >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
143:     >>> root
144:     1.0
145:     
146:     ``fprime2`` provided, ``fprime`` provided/not provided use parabolic
147:     Halley's method
148: 
149:     >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)
150:     >>> root
151:     1.0000000000000016
152:     >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
153:     ...                        fprime2=lambda x: 6 * x)
154:     >>> root
155:     1.0
156: 
157:     '''
158:     if tol <= 0:
159:         raise ValueError("tol too small (%g <= 0)" % tol)
160:     if maxiter < 1:
161:         raise ValueError("maxiter must be greater than 0")
162:     if fprime is not None:
163:         # Newton-Rapheson method
164:         # Multiply by 1.0 to convert to floating point.  We don't use float(x0)
165:         # so it still works if x0 is complex.
166:         p0 = 1.0 * x0
167:         fder2 = 0
168:         for iter in range(maxiter):
169:             myargs = (p0,) + args
170:             fder = fprime(*myargs)
171:             if fder == 0:
172:                 msg = "derivative was zero."
173:                 warnings.warn(msg, RuntimeWarning)
174:                 return p0
175:             fval = func(*myargs)
176:             if fprime2 is not None:
177:                 fder2 = fprime2(*myargs)
178:             if fder2 == 0:
179:                 # Newton step
180:                 p = p0 - fval / fder
181:             else:
182:                 # Parabolic Halley's method
183:                 discr = fder ** 2 - 2 * fval * fder2
184:                 if discr < 0:
185:                     p = p0 - fder / fder2
186:                 else:
187:                     p = p0 - 2*fval / (fder + sign(fder) * sqrt(discr))
188:             if abs(p - p0) < tol:
189:                 return p
190:             p0 = p
191:     else:
192:         # Secant method
193:         p0 = x0
194:         if x0 >= 0:
195:             p1 = x0*(1 + 1e-4) + 1e-4
196:         else:
197:             p1 = x0*(1 + 1e-4) - 1e-4
198:         q0 = func(*((p0,) + args))
199:         q1 = func(*((p1,) + args))
200:         for iter in range(maxiter):
201:             if q1 == q0:
202:                 if p1 != p0:
203:                     msg = "Tolerance of %s reached" % (p1 - p0)
204:                     warnings.warn(msg, RuntimeWarning)
205:                 return (p1 + p0)/2.0
206:             else:
207:                 p = p1 - q1*(p1 - p0)/(q1 - q0)
208:             if abs(p - p1) < tol:
209:                 return p
210:             p0 = p1
211:             q0 = q1
212:             p1 = p
213:             q1 = func(*((p1,) + args))
214:     msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
215:     raise RuntimeError(msg)
216: 
217: 
218: def bisect(f, a, b, args=(),
219:            xtol=_xtol, rtol=_rtol, maxiter=_iter,
220:            full_output=False, disp=True):
221:     '''
222:     Find root of a function within an interval.
223: 
224:     Basic bisection routine to find a zero of the function `f` between the
225:     arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.
226:     Slow but sure.
227: 
228:     Parameters
229:     ----------
230:     f : function
231:         Python function returning a number.  `f` must be continuous, and
232:         f(a) and f(b) must have opposite signs.
233:     a : number
234:         One end of the bracketing interval [a,b].
235:     b : number
236:         The other end of the bracketing interval [a,b].
237:     xtol : number, optional
238:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
239:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
240:         parameter must be nonnegative.
241:     rtol : number, optional
242:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
243:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
244:         parameter cannot be smaller than its default value of
245:         ``4*np.finfo(float).eps``.
246:     maxiter : number, optional
247:         if convergence is not achieved in `maxiter` iterations, an error is
248:         raised.  Must be >= 0.
249:     args : tuple, optional
250:         containing extra arguments for the function `f`.
251:         `f` is called by ``apply(f, (x)+args)``.
252:     full_output : bool, optional
253:         If `full_output` is False, the root is returned.  If `full_output` is
254:         True, the return value is ``(x, r)``, where x is the root, and r is
255:         a `RootResults` object.
256:     disp : bool, optional
257:         If True, raise RuntimeError if the algorithm didn't converge.
258: 
259:     Returns
260:     -------
261:     x0 : float
262:         Zero of `f` between `a` and `b`.
263:     r : RootResults (present if ``full_output = True``)
264:         Object containing information about the convergence.  In particular,
265:         ``r.converged`` is True if the routine converged.
266: 
267:     Examples
268:     --------
269: 
270:     >>> def f(x):
271:     ...     return (x**2 - 1)
272: 
273:     >>> from scipy import optimize
274: 
275:     >>> root = optimize.bisect(f, 0, 2)
276:     >>> root
277:     1.0
278: 
279:     >>> root = optimize.bisect(f, -2, 0)
280:     >>> root
281:     -1.0
282: 
283:     See Also
284:     --------
285:     brentq, brenth, bisect, newton
286:     fixed_point : scalar fixed-point finder
287:     fsolve : n-dimensional root-finding
288: 
289:     '''
290:     if not isinstance(args, tuple):
291:         args = (args,)
292:     if xtol <= 0:
293:         raise ValueError("xtol too small (%g <= 0)" % xtol)
294:     if rtol < _rtol:
295:         raise ValueError("rtol too small (%g < %g)" % (rtol, _rtol))
296:     r = _zeros._bisect(f,a,b,xtol,rtol,maxiter,args,full_output,disp)
297:     return results_c(full_output, r)
298: 
299: 
300: def ridder(f, a, b, args=(),
301:            xtol=_xtol, rtol=_rtol, maxiter=_iter,
302:            full_output=False, disp=True):
303:     '''
304:     Find a root of a function in an interval.
305: 
306:     Parameters
307:     ----------
308:     f : function
309:         Python function returning a number.  f must be continuous, and f(a) and
310:         f(b) must have opposite signs.
311:     a : number
312:         One end of the bracketing interval [a,b].
313:     b : number
314:         The other end of the bracketing interval [a,b].
315:     xtol : number, optional
316:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
317:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
318:         parameter must be nonnegative.
319:     rtol : number, optional
320:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
321:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
322:         parameter cannot be smaller than its default value of
323:         ``4*np.finfo(float).eps``.
324:     maxiter : number, optional
325:         if convergence is not achieved in maxiter iterations, an error is
326:         raised.  Must be >= 0.
327:     args : tuple, optional
328:         containing extra arguments for the function `f`.
329:         `f` is called by ``apply(f, (x)+args)``.
330:     full_output : bool, optional
331:         If `full_output` is False, the root is returned.  If `full_output` is
332:         True, the return value is ``(x, r)``, where `x` is the root, and `r` is
333:         a RootResults object.
334:     disp : bool, optional
335:         If True, raise RuntimeError if the algorithm didn't converge.
336: 
337:     Returns
338:     -------
339:     x0 : float
340:         Zero of `f` between `a` and `b`.
341:     r : RootResults (present if ``full_output = True``)
342:         Object containing information about the convergence.
343:         In particular, ``r.converged`` is True if the routine converged.
344: 
345:     See Also
346:     --------
347:     brentq, brenth, bisect, newton : one-dimensional root-finding
348:     fixed_point : scalar fixed-point finder
349: 
350:     Notes
351:     -----
352:     Uses [Ridders1979]_ method to find a zero of the function `f` between the
353:     arguments `a` and `b`. Ridders' method is faster than bisection, but not
354:     generally as fast as the Brent rountines. [Ridders1979]_ provides the
355:     classic description and source of the algorithm. A description can also be
356:     found in any recent edition of Numerical Recipes.
357: 
358:     The routine used here diverges slightly from standard presentations in
359:     order to be a bit more careful of tolerance.
360: 
361:     Examples
362:     --------
363: 
364:     >>> def f(x):
365:     ...     return (x**2 - 1)
366: 
367:     >>> from scipy import optimize
368: 
369:     >>> root = optimize.ridder(f, 0, 2)
370:     >>> root
371:     1.0
372: 
373:     >>> root = optimize.ridder(f, -2, 0)
374:     >>> root
375:     -1.0
376: 
377:     References
378:     ----------
379:     .. [Ridders1979]
380:        Ridders, C. F. J. "A New Algorithm for Computing a
381:        Single Root of a Real Continuous Function."
382:        IEEE Trans. Circuits Systems 26, 979-980, 1979.
383: 
384:     '''
385:     if not isinstance(args, tuple):
386:         args = (args,)
387:     if xtol <= 0:
388:         raise ValueError("xtol too small (%g <= 0)" % xtol)
389:     if rtol < _rtol:
390:         raise ValueError("rtol too small (%g < %g)" % (rtol, _rtol))
391:     r = _zeros._ridder(f,a,b,xtol,rtol,maxiter,args,full_output,disp)
392:     return results_c(full_output, r)
393: 
394: 
395: def brentq(f, a, b, args=(),
396:            xtol=_xtol, rtol=_rtol, maxiter=_iter,
397:            full_output=False, disp=True):
398:     '''
399:     Find a root of a function in a bracketing interval using Brent's method.
400: 
401:     Uses the classic Brent's method to find a zero of the function `f` on
402:     the sign changing interval [a , b].  Generally considered the best of the
403:     rootfinding routines here.  It is a safe version of the secant method that
404:     uses inverse quadratic extrapolation.  Brent's method combines root
405:     bracketing, interval bisection, and inverse quadratic interpolation.  It is
406:     sometimes known as the van Wijngaarden-Dekker-Brent method.  Brent (1973)
407:     claims convergence is guaranteed for functions computable within [a,b].
408: 
409:     [Brent1973]_ provides the classic description of the algorithm.  Another
410:     description can be found in a recent edition of Numerical Recipes, including
411:     [PressEtal1992]_.  Another description is at
412:     http://mathworld.wolfram.com/BrentsMethod.html.  It should be easy to
413:     understand the algorithm just by reading our code.  Our code diverges a bit
414:     from standard presentations: we choose a different formula for the
415:     extrapolation step.
416: 
417:     Parameters
418:     ----------
419:     f : function
420:         Python function returning a number.  The function :math:`f`
421:         must be continuous, and :math:`f(a)` and :math:`f(b)` must
422:         have opposite signs.
423:     a : number
424:         One end of the bracketing interval :math:`[a, b]`.
425:     b : number
426:         The other end of the bracketing interval :math:`[a, b]`.
427:     xtol : number, optional
428:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
429:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
430:         parameter must be nonnegative. For nice functions, Brent's
431:         method will often satisfy the above condition with ``xtol/2``
432:         and ``rtol/2``. [Brent1973]_
433:     rtol : number, optional
434:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
435:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
436:         parameter cannot be smaller than its default value of
437:         ``4*np.finfo(float).eps``. For nice functions, Brent's
438:         method will often satisfy the above condition with ``xtol/2``
439:         and ``rtol/2``. [Brent1973]_
440:     maxiter : number, optional
441:         if convergence is not achieved in maxiter iterations, an error is
442:         raised.  Must be >= 0.
443:     args : tuple, optional
444:         containing extra arguments for the function `f`.
445:         `f` is called by ``apply(f, (x)+args)``.
446:     full_output : bool, optional
447:         If `full_output` is False, the root is returned.  If `full_output` is
448:         True, the return value is ``(x, r)``, where `x` is the root, and `r` is
449:         a RootResults object.
450:     disp : bool, optional
451:         If True, raise RuntimeError if the algorithm didn't converge.
452: 
453:     Returns
454:     -------
455:     x0 : float
456:         Zero of `f` between `a` and `b`.
457:     r : RootResults (present if ``full_output = True``)
458:         Object containing information about the convergence.  In particular,
459:         ``r.converged`` is True if the routine converged.
460: 
461:     See Also
462:     --------
463:     multivariate local optimizers
464:       `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, `fmin_ncg`
465:     nonlinear least squares minimizer
466:       `leastsq`
467:     constrained multivariate optimizers
468:       `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`
469:     global optimizers
470:       `basinhopping`, `brute`, `differential_evolution`
471:     local scalar minimizers
472:       `fminbound`, `brent`, `golden`, `bracket`
473:     n-dimensional root-finding
474:       `fsolve`
475:     one-dimensional root-finding
476:       `brenth`, `ridder`, `bisect`, `newton`
477:     scalar fixed-point finder
478:       `fixed_point`
479: 
480:     Notes
481:     -----
482:     `f` must be continuous.  f(a) and f(b) must have opposite signs.
483: 
484:     Examples
485:     --------
486:     >>> def f(x):
487:     ...     return (x**2 - 1)
488: 
489:     >>> from scipy import optimize
490: 
491:     >>> root = optimize.brentq(f, -2, 0)
492:     >>> root
493:     -1.0
494: 
495:     >>> root = optimize.brentq(f, 0, 2)
496:     >>> root
497:     1.0
498: 
499:     References
500:     ----------
501:     .. [Brent1973]
502:        Brent, R. P.,
503:        *Algorithms for Minimization Without Derivatives*.
504:        Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.
505: 
506:     .. [PressEtal1992]
507:        Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.
508:        *Numerical Recipes in FORTRAN: The Art of Scientific Computing*, 2nd ed.
509:        Cambridge, England: Cambridge University Press, pp. 352-355, 1992.
510:        Section 9.3:  "Van Wijngaarden-Dekker-Brent Method."
511: 
512:     '''
513:     if not isinstance(args, tuple):
514:         args = (args,)
515:     if xtol <= 0:
516:         raise ValueError("xtol too small (%g <= 0)" % xtol)
517:     if rtol < _rtol:
518:         raise ValueError("rtol too small (%g < %g)" % (rtol, _rtol))
519:     r = _zeros._brentq(f,a,b,xtol,rtol,maxiter,args,full_output,disp)
520:     return results_c(full_output, r)
521: 
522: 
523: def brenth(f, a, b, args=(),
524:            xtol=_xtol, rtol=_rtol, maxiter=_iter,
525:            full_output=False, disp=True):
526:     '''Find root of f in [a,b].
527: 
528:     A variation on the classic Brent routine to find a zero of the function f
529:     between the arguments a and b that uses hyperbolic extrapolation instead of
530:     inverse quadratic extrapolation. There was a paper back in the 1980's ...
531:     f(a) and f(b) cannot have the same signs. Generally on a par with the
532:     brent routine, but not as heavily tested.  It is a safe version of the
533:     secant method that uses hyperbolic extrapolation. The version here is by
534:     Chuck Harris.
535: 
536:     Parameters
537:     ----------
538:     f : function
539:         Python function returning a number.  f must be continuous, and f(a) and
540:         f(b) must have opposite signs.
541:     a : number
542:         One end of the bracketing interval [a,b].
543:     b : number
544:         The other end of the bracketing interval [a,b].
545:     xtol : number, optional
546:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
547:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
548:         parameter must be nonnegative. As with `brentq`, for nice
549:         functions the method will often satisfy the above condition
550:         with ``xtol/2`` and ``rtol/2``.
551:     rtol : number, optional
552:         The computed root ``x0`` will satisfy ``np.allclose(x, x0,
553:         atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
554:         parameter cannot be smaller than its default value of
555:         ``4*np.finfo(float).eps``. As with `brentq`, for nice functions
556:         the method will often satisfy the above condition with
557:         ``xtol/2`` and ``rtol/2``.
558:     maxiter : number, optional
559:         if convergence is not achieved in maxiter iterations, an error is
560:         raised.  Must be >= 0.
561:     args : tuple, optional
562:         containing extra arguments for the function `f`.
563:         `f` is called by ``apply(f, (x)+args)``.
564:     full_output : bool, optional
565:         If `full_output` is False, the root is returned.  If `full_output` is
566:         True, the return value is ``(x, r)``, where `x` is the root, and `r` is
567:         a RootResults object.
568:     disp : bool, optional
569:         If True, raise RuntimeError if the algorithm didn't converge.
570: 
571:     Returns
572:     -------
573:     x0 : float
574:         Zero of `f` between `a` and `b`.
575:     r : RootResults (present if ``full_output = True``)
576:         Object containing information about the convergence.  In particular,
577:         ``r.converged`` is True if the routine converged.
578: 
579:     Examples
580:     --------
581:     >>> def f(x):
582:     ...     return (x**2 - 1)
583: 
584:     >>> from scipy import optimize
585: 
586:     >>> root = optimize.brenth(f, -2, 0)
587:     >>> root
588:     -1.0
589: 
590:     >>> root = optimize.brenth(f, 0, 2)
591:     >>> root
592:     1.0
593: 
594:     See Also
595:     --------
596:     fmin, fmin_powell, fmin_cg,
597:            fmin_bfgs, fmin_ncg : multivariate local optimizers
598: 
599:     leastsq : nonlinear least squares minimizer
600: 
601:     fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
602: 
603:     basinhopping, differential_evolution, brute : global optimizers
604: 
605:     fminbound, brent, golden, bracket : local scalar minimizers
606: 
607:     fsolve : n-dimensional root-finding
608: 
609:     brentq, brenth, ridder, bisect, newton : one-dimensional root-finding
610: 
611:     fixed_point : scalar fixed-point finder
612: 
613:     '''
614:     if not isinstance(args, tuple):
615:         args = (args,)
616:     if xtol <= 0:
617:         raise ValueError("xtol too small (%g <= 0)" % xtol)
618:     if rtol < _rtol:
619:         raise ValueError("rtol too small (%g < %g)" % (rtol, _rtol))
620:     r = _zeros._brenth(f,a, b, xtol, rtol, maxiter, args, full_output, disp)
621:     return results_c(full_output, r)
622: 

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

# 'from scipy.optimize import _zeros' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_186767 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize')

if (type(import_186767) is not StypyTypeError):

    if (import_186767 != 'pyd_module'):
        __import__(import_186767)
        sys_modules_186768 = sys.modules[import_186767]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize', sys_modules_186768.module_type_store, module_type_store, ['_zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_186768, sys_modules_186768.module_type_store, module_type_store)
    else:
        from scipy.optimize import _zeros

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize', None, module_type_store, ['_zeros'], [_zeros])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize', import_186767)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import finfo, sign, sqrt' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_186769 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_186769) is not StypyTypeError):

    if (import_186769 != 'pyd_module'):
        __import__(import_186769)
        sys_modules_186770 = sys.modules[import_186769]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_186770.module_type_store, module_type_store, ['finfo', 'sign', 'sqrt'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_186770, sys_modules_186770.module_type_store, module_type_store)
    else:
        from numpy import finfo, sign, sqrt

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['finfo', 'sign', 'sqrt'], [finfo, sign, sqrt])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_186769)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a Num to a Name (line 8):

# Assigning a Num to a Name (line 8):
int_186771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'int')
# Assigning a type to the variable '_iter' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '_iter', int_186771)

# Assigning a Num to a Name (line 9):

# Assigning a Num to a Name (line 9):
float_186772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'float')
# Assigning a type to the variable '_xtol' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '_xtol', float_186772)

# Assigning a BinOp to a Name (line 10):

# Assigning a BinOp to a Name (line 10):
int_186773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')

# Call to finfo(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'float' (line 10)
float_186775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'float', False)
# Processing the call keyword arguments (line 10)
kwargs_186776 = {}
# Getting the type of 'finfo' (line 10)
finfo_186774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'finfo', False)
# Calling finfo(args, kwargs) (line 10)
finfo_call_result_186777 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), finfo_186774, *[float_186775], **kwargs_186776)

# Obtaining the member 'eps' of a type (line 10)
eps_186778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), finfo_call_result_186777, 'eps')
# Applying the binary operator '*' (line 10)
result_mul_186779 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 8), '*', int_186773, eps_186778)

# Assigning a type to the variable '_rtol' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '_rtol', result_mul_186779)

# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['newton', 'bisect', 'ridder', 'brentq', 'brenth']
module_type_store.set_exportable_members(['newton', 'bisect', 'ridder', 'brentq', 'brenth'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_186780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_186781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'newton')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_186780, str_186781)
# Adding element type (line 12)
str_186782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'str', 'bisect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_186780, str_186782)
# Adding element type (line 12)
str_186783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'str', 'ridder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_186780, str_186783)
# Adding element type (line 12)
str_186784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 41), 'str', 'brentq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_186780, str_186784)
# Adding element type (line 12)
str_186785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 51), 'str', 'brenth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_186780, str_186785)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_186780)

# Assigning a Str to a Name (line 14):

# Assigning a Str to a Name (line 14):
str_186786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'str', 'converged')
# Assigning a type to the variable 'CONVERGED' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'CONVERGED', str_186786)

# Assigning a Str to a Name (line 15):

# Assigning a Str to a Name (line 15):
str_186787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'str', 'sign error')
# Assigning a type to the variable 'SIGNERR' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'SIGNERR', str_186787)

# Assigning a Str to a Name (line 16):

# Assigning a Str to a Name (line 16):
str_186788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'str', 'convergence error')
# Assigning a type to the variable 'CONVERR' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'CONVERR', str_186788)

# Assigning a Dict to a Name (line 17):

# Assigning a Dict to a Name (line 17):

# Obtaining an instance of the builtin type 'dict' (line 17)
dict_186789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 17)
# Adding element type (key, value) (line 17)
int_186790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'int')
# Getting the type of 'CONVERGED' (line 17)
CONVERGED_186791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'CONVERGED')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 11), dict_186789, (int_186790, CONVERGED_186791))
# Adding element type (key, value) (line 17)
int_186792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
# Getting the type of 'SIGNERR' (line 17)
SIGNERR_186793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 30), 'SIGNERR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 11), dict_186789, (int_186792, SIGNERR_186793))
# Adding element type (key, value) (line 17)
int_186794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 39), 'int')
# Getting the type of 'CONVERR' (line 17)
CONVERR_186795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 43), 'CONVERR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 11), dict_186789, (int_186794, CONVERR_186795))

# Assigning a type to the variable 'flag_map' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'flag_map', dict_186789)
# Declaration of the 'RootResults' class

class RootResults(object, ):
    str_186796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', ' Represents the root finding result.\n    Attributes\n    ----------\n    root : float\n        Estimated root location.\n    iterations : int\n        Number of iterations needed to find the root.\n    function_calls : int\n        Number of times the function was called.\n    converged : bool\n        True if the routine converged.\n    flag : str\n        Description of the cause of termination.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RootResults.__init__', ['root', 'iterations', 'function_calls', 'flag'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['root', 'iterations', 'function_calls', 'flag'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 36):
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'root' (line 36)
        root_186797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'root')
        # Getting the type of 'self' (line 36)
        self_186798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'root' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_186798, 'root', root_186797)
        
        # Assigning a Name to a Attribute (line 37):
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'iterations' (line 37)
        iterations_186799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'iterations')
        # Getting the type of 'self' (line 37)
        self_186800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'iterations' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_186800, 'iterations', iterations_186799)
        
        # Assigning a Name to a Attribute (line 38):
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'function_calls' (line 38)
        function_calls_186801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'function_calls')
        # Getting the type of 'self' (line 38)
        self_186802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'function_calls' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_186802, 'function_calls', function_calls_186801)
        
        # Assigning a Compare to a Attribute (line 39):
        
        # Assigning a Compare to a Attribute (line 39):
        
        # Getting the type of 'flag' (line 39)
        flag_186803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'flag')
        int_186804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
        # Applying the binary operator '==' (line 39)
        result_eq_186805 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 25), '==', flag_186803, int_186804)
        
        # Getting the type of 'self' (line 39)
        self_186806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'converged' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_186806, 'converged', result_eq_186805)
        
        
        # SSA begins for try-except statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Attribute (line 41):
        
        # Assigning a Subscript to a Attribute (line 41):
        
        # Obtaining the type of the subscript
        # Getting the type of 'flag' (line 41)
        flag_186807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'flag')
        # Getting the type of 'flag_map' (line 41)
        flag_map_186808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'flag_map')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___186809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 24), flag_map_186808, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_186810 = invoke(stypy.reporting.localization.Localization(__file__, 41, 24), getitem___186809, flag_186807)
        
        # Getting the type of 'self' (line 41)
        self_186811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self')
        # Setting the type of the member 'flag' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_186811, 'flag', subscript_call_result_186810)
        # SSA branch for the except part of a try statement (line 40)
        # SSA branch for the except 'KeyError' branch of a try statement (line 40)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BinOp to a Attribute (line 43):
        
        # Assigning a BinOp to a Attribute (line 43):
        str_186812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'str', 'unknown error %d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_186813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        # Getting the type of 'flag' (line 43)
        flag_186814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'flag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 46), tuple_186813, flag_186814)
        
        # Applying the binary operator '%' (line 43)
        result_mod_186815 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 24), '%', str_186812, tuple_186813)
        
        # Getting the type of 'self' (line 43)
        self_186816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
        # Setting the type of the member 'flag' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_186816, 'flag', result_mod_186815)
        # SSA join for try-except statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'RootResults.stypy__repr__')
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RootResults.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RootResults.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 46):
        
        # Assigning a List to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_186817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        str_186818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'str', 'converged')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 16), list_186817, str_186818)
        # Adding element type (line 46)
        str_186819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'str', 'flag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 16), list_186817, str_186819)
        # Adding element type (line 46)
        str_186820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 38), 'str', 'function_calls')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 16), list_186817, str_186820)
        # Adding element type (line 46)
        str_186821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'str', 'iterations')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 16), list_186817, str_186821)
        # Adding element type (line 46)
        str_186822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'str', 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 16), list_186817, str_186822)
        
        # Assigning a type to the variable 'attrs' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'attrs', list_186817)
        
        # Assigning a BinOp to a Name (line 48):
        
        # Assigning a BinOp to a Name (line 48):
        
        # Call to max(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to map(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'len' (line 48)
        len_186825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'len', False)
        # Getting the type of 'attrs' (line 48)
        attrs_186826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'attrs', False)
        # Processing the call keyword arguments (line 48)
        kwargs_186827 = {}
        # Getting the type of 'map' (line 48)
        map_186824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'map', False)
        # Calling map(args, kwargs) (line 48)
        map_call_result_186828 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), map_186824, *[len_186825, attrs_186826], **kwargs_186827)
        
        # Processing the call keyword arguments (line 48)
        kwargs_186829 = {}
        # Getting the type of 'max' (line 48)
        max_186823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'max', False)
        # Calling max(args, kwargs) (line 48)
        max_call_result_186830 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), max_186823, *[map_call_result_186828], **kwargs_186829)
        
        int_186831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 35), 'int')
        # Applying the binary operator '+' (line 48)
        result_add_186832 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 12), '+', max_call_result_186830, int_186831)
        
        # Assigning a type to the variable 'm' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'm', result_add_186832)
        
        # Call to join(...): (line 49)
        # Processing the call arguments (line 49)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'attrs' (line 50)
        attrs_186851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'attrs', False)
        comprehension_186852 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 26), attrs_186851)
        # Assigning a type to the variable 'a' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'a', comprehension_186852)
        
        # Call to rjust(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'm' (line 49)
        m_186837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'm', False)
        # Processing the call keyword arguments (line 49)
        kwargs_186838 = {}
        # Getting the type of 'a' (line 49)
        a_186835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'a', False)
        # Obtaining the member 'rjust' of a type (line 49)
        rjust_186836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), a_186835, 'rjust')
        # Calling rjust(args, kwargs) (line 49)
        rjust_call_result_186839 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), rjust_186836, *[m_186837], **kwargs_186838)
        
        str_186840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'str', ': ')
        # Applying the binary operator '+' (line 49)
        result_add_186841 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 26), '+', rjust_call_result_186839, str_186840)
        
        
        # Call to repr(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to getattr(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'self' (line 49)
        self_186844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 59), 'self', False)
        # Getting the type of 'a' (line 49)
        a_186845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 65), 'a', False)
        # Processing the call keyword arguments (line 49)
        kwargs_186846 = {}
        # Getting the type of 'getattr' (line 49)
        getattr_186843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 51), 'getattr', False)
        # Calling getattr(args, kwargs) (line 49)
        getattr_call_result_186847 = invoke(stypy.reporting.localization.Localization(__file__, 49, 51), getattr_186843, *[self_186844, a_186845], **kwargs_186846)
        
        # Processing the call keyword arguments (line 49)
        kwargs_186848 = {}
        # Getting the type of 'repr' (line 49)
        repr_186842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 46), 'repr', False)
        # Calling repr(args, kwargs) (line 49)
        repr_call_result_186849 = invoke(stypy.reporting.localization.Localization(__file__, 49, 46), repr_186842, *[getattr_call_result_186847], **kwargs_186848)
        
        # Applying the binary operator '+' (line 49)
        result_add_186850 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 44), '+', result_add_186841, repr_call_result_186849)
        
        list_186853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 26), list_186853, result_add_186850)
        # Processing the call keyword arguments (line 49)
        kwargs_186854 = {}
        str_186833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 49)
        join_186834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), str_186833, 'join')
        # Calling join(args, kwargs) (line 49)
        join_call_result_186855 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), join_186834, *[list_186853], **kwargs_186854)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', join_call_result_186855)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_186856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_186856


# Assigning a type to the variable 'RootResults' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'RootResults', RootResults)

@norecursion
def results_c(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'results_c'
    module_type_store = module_type_store.open_function_context('results_c', 53, 0, False)
    
    # Passed parameters checking function
    results_c.stypy_localization = localization
    results_c.stypy_type_of_self = None
    results_c.stypy_type_store = module_type_store
    results_c.stypy_function_name = 'results_c'
    results_c.stypy_param_names_list = ['full_output', 'r']
    results_c.stypy_varargs_param_name = None
    results_c.stypy_kwargs_param_name = None
    results_c.stypy_call_defaults = defaults
    results_c.stypy_call_varargs = varargs
    results_c.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'results_c', ['full_output', 'r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'results_c', localization, ['full_output', 'r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'results_c(...)' code ##################

    
    # Getting the type of 'full_output' (line 54)
    full_output_186857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'full_output')
    # Testing the type of an if condition (line 54)
    if_condition_186858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), full_output_186857)
    # Assigning a type to the variable 'if_condition_186858' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_186858', if_condition_186858)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 55):
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    int_186859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
    # Getting the type of 'r' (line 55)
    r_186860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'r')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___186861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), r_186860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_186862 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___186861, int_186859)
    
    # Assigning a type to the variable 'tuple_var_assignment_186763' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186763', subscript_call_result_186862)
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    int_186863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
    # Getting the type of 'r' (line 55)
    r_186864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'r')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___186865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), r_186864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_186866 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___186865, int_186863)
    
    # Assigning a type to the variable 'tuple_var_assignment_186764' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186764', subscript_call_result_186866)
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    int_186867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
    # Getting the type of 'r' (line 55)
    r_186868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'r')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___186869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), r_186868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_186870 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___186869, int_186867)
    
    # Assigning a type to the variable 'tuple_var_assignment_186765' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186765', subscript_call_result_186870)
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    int_186871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
    # Getting the type of 'r' (line 55)
    r_186872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'r')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___186873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), r_186872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_186874 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___186873, int_186871)
    
    # Assigning a type to the variable 'tuple_var_assignment_186766' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186766', subscript_call_result_186874)
    
    # Assigning a Name to a Name (line 55):
    # Getting the type of 'tuple_var_assignment_186763' (line 55)
    tuple_var_assignment_186763_186875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186763')
    # Assigning a type to the variable 'x' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'x', tuple_var_assignment_186763_186875)
    
    # Assigning a Name to a Name (line 55):
    # Getting the type of 'tuple_var_assignment_186764' (line 55)
    tuple_var_assignment_186764_186876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186764')
    # Assigning a type to the variable 'funcalls' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'funcalls', tuple_var_assignment_186764_186876)
    
    # Assigning a Name to a Name (line 55):
    # Getting the type of 'tuple_var_assignment_186765' (line 55)
    tuple_var_assignment_186765_186877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186765')
    # Assigning a type to the variable 'iterations' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'iterations', tuple_var_assignment_186765_186877)
    
    # Assigning a Name to a Name (line 55):
    # Getting the type of 'tuple_var_assignment_186766' (line 55)
    tuple_var_assignment_186766_186878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_186766')
    # Assigning a type to the variable 'flag' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'flag', tuple_var_assignment_186766_186878)
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to RootResults(...): (line 56)
    # Processing the call keyword arguments (line 56)
    # Getting the type of 'x' (line 56)
    x_186880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'x', False)
    keyword_186881 = x_186880
    # Getting the type of 'iterations' (line 57)
    iterations_186882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 41), 'iterations', False)
    keyword_186883 = iterations_186882
    # Getting the type of 'funcalls' (line 58)
    funcalls_186884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 45), 'funcalls', False)
    keyword_186885 = funcalls_186884
    # Getting the type of 'flag' (line 59)
    flag_186886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 35), 'flag', False)
    keyword_186887 = flag_186886
    kwargs_186888 = {'function_calls': keyword_186885, 'flag': keyword_186887, 'root': keyword_186881, 'iterations': keyword_186883}
    # Getting the type of 'RootResults' (line 56)
    RootResults_186879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'RootResults', False)
    # Calling RootResults(args, kwargs) (line 56)
    RootResults_call_result_186889 = invoke(stypy.reporting.localization.Localization(__file__, 56, 18), RootResults_186879, *[], **kwargs_186888)
    
    # Assigning a type to the variable 'results' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'results', RootResults_call_result_186889)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_186890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'x' (line 60)
    x_186891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 15), tuple_186890, x_186891)
    # Adding element type (line 60)
    # Getting the type of 'results' (line 60)
    results_186892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'results')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 15), tuple_186890, results_186892)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', tuple_186890)
    # SSA branch for the else part of an if statement (line 54)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'r' (line 62)
    r_186893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', r_186893)
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'results_c(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'results_c' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_186894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186894)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'results_c'
    return stypy_return_type_186894

# Assigning a type to the variable 'results_c' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'results_c', results_c)

@norecursion
def newton(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 66)
    None_186895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_186896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    
    float_186897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 47), 'float')
    int_186898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 64), 'int')
    # Getting the type of 'None' (line 67)
    None_186899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'None')
    defaults = [None_186895, tuple_186896, float_186897, int_186898, None_186899]
    # Create a new context for function 'newton'
    module_type_store = module_type_store.open_function_context('newton', 66, 0, False)
    
    # Passed parameters checking function
    newton.stypy_localization = localization
    newton.stypy_type_of_self = None
    newton.stypy_type_store = module_type_store
    newton.stypy_function_name = 'newton'
    newton.stypy_param_names_list = ['func', 'x0', 'fprime', 'args', 'tol', 'maxiter', 'fprime2']
    newton.stypy_varargs_param_name = None
    newton.stypy_kwargs_param_name = None
    newton.stypy_call_defaults = defaults
    newton.stypy_call_varargs = varargs
    newton.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'newton', ['func', 'x0', 'fprime', 'args', 'tol', 'maxiter', 'fprime2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'newton', localization, ['func', 'x0', 'fprime', 'args', 'tol', 'maxiter', 'fprime2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'newton(...)' code ##################

    str_186900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'str', "\n    Find a zero using the Newton-Raphson or secant method.\n\n    Find a zero of the function `func` given a nearby starting point `x0`.\n    The Newton-Raphson method is used if the derivative `fprime` of `func`\n    is provided, otherwise the secant method is used.  If the second order\n    derivate `fprime2` of `func` is provided, parabolic Halley's method\n    is used.\n\n    Parameters\n    ----------\n    func : function\n        The function whose zero is wanted. It must be a function of a\n        single variable of the form f(x,a,b,c...), where a,b,c... are extra\n        arguments that can be passed in the `args` parameter.\n    x0 : float\n        An initial estimate of the zero that should be somewhere near the\n        actual zero.\n    fprime : function, optional\n        The derivative of the function when available and convenient. If it\n        is None (default), then the secant method is used.\n    args : tuple, optional\n        Extra arguments to be used in the function call.\n    tol : float, optional\n        The allowable error of the zero value.\n    maxiter : int, optional\n        Maximum number of iterations.\n    fprime2 : function, optional\n        The second order derivative of the function when available and\n        convenient. If it is None (default), then the normal Newton-Raphson\n        or the secant method is used. If it is given, parabolic Halley's\n        method is used.\n\n    Returns\n    -------\n    zero : float\n        Estimated location where function is zero.\n\n    See Also\n    --------\n    brentq, brenth, ridder, bisect\n    fsolve : find zeroes in n dimensions.\n\n    Notes\n    -----\n    The convergence rate of the Newton-Raphson method is quadratic,\n    the Halley method is cubic, and the secant method is\n    sub-quadratic.  This means that if the function is well behaved\n    the actual error in the estimated zero is approximately the square\n    (cube for Halley) of the requested tolerance up to roundoff\n    error. However, the stopping criterion used here is the step size\n    and there is no guarantee that a zero has been found. Consequently\n    the result should be verified. Safer algorithms are brentq,\n    brenth, ridder, and bisect, but they all require that the root\n    first be bracketed in an interval where the function changes\n    sign. The brentq algorithm is recommended for general use in one\n    dimensional problems when such an interval has been found.\n\n    Examples\n    --------\n\n    >>> def f(x):\n    ...     return (x**3 - 1)  # only one real root at x = 1\n    \n    >>> from scipy import optimize\n\n    ``fprime`` and ``fprime2`` not provided, use secant method\n    \n    >>> root = optimize.newton(f, 1.5)\n    >>> root\n    1.0000000000000016\n\n    Only ``fprime`` provided, use Newton Raphson method\n    \n    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)\n    >>> root\n    1.0\n    \n    ``fprime2`` provided, ``fprime`` provided/not provided use parabolic\n    Halley's method\n\n    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)\n    >>> root\n    1.0000000000000016\n    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,\n    ...                        fprime2=lambda x: 6 * x)\n    >>> root\n    1.0\n\n    ")
    
    
    # Getting the type of 'tol' (line 158)
    tol_186901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 7), 'tol')
    int_186902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 14), 'int')
    # Applying the binary operator '<=' (line 158)
    result_le_186903 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 7), '<=', tol_186901, int_186902)
    
    # Testing the type of an if condition (line 158)
    if_condition_186904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 4), result_le_186903)
    # Assigning a type to the variable 'if_condition_186904' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'if_condition_186904', if_condition_186904)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 159)
    # Processing the call arguments (line 159)
    str_186906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'str', 'tol too small (%g <= 0)')
    # Getting the type of 'tol' (line 159)
    tol_186907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 53), 'tol', False)
    # Applying the binary operator '%' (line 159)
    result_mod_186908 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 25), '%', str_186906, tol_186907)
    
    # Processing the call keyword arguments (line 159)
    kwargs_186909 = {}
    # Getting the type of 'ValueError' (line 159)
    ValueError_186905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 159)
    ValueError_call_result_186910 = invoke(stypy.reporting.localization.Localization(__file__, 159, 14), ValueError_186905, *[result_mod_186908], **kwargs_186909)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 159, 8), ValueError_call_result_186910, 'raise parameter', BaseException)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'maxiter' (line 160)
    maxiter_186911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 7), 'maxiter')
    int_186912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 17), 'int')
    # Applying the binary operator '<' (line 160)
    result_lt_186913 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 7), '<', maxiter_186911, int_186912)
    
    # Testing the type of an if condition (line 160)
    if_condition_186914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 4), result_lt_186913)
    # Assigning a type to the variable 'if_condition_186914' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'if_condition_186914', if_condition_186914)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 161)
    # Processing the call arguments (line 161)
    str_186916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'str', 'maxiter must be greater than 0')
    # Processing the call keyword arguments (line 161)
    kwargs_186917 = {}
    # Getting the type of 'ValueError' (line 161)
    ValueError_186915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 161)
    ValueError_call_result_186918 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), ValueError_186915, *[str_186916], **kwargs_186917)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 161, 8), ValueError_call_result_186918, 'raise parameter', BaseException)
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 162)
    # Getting the type of 'fprime' (line 162)
    fprime_186919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'fprime')
    # Getting the type of 'None' (line 162)
    None_186920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'None')
    
    (may_be_186921, more_types_in_union_186922) = may_not_be_none(fprime_186919, None_186920)

    if may_be_186921:

        if more_types_in_union_186922:
            # Runtime conditional SSA (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 166):
        
        # Assigning a BinOp to a Name (line 166):
        float_186923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'float')
        # Getting the type of 'x0' (line 166)
        x0_186924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'x0')
        # Applying the binary operator '*' (line 166)
        result_mul_186925 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 13), '*', float_186923, x0_186924)
        
        # Assigning a type to the variable 'p0' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'p0', result_mul_186925)
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        int_186926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 16), 'int')
        # Assigning a type to the variable 'fder2' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'fder2', int_186926)
        
        
        # Call to range(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'maxiter' (line 168)
        maxiter_186928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'maxiter', False)
        # Processing the call keyword arguments (line 168)
        kwargs_186929 = {}
        # Getting the type of 'range' (line 168)
        range_186927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'range', False)
        # Calling range(args, kwargs) (line 168)
        range_call_result_186930 = invoke(stypy.reporting.localization.Localization(__file__, 168, 20), range_186927, *[maxiter_186928], **kwargs_186929)
        
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_186930)
        # Getting the type of the for loop variable (line 168)
        for_loop_var_186931 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_186930)
        # Assigning a type to the variable 'iter' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'iter', for_loop_var_186931)
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 169):
        
        # Assigning a BinOp to a Name (line 169):
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_186932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'p0' (line 169)
        p0_186933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'p0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), tuple_186932, p0_186933)
        
        # Getting the type of 'args' (line 169)
        args_186934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'args')
        # Applying the binary operator '+' (line 169)
        result_add_186935 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 21), '+', tuple_186932, args_186934)
        
        # Assigning a type to the variable 'myargs' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'myargs', result_add_186935)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to fprime(...): (line 170)
        # Getting the type of 'myargs' (line 170)
        myargs_186937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'myargs', False)
        # Processing the call keyword arguments (line 170)
        kwargs_186938 = {}
        # Getting the type of 'fprime' (line 170)
        fprime_186936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'fprime', False)
        # Calling fprime(args, kwargs) (line 170)
        fprime_call_result_186939 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), fprime_186936, *[myargs_186937], **kwargs_186938)
        
        # Assigning a type to the variable 'fder' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'fder', fprime_call_result_186939)
        
        
        # Getting the type of 'fder' (line 171)
        fder_186940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'fder')
        int_186941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'int')
        # Applying the binary operator '==' (line 171)
        result_eq_186942 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '==', fder_186940, int_186941)
        
        # Testing the type of an if condition (line 171)
        if_condition_186943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 12), result_eq_186942)
        # Assigning a type to the variable 'if_condition_186943' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'if_condition_186943', if_condition_186943)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 172):
        
        # Assigning a Str to a Name (line 172):
        str_186944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 22), 'str', 'derivative was zero.')
        # Assigning a type to the variable 'msg' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'msg', str_186944)
        
        # Call to warn(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'msg' (line 173)
        msg_186947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'msg', False)
        # Getting the type of 'RuntimeWarning' (line 173)
        RuntimeWarning_186948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 35), 'RuntimeWarning', False)
        # Processing the call keyword arguments (line 173)
        kwargs_186949 = {}
        # Getting the type of 'warnings' (line 173)
        warnings_186945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 173)
        warn_186946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), warnings_186945, 'warn')
        # Calling warn(args, kwargs) (line 173)
        warn_call_result_186950 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), warn_186946, *[msg_186947, RuntimeWarning_186948], **kwargs_186949)
        
        # Getting the type of 'p0' (line 174)
        p0_186951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'p0')
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'stypy_return_type', p0_186951)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to func(...): (line 175)
        # Getting the type of 'myargs' (line 175)
        myargs_186953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'myargs', False)
        # Processing the call keyword arguments (line 175)
        kwargs_186954 = {}
        # Getting the type of 'func' (line 175)
        func_186952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'func', False)
        # Calling func(args, kwargs) (line 175)
        func_call_result_186955 = invoke(stypy.reporting.localization.Localization(__file__, 175, 19), func_186952, *[myargs_186953], **kwargs_186954)
        
        # Assigning a type to the variable 'fval' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'fval', func_call_result_186955)
        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'fprime2' (line 176)
        fprime2_186956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'fprime2')
        # Getting the type of 'None' (line 176)
        None_186957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'None')
        
        (may_be_186958, more_types_in_union_186959) = may_not_be_none(fprime2_186956, None_186957)

        if may_be_186958:

            if more_types_in_union_186959:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 177):
            
            # Assigning a Call to a Name (line 177):
            
            # Call to fprime2(...): (line 177)
            # Getting the type of 'myargs' (line 177)
            myargs_186961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 33), 'myargs', False)
            # Processing the call keyword arguments (line 177)
            kwargs_186962 = {}
            # Getting the type of 'fprime2' (line 177)
            fprime2_186960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'fprime2', False)
            # Calling fprime2(args, kwargs) (line 177)
            fprime2_call_result_186963 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), fprime2_186960, *[myargs_186961], **kwargs_186962)
            
            # Assigning a type to the variable 'fder2' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'fder2', fprime2_call_result_186963)

            if more_types_in_union_186959:
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'fder2' (line 178)
        fder2_186964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'fder2')
        int_186965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'int')
        # Applying the binary operator '==' (line 178)
        result_eq_186966 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 15), '==', fder2_186964, int_186965)
        
        # Testing the type of an if condition (line 178)
        if_condition_186967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), result_eq_186966)
        # Assigning a type to the variable 'if_condition_186967' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'if_condition_186967', if_condition_186967)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 180):
        
        # Assigning a BinOp to a Name (line 180):
        # Getting the type of 'p0' (line 180)
        p0_186968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'p0')
        # Getting the type of 'fval' (line 180)
        fval_186969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'fval')
        # Getting the type of 'fder' (line 180)
        fder_186970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 32), 'fder')
        # Applying the binary operator 'div' (line 180)
        result_div_186971 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 25), 'div', fval_186969, fder_186970)
        
        # Applying the binary operator '-' (line 180)
        result_sub_186972 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 20), '-', p0_186968, result_div_186971)
        
        # Assigning a type to the variable 'p' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'p', result_sub_186972)
        # SSA branch for the else part of an if statement (line 178)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 183):
        
        # Assigning a BinOp to a Name (line 183):
        # Getting the type of 'fder' (line 183)
        fder_186973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'fder')
        int_186974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 32), 'int')
        # Applying the binary operator '**' (line 183)
        result_pow_186975 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 24), '**', fder_186973, int_186974)
        
        int_186976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'int')
        # Getting the type of 'fval' (line 183)
        fval_186977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 40), 'fval')
        # Applying the binary operator '*' (line 183)
        result_mul_186978 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 36), '*', int_186976, fval_186977)
        
        # Getting the type of 'fder2' (line 183)
        fder2_186979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 47), 'fder2')
        # Applying the binary operator '*' (line 183)
        result_mul_186980 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 45), '*', result_mul_186978, fder2_186979)
        
        # Applying the binary operator '-' (line 183)
        result_sub_186981 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 24), '-', result_pow_186975, result_mul_186980)
        
        # Assigning a type to the variable 'discr' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'discr', result_sub_186981)
        
        
        # Getting the type of 'discr' (line 184)
        discr_186982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'discr')
        int_186983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 27), 'int')
        # Applying the binary operator '<' (line 184)
        result_lt_186984 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 19), '<', discr_186982, int_186983)
        
        # Testing the type of an if condition (line 184)
        if_condition_186985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 16), result_lt_186984)
        # Assigning a type to the variable 'if_condition_186985' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'if_condition_186985', if_condition_186985)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 185):
        
        # Assigning a BinOp to a Name (line 185):
        # Getting the type of 'p0' (line 185)
        p0_186986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'p0')
        # Getting the type of 'fder' (line 185)
        fder_186987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'fder')
        # Getting the type of 'fder2' (line 185)
        fder2_186988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 36), 'fder2')
        # Applying the binary operator 'div' (line 185)
        result_div_186989 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 29), 'div', fder_186987, fder2_186988)
        
        # Applying the binary operator '-' (line 185)
        result_sub_186990 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 24), '-', p0_186986, result_div_186989)
        
        # Assigning a type to the variable 'p' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'p', result_sub_186990)
        # SSA branch for the else part of an if statement (line 184)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 187):
        
        # Assigning a BinOp to a Name (line 187):
        # Getting the type of 'p0' (line 187)
        p0_186991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'p0')
        int_186992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 29), 'int')
        # Getting the type of 'fval' (line 187)
        fval_186993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'fval')
        # Applying the binary operator '*' (line 187)
        result_mul_186994 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 29), '*', int_186992, fval_186993)
        
        # Getting the type of 'fder' (line 187)
        fder_186995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 39), 'fder')
        
        # Call to sign(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'fder' (line 187)
        fder_186997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 51), 'fder', False)
        # Processing the call keyword arguments (line 187)
        kwargs_186998 = {}
        # Getting the type of 'sign' (line 187)
        sign_186996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 46), 'sign', False)
        # Calling sign(args, kwargs) (line 187)
        sign_call_result_186999 = invoke(stypy.reporting.localization.Localization(__file__, 187, 46), sign_186996, *[fder_186997], **kwargs_186998)
        
        
        # Call to sqrt(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'discr' (line 187)
        discr_187001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 64), 'discr', False)
        # Processing the call keyword arguments (line 187)
        kwargs_187002 = {}
        # Getting the type of 'sqrt' (line 187)
        sqrt_187000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 59), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 187)
        sqrt_call_result_187003 = invoke(stypy.reporting.localization.Localization(__file__, 187, 59), sqrt_187000, *[discr_187001], **kwargs_187002)
        
        # Applying the binary operator '*' (line 187)
        result_mul_187004 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 46), '*', sign_call_result_186999, sqrt_call_result_187003)
        
        # Applying the binary operator '+' (line 187)
        result_add_187005 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 39), '+', fder_186995, result_mul_187004)
        
        # Applying the binary operator 'div' (line 187)
        result_div_187006 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 36), 'div', result_mul_186994, result_add_187005)
        
        # Applying the binary operator '-' (line 187)
        result_sub_187007 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 24), '-', p0_186991, result_div_187006)
        
        # Assigning a type to the variable 'p' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'p', result_sub_187007)
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'p' (line 188)
        p_187009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'p', False)
        # Getting the type of 'p0' (line 188)
        p0_187010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'p0', False)
        # Applying the binary operator '-' (line 188)
        result_sub_187011 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 19), '-', p_187009, p0_187010)
        
        # Processing the call keyword arguments (line 188)
        kwargs_187012 = {}
        # Getting the type of 'abs' (line 188)
        abs_187008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 188)
        abs_call_result_187013 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), abs_187008, *[result_sub_187011], **kwargs_187012)
        
        # Getting the type of 'tol' (line 188)
        tol_187014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'tol')
        # Applying the binary operator '<' (line 188)
        result_lt_187015 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 15), '<', abs_call_result_187013, tol_187014)
        
        # Testing the type of an if condition (line 188)
        if_condition_187016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 12), result_lt_187015)
        # Assigning a type to the variable 'if_condition_187016' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'if_condition_187016', if_condition_187016)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'p' (line 189)
        p_187017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'stypy_return_type', p_187017)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 190):
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'p' (line 190)
        p_187018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'p')
        # Assigning a type to the variable 'p0' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'p0', p_187018)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_186922:
            # Runtime conditional SSA for else branch (line 162)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_186921) or more_types_in_union_186922):
        
        # Assigning a Name to a Name (line 193):
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'x0' (line 193)
        x0_187019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'x0')
        # Assigning a type to the variable 'p0' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'p0', x0_187019)
        
        
        # Getting the type of 'x0' (line 194)
        x0_187020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'x0')
        int_187021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'int')
        # Applying the binary operator '>=' (line 194)
        result_ge_187022 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), '>=', x0_187020, int_187021)
        
        # Testing the type of an if condition (line 194)
        if_condition_187023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_ge_187022)
        # Assigning a type to the variable 'if_condition_187023' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_187023', if_condition_187023)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 195):
        
        # Assigning a BinOp to a Name (line 195):
        # Getting the type of 'x0' (line 195)
        x0_187024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'x0')
        int_187025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 21), 'int')
        float_187026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'float')
        # Applying the binary operator '+' (line 195)
        result_add_187027 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 21), '+', int_187025, float_187026)
        
        # Applying the binary operator '*' (line 195)
        result_mul_187028 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 17), '*', x0_187024, result_add_187027)
        
        float_187029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 33), 'float')
        # Applying the binary operator '+' (line 195)
        result_add_187030 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 17), '+', result_mul_187028, float_187029)
        
        # Assigning a type to the variable 'p1' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'p1', result_add_187030)
        # SSA branch for the else part of an if statement (line 194)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 197):
        
        # Assigning a BinOp to a Name (line 197):
        # Getting the type of 'x0' (line 197)
        x0_187031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'x0')
        int_187032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'int')
        float_187033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 25), 'float')
        # Applying the binary operator '+' (line 197)
        result_add_187034 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 21), '+', int_187032, float_187033)
        
        # Applying the binary operator '*' (line 197)
        result_mul_187035 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 17), '*', x0_187031, result_add_187034)
        
        float_187036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'float')
        # Applying the binary operator '-' (line 197)
        result_sub_187037 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 17), '-', result_mul_187035, float_187036)
        
        # Assigning a type to the variable 'p1' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'p1', result_sub_187037)
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to func(...): (line 198)
        
        # Obtaining an instance of the builtin type 'tuple' (line 198)
        tuple_187039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 198)
        # Adding element type (line 198)
        # Getting the type of 'p0' (line 198)
        p0_187040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'p0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 21), tuple_187039, p0_187040)
        
        # Getting the type of 'args' (line 198)
        args_187041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'args', False)
        # Applying the binary operator '+' (line 198)
        result_add_187042 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 20), '+', tuple_187039, args_187041)
        
        # Processing the call keyword arguments (line 198)
        kwargs_187043 = {}
        # Getting the type of 'func' (line 198)
        func_187038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 13), 'func', False)
        # Calling func(args, kwargs) (line 198)
        func_call_result_187044 = invoke(stypy.reporting.localization.Localization(__file__, 198, 13), func_187038, *[result_add_187042], **kwargs_187043)
        
        # Assigning a type to the variable 'q0' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'q0', func_call_result_187044)
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to func(...): (line 199)
        
        # Obtaining an instance of the builtin type 'tuple' (line 199)
        tuple_187046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 199)
        # Adding element type (line 199)
        # Getting the type of 'p1' (line 199)
        p1_187047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'p1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), tuple_187046, p1_187047)
        
        # Getting the type of 'args' (line 199)
        args_187048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'args', False)
        # Applying the binary operator '+' (line 199)
        result_add_187049 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 20), '+', tuple_187046, args_187048)
        
        # Processing the call keyword arguments (line 199)
        kwargs_187050 = {}
        # Getting the type of 'func' (line 199)
        func_187045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'func', False)
        # Calling func(args, kwargs) (line 199)
        func_call_result_187051 = invoke(stypy.reporting.localization.Localization(__file__, 199, 13), func_187045, *[result_add_187049], **kwargs_187050)
        
        # Assigning a type to the variable 'q1' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'q1', func_call_result_187051)
        
        
        # Call to range(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'maxiter' (line 200)
        maxiter_187053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'maxiter', False)
        # Processing the call keyword arguments (line 200)
        kwargs_187054 = {}
        # Getting the type of 'range' (line 200)
        range_187052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'range', False)
        # Calling range(args, kwargs) (line 200)
        range_call_result_187055 = invoke(stypy.reporting.localization.Localization(__file__, 200, 20), range_187052, *[maxiter_187053], **kwargs_187054)
        
        # Testing the type of a for loop iterable (line 200)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 200, 8), range_call_result_187055)
        # Getting the type of the for loop variable (line 200)
        for_loop_var_187056 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 200, 8), range_call_result_187055)
        # Assigning a type to the variable 'iter' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'iter', for_loop_var_187056)
        # SSA begins for a for statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'q1' (line 201)
        q1_187057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'q1')
        # Getting the type of 'q0' (line 201)
        q0_187058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 21), 'q0')
        # Applying the binary operator '==' (line 201)
        result_eq_187059 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 15), '==', q1_187057, q0_187058)
        
        # Testing the type of an if condition (line 201)
        if_condition_187060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), result_eq_187059)
        # Assigning a type to the variable 'if_condition_187060' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_187060', if_condition_187060)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'p1' (line 202)
        p1_187061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'p1')
        # Getting the type of 'p0' (line 202)
        p0_187062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'p0')
        # Applying the binary operator '!=' (line 202)
        result_ne_187063 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 19), '!=', p1_187061, p0_187062)
        
        # Testing the type of an if condition (line 202)
        if_condition_187064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 16), result_ne_187063)
        # Assigning a type to the variable 'if_condition_187064' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'if_condition_187064', if_condition_187064)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 203):
        
        # Assigning a BinOp to a Name (line 203):
        str_187065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 26), 'str', 'Tolerance of %s reached')
        # Getting the type of 'p1' (line 203)
        p1_187066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 55), 'p1')
        # Getting the type of 'p0' (line 203)
        p0_187067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 60), 'p0')
        # Applying the binary operator '-' (line 203)
        result_sub_187068 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 55), '-', p1_187066, p0_187067)
        
        # Applying the binary operator '%' (line 203)
        result_mod_187069 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 26), '%', str_187065, result_sub_187068)
        
        # Assigning a type to the variable 'msg' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'msg', result_mod_187069)
        
        # Call to warn(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'msg' (line 204)
        msg_187072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'msg', False)
        # Getting the type of 'RuntimeWarning' (line 204)
        RuntimeWarning_187073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 39), 'RuntimeWarning', False)
        # Processing the call keyword arguments (line 204)
        kwargs_187074 = {}
        # Getting the type of 'warnings' (line 204)
        warnings_187070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 204)
        warn_187071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 20), warnings_187070, 'warn')
        # Calling warn(args, kwargs) (line 204)
        warn_call_result_187075 = invoke(stypy.reporting.localization.Localization(__file__, 204, 20), warn_187071, *[msg_187072, RuntimeWarning_187073], **kwargs_187074)
        
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'p1' (line 205)
        p1_187076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'p1')
        # Getting the type of 'p0' (line 205)
        p0_187077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'p0')
        # Applying the binary operator '+' (line 205)
        result_add_187078 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 24), '+', p1_187076, p0_187077)
        
        float_187079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 33), 'float')
        # Applying the binary operator 'div' (line 205)
        result_div_187080 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 23), 'div', result_add_187078, float_187079)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'stypy_return_type', result_div_187080)
        # SSA branch for the else part of an if statement (line 201)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 207):
        
        # Assigning a BinOp to a Name (line 207):
        # Getting the type of 'p1' (line 207)
        p1_187081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'p1')
        # Getting the type of 'q1' (line 207)
        q1_187082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 'q1')
        # Getting the type of 'p1' (line 207)
        p1_187083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'p1')
        # Getting the type of 'p0' (line 207)
        p0_187084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'p0')
        # Applying the binary operator '-' (line 207)
        result_sub_187085 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 29), '-', p1_187083, p0_187084)
        
        # Applying the binary operator '*' (line 207)
        result_mul_187086 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 25), '*', q1_187082, result_sub_187085)
        
        # Getting the type of 'q1' (line 207)
        q1_187087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 'q1')
        # Getting the type of 'q0' (line 207)
        q0_187088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'q0')
        # Applying the binary operator '-' (line 207)
        result_sub_187089 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 39), '-', q1_187087, q0_187088)
        
        # Applying the binary operator 'div' (line 207)
        result_div_187090 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 37), 'div', result_mul_187086, result_sub_187089)
        
        # Applying the binary operator '-' (line 207)
        result_sub_187091 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 20), '-', p1_187081, result_div_187090)
        
        # Assigning a type to the variable 'p' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'p', result_sub_187091)
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'p' (line 208)
        p_187093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'p', False)
        # Getting the type of 'p1' (line 208)
        p1_187094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'p1', False)
        # Applying the binary operator '-' (line 208)
        result_sub_187095 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 19), '-', p_187093, p1_187094)
        
        # Processing the call keyword arguments (line 208)
        kwargs_187096 = {}
        # Getting the type of 'abs' (line 208)
        abs_187092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 208)
        abs_call_result_187097 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), abs_187092, *[result_sub_187095], **kwargs_187096)
        
        # Getting the type of 'tol' (line 208)
        tol_187098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'tol')
        # Applying the binary operator '<' (line 208)
        result_lt_187099 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '<', abs_call_result_187097, tol_187098)
        
        # Testing the type of an if condition (line 208)
        if_condition_187100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 12), result_lt_187099)
        # Assigning a type to the variable 'if_condition_187100' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'if_condition_187100', if_condition_187100)
        # SSA begins for if statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'p' (line 209)
        p_187101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'stypy_return_type', p_187101)
        # SSA join for if statement (line 208)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 210):
        
        # Assigning a Name to a Name (line 210):
        # Getting the type of 'p1' (line 210)
        p1_187102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'p1')
        # Assigning a type to the variable 'p0' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'p0', p1_187102)
        
        # Assigning a Name to a Name (line 211):
        
        # Assigning a Name to a Name (line 211):
        # Getting the type of 'q1' (line 211)
        q1_187103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'q1')
        # Assigning a type to the variable 'q0' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'q0', q1_187103)
        
        # Assigning a Name to a Name (line 212):
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'p' (line 212)
        p_187104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'p')
        # Assigning a type to the variable 'p1' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'p1', p_187104)
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to func(...): (line 213)
        
        # Obtaining an instance of the builtin type 'tuple' (line 213)
        tuple_187106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 213)
        # Adding element type (line 213)
        # Getting the type of 'p1' (line 213)
        p1_187107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'p1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 25), tuple_187106, p1_187107)
        
        # Getting the type of 'args' (line 213)
        args_187108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'args', False)
        # Applying the binary operator '+' (line 213)
        result_add_187109 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 24), '+', tuple_187106, args_187108)
        
        # Processing the call keyword arguments (line 213)
        kwargs_187110 = {}
        # Getting the type of 'func' (line 213)
        func_187105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'func', False)
        # Calling func(args, kwargs) (line 213)
        func_call_result_187111 = invoke(stypy.reporting.localization.Localization(__file__, 213, 17), func_187105, *[result_add_187109], **kwargs_187110)
        
        # Assigning a type to the variable 'q1' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'q1', func_call_result_187111)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_186921 and more_types_in_union_186922):
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 214):
    
    # Assigning a BinOp to a Name (line 214):
    str_187112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 10), 'str', 'Failed to converge after %d iterations, value is %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_187113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    # Getting the type of 'maxiter' (line 214)
    maxiter_187114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 67), 'maxiter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 67), tuple_187113, maxiter_187114)
    # Adding element type (line 214)
    # Getting the type of 'p' (line 214)
    p_187115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 76), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 67), tuple_187113, p_187115)
    
    # Applying the binary operator '%' (line 214)
    result_mod_187116 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 10), '%', str_187112, tuple_187113)
    
    # Assigning a type to the variable 'msg' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'msg', result_mod_187116)
    
    # Call to RuntimeError(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'msg' (line 215)
    msg_187118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'msg', False)
    # Processing the call keyword arguments (line 215)
    kwargs_187119 = {}
    # Getting the type of 'RuntimeError' (line 215)
    RuntimeError_187117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 10), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 215)
    RuntimeError_call_result_187120 = invoke(stypy.reporting.localization.Localization(__file__, 215, 10), RuntimeError_187117, *[msg_187118], **kwargs_187119)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 4), RuntimeError_call_result_187120, 'raise parameter', BaseException)
    
    # ################# End of 'newton(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'newton' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_187121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'newton'
    return stypy_return_type_187121

# Assigning a type to the variable 'newton' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'newton', newton)

@norecursion
def bisect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 218)
    tuple_187122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 218)
    
    # Getting the type of '_xtol' (line 219)
    _xtol_187123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), '_xtol')
    # Getting the type of '_rtol' (line 219)
    _rtol_187124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), '_rtol')
    # Getting the type of '_iter' (line 219)
    _iter_187125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 43), '_iter')
    # Getting the type of 'False' (line 220)
    False_187126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'False')
    # Getting the type of 'True' (line 220)
    True_187127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 'True')
    defaults = [tuple_187122, _xtol_187123, _rtol_187124, _iter_187125, False_187126, True_187127]
    # Create a new context for function 'bisect'
    module_type_store = module_type_store.open_function_context('bisect', 218, 0, False)
    
    # Passed parameters checking function
    bisect.stypy_localization = localization
    bisect.stypy_type_of_self = None
    bisect.stypy_type_store = module_type_store
    bisect.stypy_function_name = 'bisect'
    bisect.stypy_param_names_list = ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp']
    bisect.stypy_varargs_param_name = None
    bisect.stypy_kwargs_param_name = None
    bisect.stypy_call_defaults = defaults
    bisect.stypy_call_varargs = varargs
    bisect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bisect', ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bisect', localization, ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bisect(...)' code ##################

    str_187128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', "\n    Find root of a function within an interval.\n\n    Basic bisection routine to find a zero of the function `f` between the\n    arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.\n    Slow but sure.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number.  `f` must be continuous, and\n        f(a) and f(b) must have opposite signs.\n    a : number\n        One end of the bracketing interval [a,b].\n    b : number\n        The other end of the bracketing interval [a,b].\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be nonnegative.\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``.\n    maxiter : number, optional\n        if convergence is not achieved in `maxiter` iterations, an error is\n        raised.  Must be >= 0.\n    args : tuple, optional\n        containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned.  If `full_output` is\n        True, the return value is ``(x, r)``, where x is the root, and r is\n        a `RootResults` object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn't converge.\n\n    Returns\n    -------\n    x0 : float\n        Zero of `f` between `a` and `b`.\n    r : RootResults (present if ``full_output = True``)\n        Object containing information about the convergence.  In particular,\n        ``r.converged`` is True if the routine converged.\n\n    Examples\n    --------\n\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.bisect(f, 0, 2)\n    >>> root\n    1.0\n\n    >>> root = optimize.bisect(f, -2, 0)\n    >>> root\n    -1.0\n\n    See Also\n    --------\n    brentq, brenth, bisect, newton\n    fixed_point : scalar fixed-point finder\n    fsolve : n-dimensional root-finding\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 290)
    # Getting the type of 'tuple' (line 290)
    tuple_187129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 28), 'tuple')
    # Getting the type of 'args' (line 290)
    args_187130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'args')
    
    (may_be_187131, more_types_in_union_187132) = may_not_be_subtype(tuple_187129, args_187130)

    if may_be_187131:

        if more_types_in_union_187132:
            # Runtime conditional SSA (line 290)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'args', remove_subtype_from_union(args_187130, tuple))
        
        # Assigning a Tuple to a Name (line 291):
        
        # Assigning a Tuple to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_187133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        # Getting the type of 'args' (line 291)
        args_187134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 16), tuple_187133, args_187134)
        
        # Assigning a type to the variable 'args' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'args', tuple_187133)

        if more_types_in_union_187132:
            # SSA join for if statement (line 290)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'xtol' (line 292)
    xtol_187135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'xtol')
    int_187136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 15), 'int')
    # Applying the binary operator '<=' (line 292)
    result_le_187137 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 7), '<=', xtol_187135, int_187136)
    
    # Testing the type of an if condition (line 292)
    if_condition_187138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 4), result_le_187137)
    # Assigning a type to the variable 'if_condition_187138' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'if_condition_187138', if_condition_187138)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 293)
    # Processing the call arguments (line 293)
    str_187140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 25), 'str', 'xtol too small (%g <= 0)')
    # Getting the type of 'xtol' (line 293)
    xtol_187141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 54), 'xtol', False)
    # Applying the binary operator '%' (line 293)
    result_mod_187142 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 25), '%', str_187140, xtol_187141)
    
    # Processing the call keyword arguments (line 293)
    kwargs_187143 = {}
    # Getting the type of 'ValueError' (line 293)
    ValueError_187139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 293)
    ValueError_call_result_187144 = invoke(stypy.reporting.localization.Localization(__file__, 293, 14), ValueError_187139, *[result_mod_187142], **kwargs_187143)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 293, 8), ValueError_call_result_187144, 'raise parameter', BaseException)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rtol' (line 294)
    rtol_187145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 7), 'rtol')
    # Getting the type of '_rtol' (line 294)
    _rtol_187146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), '_rtol')
    # Applying the binary operator '<' (line 294)
    result_lt_187147 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), '<', rtol_187145, _rtol_187146)
    
    # Testing the type of an if condition (line 294)
    if_condition_187148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 4), result_lt_187147)
    # Assigning a type to the variable 'if_condition_187148' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'if_condition_187148', if_condition_187148)
    # SSA begins for if statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 295)
    # Processing the call arguments (line 295)
    str_187150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 25), 'str', 'rtol too small (%g < %g)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 295)
    tuple_187151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 295)
    # Adding element type (line 295)
    # Getting the type of 'rtol' (line 295)
    rtol_187152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 55), 'rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 55), tuple_187151, rtol_187152)
    # Adding element type (line 295)
    # Getting the type of '_rtol' (line 295)
    _rtol_187153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 61), '_rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 55), tuple_187151, _rtol_187153)
    
    # Applying the binary operator '%' (line 295)
    result_mod_187154 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 25), '%', str_187150, tuple_187151)
    
    # Processing the call keyword arguments (line 295)
    kwargs_187155 = {}
    # Getting the type of 'ValueError' (line 295)
    ValueError_187149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 295)
    ValueError_call_result_187156 = invoke(stypy.reporting.localization.Localization(__file__, 295, 14), ValueError_187149, *[result_mod_187154], **kwargs_187155)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 295, 8), ValueError_call_result_187156, 'raise parameter', BaseException)
    # SSA join for if statement (line 294)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 296):
    
    # Assigning a Call to a Name (line 296):
    
    # Call to _bisect(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'f' (line 296)
    f_187159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 23), 'f', False)
    # Getting the type of 'a' (line 296)
    a_187160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'a', False)
    # Getting the type of 'b' (line 296)
    b_187161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'b', False)
    # Getting the type of 'xtol' (line 296)
    xtol_187162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'xtol', False)
    # Getting the type of 'rtol' (line 296)
    rtol_187163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'rtol', False)
    # Getting the type of 'maxiter' (line 296)
    maxiter_187164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'maxiter', False)
    # Getting the type of 'args' (line 296)
    args_187165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'args', False)
    # Getting the type of 'full_output' (line 296)
    full_output_187166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 52), 'full_output', False)
    # Getting the type of 'disp' (line 296)
    disp_187167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 64), 'disp', False)
    # Processing the call keyword arguments (line 296)
    kwargs_187168 = {}
    # Getting the type of '_zeros' (line 296)
    _zeros_187157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), '_zeros', False)
    # Obtaining the member '_bisect' of a type (line 296)
    _bisect_187158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), _zeros_187157, '_bisect')
    # Calling _bisect(args, kwargs) (line 296)
    _bisect_call_result_187169 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), _bisect_187158, *[f_187159, a_187160, b_187161, xtol_187162, rtol_187163, maxiter_187164, args_187165, full_output_187166, disp_187167], **kwargs_187168)
    
    # Assigning a type to the variable 'r' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'r', _bisect_call_result_187169)
    
    # Call to results_c(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'full_output' (line 297)
    full_output_187171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'full_output', False)
    # Getting the type of 'r' (line 297)
    r_187172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 34), 'r', False)
    # Processing the call keyword arguments (line 297)
    kwargs_187173 = {}
    # Getting the type of 'results_c' (line 297)
    results_c_187170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'results_c', False)
    # Calling results_c(args, kwargs) (line 297)
    results_c_call_result_187174 = invoke(stypy.reporting.localization.Localization(__file__, 297, 11), results_c_187170, *[full_output_187171, r_187172], **kwargs_187173)
    
    # Assigning a type to the variable 'stypy_return_type' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'stypy_return_type', results_c_call_result_187174)
    
    # ################# End of 'bisect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bisect' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_187175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187175)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bisect'
    return stypy_return_type_187175

# Assigning a type to the variable 'bisect' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'bisect', bisect)

@norecursion
def ridder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 300)
    tuple_187176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 300)
    
    # Getting the type of '_xtol' (line 301)
    _xtol_187177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), '_xtol')
    # Getting the type of '_rtol' (line 301)
    _rtol_187178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), '_rtol')
    # Getting the type of '_iter' (line 301)
    _iter_187179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), '_iter')
    # Getting the type of 'False' (line 302)
    False_187180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'False')
    # Getting the type of 'True' (line 302)
    True_187181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 35), 'True')
    defaults = [tuple_187176, _xtol_187177, _rtol_187178, _iter_187179, False_187180, True_187181]
    # Create a new context for function 'ridder'
    module_type_store = module_type_store.open_function_context('ridder', 300, 0, False)
    
    # Passed parameters checking function
    ridder.stypy_localization = localization
    ridder.stypy_type_of_self = None
    ridder.stypy_type_store = module_type_store
    ridder.stypy_function_name = 'ridder'
    ridder.stypy_param_names_list = ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp']
    ridder.stypy_varargs_param_name = None
    ridder.stypy_kwargs_param_name = None
    ridder.stypy_call_defaults = defaults
    ridder.stypy_call_varargs = varargs
    ridder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ridder', ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ridder', localization, ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ridder(...)' code ##################

    str_187182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, (-1)), 'str', '\n    Find a root of a function in an interval.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number.  f must be continuous, and f(a) and\n        f(b) must have opposite signs.\n    a : number\n        One end of the bracketing interval [a,b].\n    b : number\n        The other end of the bracketing interval [a,b].\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be nonnegative.\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``.\n    maxiter : number, optional\n        if convergence is not achieved in maxiter iterations, an error is\n        raised.  Must be >= 0.\n    args : tuple, optional\n        containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned.  If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a RootResults object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn\'t converge.\n\n    Returns\n    -------\n    x0 : float\n        Zero of `f` between `a` and `b`.\n    r : RootResults (present if ``full_output = True``)\n        Object containing information about the convergence.\n        In particular, ``r.converged`` is True if the routine converged.\n\n    See Also\n    --------\n    brentq, brenth, bisect, newton : one-dimensional root-finding\n    fixed_point : scalar fixed-point finder\n\n    Notes\n    -----\n    Uses [Ridders1979]_ method to find a zero of the function `f` between the\n    arguments `a` and `b`. Ridders\' method is faster than bisection, but not\n    generally as fast as the Brent rountines. [Ridders1979]_ provides the\n    classic description and source of the algorithm. A description can also be\n    found in any recent edition of Numerical Recipes.\n\n    The routine used here diverges slightly from standard presentations in\n    order to be a bit more careful of tolerance.\n\n    Examples\n    --------\n\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.ridder(f, 0, 2)\n    >>> root\n    1.0\n\n    >>> root = optimize.ridder(f, -2, 0)\n    >>> root\n    -1.0\n\n    References\n    ----------\n    .. [Ridders1979]\n       Ridders, C. F. J. "A New Algorithm for Computing a\n       Single Root of a Real Continuous Function."\n       IEEE Trans. Circuits Systems 26, 979-980, 1979.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 385)
    # Getting the type of 'tuple' (line 385)
    tuple_187183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'tuple')
    # Getting the type of 'args' (line 385)
    args_187184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 22), 'args')
    
    (may_be_187185, more_types_in_union_187186) = may_not_be_subtype(tuple_187183, args_187184)

    if may_be_187185:

        if more_types_in_union_187186:
            # Runtime conditional SSA (line 385)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'args', remove_subtype_from_union(args_187184, tuple))
        
        # Assigning a Tuple to a Name (line 386):
        
        # Assigning a Tuple to a Name (line 386):
        
        # Obtaining an instance of the builtin type 'tuple' (line 386)
        tuple_187187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 386)
        # Adding element type (line 386)
        # Getting the type of 'args' (line 386)
        args_187188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 16), tuple_187187, args_187188)
        
        # Assigning a type to the variable 'args' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'args', tuple_187187)

        if more_types_in_union_187186:
            # SSA join for if statement (line 385)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'xtol' (line 387)
    xtol_187189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 7), 'xtol')
    int_187190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'int')
    # Applying the binary operator '<=' (line 387)
    result_le_187191 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 7), '<=', xtol_187189, int_187190)
    
    # Testing the type of an if condition (line 387)
    if_condition_187192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 4), result_le_187191)
    # Assigning a type to the variable 'if_condition_187192' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'if_condition_187192', if_condition_187192)
    # SSA begins for if statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 388)
    # Processing the call arguments (line 388)
    str_187194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 25), 'str', 'xtol too small (%g <= 0)')
    # Getting the type of 'xtol' (line 388)
    xtol_187195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 54), 'xtol', False)
    # Applying the binary operator '%' (line 388)
    result_mod_187196 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 25), '%', str_187194, xtol_187195)
    
    # Processing the call keyword arguments (line 388)
    kwargs_187197 = {}
    # Getting the type of 'ValueError' (line 388)
    ValueError_187193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 388)
    ValueError_call_result_187198 = invoke(stypy.reporting.localization.Localization(__file__, 388, 14), ValueError_187193, *[result_mod_187196], **kwargs_187197)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 388, 8), ValueError_call_result_187198, 'raise parameter', BaseException)
    # SSA join for if statement (line 387)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rtol' (line 389)
    rtol_187199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 'rtol')
    # Getting the type of '_rtol' (line 389)
    _rtol_187200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 14), '_rtol')
    # Applying the binary operator '<' (line 389)
    result_lt_187201 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), '<', rtol_187199, _rtol_187200)
    
    # Testing the type of an if condition (line 389)
    if_condition_187202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 4), result_lt_187201)
    # Assigning a type to the variable 'if_condition_187202' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'if_condition_187202', if_condition_187202)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 390)
    # Processing the call arguments (line 390)
    str_187204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'str', 'rtol too small (%g < %g)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 390)
    tuple_187205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 390)
    # Adding element type (line 390)
    # Getting the type of 'rtol' (line 390)
    rtol_187206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 55), 'rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 55), tuple_187205, rtol_187206)
    # Adding element type (line 390)
    # Getting the type of '_rtol' (line 390)
    _rtol_187207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 61), '_rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 55), tuple_187205, _rtol_187207)
    
    # Applying the binary operator '%' (line 390)
    result_mod_187208 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 25), '%', str_187204, tuple_187205)
    
    # Processing the call keyword arguments (line 390)
    kwargs_187209 = {}
    # Getting the type of 'ValueError' (line 390)
    ValueError_187203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 390)
    ValueError_call_result_187210 = invoke(stypy.reporting.localization.Localization(__file__, 390, 14), ValueError_187203, *[result_mod_187208], **kwargs_187209)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 390, 8), ValueError_call_result_187210, 'raise parameter', BaseException)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 391):
    
    # Assigning a Call to a Name (line 391):
    
    # Call to _ridder(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'f' (line 391)
    f_187213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'f', False)
    # Getting the type of 'a' (line 391)
    a_187214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'a', False)
    # Getting the type of 'b' (line 391)
    b_187215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 27), 'b', False)
    # Getting the type of 'xtol' (line 391)
    xtol_187216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'xtol', False)
    # Getting the type of 'rtol' (line 391)
    rtol_187217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 34), 'rtol', False)
    # Getting the type of 'maxiter' (line 391)
    maxiter_187218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 39), 'maxiter', False)
    # Getting the type of 'args' (line 391)
    args_187219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 47), 'args', False)
    # Getting the type of 'full_output' (line 391)
    full_output_187220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 52), 'full_output', False)
    # Getting the type of 'disp' (line 391)
    disp_187221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 64), 'disp', False)
    # Processing the call keyword arguments (line 391)
    kwargs_187222 = {}
    # Getting the type of '_zeros' (line 391)
    _zeros_187211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), '_zeros', False)
    # Obtaining the member '_ridder' of a type (line 391)
    _ridder_187212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), _zeros_187211, '_ridder')
    # Calling _ridder(args, kwargs) (line 391)
    _ridder_call_result_187223 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), _ridder_187212, *[f_187213, a_187214, b_187215, xtol_187216, rtol_187217, maxiter_187218, args_187219, full_output_187220, disp_187221], **kwargs_187222)
    
    # Assigning a type to the variable 'r' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'r', _ridder_call_result_187223)
    
    # Call to results_c(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'full_output' (line 392)
    full_output_187225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'full_output', False)
    # Getting the type of 'r' (line 392)
    r_187226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 34), 'r', False)
    # Processing the call keyword arguments (line 392)
    kwargs_187227 = {}
    # Getting the type of 'results_c' (line 392)
    results_c_187224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'results_c', False)
    # Calling results_c(args, kwargs) (line 392)
    results_c_call_result_187228 = invoke(stypy.reporting.localization.Localization(__file__, 392, 11), results_c_187224, *[full_output_187225, r_187226], **kwargs_187227)
    
    # Assigning a type to the variable 'stypy_return_type' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type', results_c_call_result_187228)
    
    # ################# End of 'ridder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ridder' in the type store
    # Getting the type of 'stypy_return_type' (line 300)
    stypy_return_type_187229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ridder'
    return stypy_return_type_187229

# Assigning a type to the variable 'ridder' (line 300)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'ridder', ridder)

@norecursion
def brentq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 395)
    tuple_187230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 395)
    
    # Getting the type of '_xtol' (line 396)
    _xtol_187231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), '_xtol')
    # Getting the type of '_rtol' (line 396)
    _rtol_187232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 28), '_rtol')
    # Getting the type of '_iter' (line 396)
    _iter_187233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 43), '_iter')
    # Getting the type of 'False' (line 397)
    False_187234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 23), 'False')
    # Getting the type of 'True' (line 397)
    True_187235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 35), 'True')
    defaults = [tuple_187230, _xtol_187231, _rtol_187232, _iter_187233, False_187234, True_187235]
    # Create a new context for function 'brentq'
    module_type_store = module_type_store.open_function_context('brentq', 395, 0, False)
    
    # Passed parameters checking function
    brentq.stypy_localization = localization
    brentq.stypy_type_of_self = None
    brentq.stypy_type_store = module_type_store
    brentq.stypy_function_name = 'brentq'
    brentq.stypy_param_names_list = ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp']
    brentq.stypy_varargs_param_name = None
    brentq.stypy_kwargs_param_name = None
    brentq.stypy_call_defaults = defaults
    brentq.stypy_call_varargs = varargs
    brentq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'brentq', ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'brentq', localization, ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'brentq(...)' code ##################

    str_187236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, (-1)), 'str', '\n    Find a root of a function in a bracketing interval using Brent\'s method.\n\n    Uses the classic Brent\'s method to find a zero of the function `f` on\n    the sign changing interval [a , b].  Generally considered the best of the\n    rootfinding routines here.  It is a safe version of the secant method that\n    uses inverse quadratic extrapolation.  Brent\'s method combines root\n    bracketing, interval bisection, and inverse quadratic interpolation.  It is\n    sometimes known as the van Wijngaarden-Dekker-Brent method.  Brent (1973)\n    claims convergence is guaranteed for functions computable within [a,b].\n\n    [Brent1973]_ provides the classic description of the algorithm.  Another\n    description can be found in a recent edition of Numerical Recipes, including\n    [PressEtal1992]_.  Another description is at\n    http://mathworld.wolfram.com/BrentsMethod.html.  It should be easy to\n    understand the algorithm just by reading our code.  Our code diverges a bit\n    from standard presentations: we choose a different formula for the\n    extrapolation step.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number.  The function :math:`f`\n        must be continuous, and :math:`f(a)` and :math:`f(b)` must\n        have opposite signs.\n    a : number\n        One end of the bracketing interval :math:`[a, b]`.\n    b : number\n        The other end of the bracketing interval :math:`[a, b]`.\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be nonnegative. For nice functions, Brent\'s\n        method will often satisfy the above condition with ``xtol/2``\n        and ``rtol/2``. [Brent1973]_\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``. For nice functions, Brent\'s\n        method will often satisfy the above condition with ``xtol/2``\n        and ``rtol/2``. [Brent1973]_\n    maxiter : number, optional\n        if convergence is not achieved in maxiter iterations, an error is\n        raised.  Must be >= 0.\n    args : tuple, optional\n        containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned.  If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a RootResults object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn\'t converge.\n\n    Returns\n    -------\n    x0 : float\n        Zero of `f` between `a` and `b`.\n    r : RootResults (present if ``full_output = True``)\n        Object containing information about the convergence.  In particular,\n        ``r.converged`` is True if the routine converged.\n\n    See Also\n    --------\n    multivariate local optimizers\n      `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, `fmin_ncg`\n    nonlinear least squares minimizer\n      `leastsq`\n    constrained multivariate optimizers\n      `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`\n    global optimizers\n      `basinhopping`, `brute`, `differential_evolution`\n    local scalar minimizers\n      `fminbound`, `brent`, `golden`, `bracket`\n    n-dimensional root-finding\n      `fsolve`\n    one-dimensional root-finding\n      `brenth`, `ridder`, `bisect`, `newton`\n    scalar fixed-point finder\n      `fixed_point`\n\n    Notes\n    -----\n    `f` must be continuous.  f(a) and f(b) must have opposite signs.\n\n    Examples\n    --------\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.brentq(f, -2, 0)\n    >>> root\n    -1.0\n\n    >>> root = optimize.brentq(f, 0, 2)\n    >>> root\n    1.0\n\n    References\n    ----------\n    .. [Brent1973]\n       Brent, R. P.,\n       *Algorithms for Minimization Without Derivatives*.\n       Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.\n\n    .. [PressEtal1992]\n       Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.\n       *Numerical Recipes in FORTRAN: The Art of Scientific Computing*, 2nd ed.\n       Cambridge, England: Cambridge University Press, pp. 352-355, 1992.\n       Section 9.3:  "Van Wijngaarden-Dekker-Brent Method."\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 513)
    # Getting the type of 'tuple' (line 513)
    tuple_187237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 28), 'tuple')
    # Getting the type of 'args' (line 513)
    args_187238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 22), 'args')
    
    (may_be_187239, more_types_in_union_187240) = may_not_be_subtype(tuple_187237, args_187238)

    if may_be_187239:

        if more_types_in_union_187240:
            # Runtime conditional SSA (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'args', remove_subtype_from_union(args_187238, tuple))
        
        # Assigning a Tuple to a Name (line 514):
        
        # Assigning a Tuple to a Name (line 514):
        
        # Obtaining an instance of the builtin type 'tuple' (line 514)
        tuple_187241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 514)
        # Adding element type (line 514)
        # Getting the type of 'args' (line 514)
        args_187242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 16), tuple_187241, args_187242)
        
        # Assigning a type to the variable 'args' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'args', tuple_187241)

        if more_types_in_union_187240:
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'xtol' (line 515)
    xtol_187243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 7), 'xtol')
    int_187244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 15), 'int')
    # Applying the binary operator '<=' (line 515)
    result_le_187245 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 7), '<=', xtol_187243, int_187244)
    
    # Testing the type of an if condition (line 515)
    if_condition_187246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 4), result_le_187245)
    # Assigning a type to the variable 'if_condition_187246' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'if_condition_187246', if_condition_187246)
    # SSA begins for if statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 516)
    # Processing the call arguments (line 516)
    str_187248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 25), 'str', 'xtol too small (%g <= 0)')
    # Getting the type of 'xtol' (line 516)
    xtol_187249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 54), 'xtol', False)
    # Applying the binary operator '%' (line 516)
    result_mod_187250 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 25), '%', str_187248, xtol_187249)
    
    # Processing the call keyword arguments (line 516)
    kwargs_187251 = {}
    # Getting the type of 'ValueError' (line 516)
    ValueError_187247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 516)
    ValueError_call_result_187252 = invoke(stypy.reporting.localization.Localization(__file__, 516, 14), ValueError_187247, *[result_mod_187250], **kwargs_187251)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 516, 8), ValueError_call_result_187252, 'raise parameter', BaseException)
    # SSA join for if statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rtol' (line 517)
    rtol_187253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 7), 'rtol')
    # Getting the type of '_rtol' (line 517)
    _rtol_187254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 14), '_rtol')
    # Applying the binary operator '<' (line 517)
    result_lt_187255 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), '<', rtol_187253, _rtol_187254)
    
    # Testing the type of an if condition (line 517)
    if_condition_187256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), result_lt_187255)
    # Assigning a type to the variable 'if_condition_187256' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_187256', if_condition_187256)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 518)
    # Processing the call arguments (line 518)
    str_187258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 25), 'str', 'rtol too small (%g < %g)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 518)
    tuple_187259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 518)
    # Adding element type (line 518)
    # Getting the type of 'rtol' (line 518)
    rtol_187260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 55), 'rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 55), tuple_187259, rtol_187260)
    # Adding element type (line 518)
    # Getting the type of '_rtol' (line 518)
    _rtol_187261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 61), '_rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 55), tuple_187259, _rtol_187261)
    
    # Applying the binary operator '%' (line 518)
    result_mod_187262 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 25), '%', str_187258, tuple_187259)
    
    # Processing the call keyword arguments (line 518)
    kwargs_187263 = {}
    # Getting the type of 'ValueError' (line 518)
    ValueError_187257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 518)
    ValueError_call_result_187264 = invoke(stypy.reporting.localization.Localization(__file__, 518, 14), ValueError_187257, *[result_mod_187262], **kwargs_187263)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 518, 8), ValueError_call_result_187264, 'raise parameter', BaseException)
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 519):
    
    # Assigning a Call to a Name (line 519):
    
    # Call to _brentq(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'f' (line 519)
    f_187267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), 'f', False)
    # Getting the type of 'a' (line 519)
    a_187268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 25), 'a', False)
    # Getting the type of 'b' (line 519)
    b_187269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 27), 'b', False)
    # Getting the type of 'xtol' (line 519)
    xtol_187270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 29), 'xtol', False)
    # Getting the type of 'rtol' (line 519)
    rtol_187271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 34), 'rtol', False)
    # Getting the type of 'maxiter' (line 519)
    maxiter_187272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 39), 'maxiter', False)
    # Getting the type of 'args' (line 519)
    args_187273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 47), 'args', False)
    # Getting the type of 'full_output' (line 519)
    full_output_187274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 52), 'full_output', False)
    # Getting the type of 'disp' (line 519)
    disp_187275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 64), 'disp', False)
    # Processing the call keyword arguments (line 519)
    kwargs_187276 = {}
    # Getting the type of '_zeros' (line 519)
    _zeros_187265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), '_zeros', False)
    # Obtaining the member '_brentq' of a type (line 519)
    _brentq_187266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), _zeros_187265, '_brentq')
    # Calling _brentq(args, kwargs) (line 519)
    _brentq_call_result_187277 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), _brentq_187266, *[f_187267, a_187268, b_187269, xtol_187270, rtol_187271, maxiter_187272, args_187273, full_output_187274, disp_187275], **kwargs_187276)
    
    # Assigning a type to the variable 'r' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'r', _brentq_call_result_187277)
    
    # Call to results_c(...): (line 520)
    # Processing the call arguments (line 520)
    # Getting the type of 'full_output' (line 520)
    full_output_187279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 21), 'full_output', False)
    # Getting the type of 'r' (line 520)
    r_187280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 34), 'r', False)
    # Processing the call keyword arguments (line 520)
    kwargs_187281 = {}
    # Getting the type of 'results_c' (line 520)
    results_c_187278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 11), 'results_c', False)
    # Calling results_c(args, kwargs) (line 520)
    results_c_call_result_187282 = invoke(stypy.reporting.localization.Localization(__file__, 520, 11), results_c_187278, *[full_output_187279, r_187280], **kwargs_187281)
    
    # Assigning a type to the variable 'stypy_return_type' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type', results_c_call_result_187282)
    
    # ################# End of 'brentq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'brentq' in the type store
    # Getting the type of 'stypy_return_type' (line 395)
    stypy_return_type_187283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187283)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'brentq'
    return stypy_return_type_187283

# Assigning a type to the variable 'brentq' (line 395)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 0), 'brentq', brentq)

@norecursion
def brenth(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 523)
    tuple_187284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 523)
    
    # Getting the type of '_xtol' (line 524)
    _xtol_187285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), '_xtol')
    # Getting the type of '_rtol' (line 524)
    _rtol_187286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 28), '_rtol')
    # Getting the type of '_iter' (line 524)
    _iter_187287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 43), '_iter')
    # Getting the type of 'False' (line 525)
    False_187288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'False')
    # Getting the type of 'True' (line 525)
    True_187289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 35), 'True')
    defaults = [tuple_187284, _xtol_187285, _rtol_187286, _iter_187287, False_187288, True_187289]
    # Create a new context for function 'brenth'
    module_type_store = module_type_store.open_function_context('brenth', 523, 0, False)
    
    # Passed parameters checking function
    brenth.stypy_localization = localization
    brenth.stypy_type_of_self = None
    brenth.stypy_type_store = module_type_store
    brenth.stypy_function_name = 'brenth'
    brenth.stypy_param_names_list = ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp']
    brenth.stypy_varargs_param_name = None
    brenth.stypy_kwargs_param_name = None
    brenth.stypy_call_defaults = defaults
    brenth.stypy_call_varargs = varargs
    brenth.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'brenth', ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'brenth', localization, ['f', 'a', 'b', 'args', 'xtol', 'rtol', 'maxiter', 'full_output', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'brenth(...)' code ##################

    str_187290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, (-1)), 'str', "Find root of f in [a,b].\n\n    A variation on the classic Brent routine to find a zero of the function f\n    between the arguments a and b that uses hyperbolic extrapolation instead of\n    inverse quadratic extrapolation. There was a paper back in the 1980's ...\n    f(a) and f(b) cannot have the same signs. Generally on a par with the\n    brent routine, but not as heavily tested.  It is a safe version of the\n    secant method that uses hyperbolic extrapolation. The version here is by\n    Chuck Harris.\n\n    Parameters\n    ----------\n    f : function\n        Python function returning a number.  f must be continuous, and f(a) and\n        f(b) must have opposite signs.\n    a : number\n        One end of the bracketing interval [a,b].\n    b : number\n        The other end of the bracketing interval [a,b].\n    xtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter must be nonnegative. As with `brentq`, for nice\n        functions the method will often satisfy the above condition\n        with ``xtol/2`` and ``rtol/2``.\n    rtol : number, optional\n        The computed root ``x0`` will satisfy ``np.allclose(x, x0,\n        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The\n        parameter cannot be smaller than its default value of\n        ``4*np.finfo(float).eps``. As with `brentq`, for nice functions\n        the method will often satisfy the above condition with\n        ``xtol/2`` and ``rtol/2``.\n    maxiter : number, optional\n        if convergence is not achieved in maxiter iterations, an error is\n        raised.  Must be >= 0.\n    args : tuple, optional\n        containing extra arguments for the function `f`.\n        `f` is called by ``apply(f, (x)+args)``.\n    full_output : bool, optional\n        If `full_output` is False, the root is returned.  If `full_output` is\n        True, the return value is ``(x, r)``, where `x` is the root, and `r` is\n        a RootResults object.\n    disp : bool, optional\n        If True, raise RuntimeError if the algorithm didn't converge.\n\n    Returns\n    -------\n    x0 : float\n        Zero of `f` between `a` and `b`.\n    r : RootResults (present if ``full_output = True``)\n        Object containing information about the convergence.  In particular,\n        ``r.converged`` is True if the routine converged.\n\n    Examples\n    --------\n    >>> def f(x):\n    ...     return (x**2 - 1)\n\n    >>> from scipy import optimize\n\n    >>> root = optimize.brenth(f, -2, 0)\n    >>> root\n    -1.0\n\n    >>> root = optimize.brenth(f, 0, 2)\n    >>> root\n    1.0\n\n    See Also\n    --------\n    fmin, fmin_powell, fmin_cg,\n           fmin_bfgs, fmin_ncg : multivariate local optimizers\n\n    leastsq : nonlinear least squares minimizer\n\n    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers\n\n    basinhopping, differential_evolution, brute : global optimizers\n\n    fminbound, brent, golden, bracket : local scalar minimizers\n\n    fsolve : n-dimensional root-finding\n\n    brentq, brenth, ridder, bisect, newton : one-dimensional root-finding\n\n    fixed_point : scalar fixed-point finder\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 614)
    # Getting the type of 'tuple' (line 614)
    tuple_187291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 28), 'tuple')
    # Getting the type of 'args' (line 614)
    args_187292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 22), 'args')
    
    (may_be_187293, more_types_in_union_187294) = may_not_be_subtype(tuple_187291, args_187292)

    if may_be_187293:

        if more_types_in_union_187294:
            # Runtime conditional SSA (line 614)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'args', remove_subtype_from_union(args_187292, tuple))
        
        # Assigning a Tuple to a Name (line 615):
        
        # Assigning a Tuple to a Name (line 615):
        
        # Obtaining an instance of the builtin type 'tuple' (line 615)
        tuple_187295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 615)
        # Adding element type (line 615)
        # Getting the type of 'args' (line 615)
        args_187296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 16), tuple_187295, args_187296)
        
        # Assigning a type to the variable 'args' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'args', tuple_187295)

        if more_types_in_union_187294:
            # SSA join for if statement (line 614)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'xtol' (line 616)
    xtol_187297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 7), 'xtol')
    int_187298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 15), 'int')
    # Applying the binary operator '<=' (line 616)
    result_le_187299 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 7), '<=', xtol_187297, int_187298)
    
    # Testing the type of an if condition (line 616)
    if_condition_187300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 4), result_le_187299)
    # Assigning a type to the variable 'if_condition_187300' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'if_condition_187300', if_condition_187300)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 617)
    # Processing the call arguments (line 617)
    str_187302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 25), 'str', 'xtol too small (%g <= 0)')
    # Getting the type of 'xtol' (line 617)
    xtol_187303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 54), 'xtol', False)
    # Applying the binary operator '%' (line 617)
    result_mod_187304 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 25), '%', str_187302, xtol_187303)
    
    # Processing the call keyword arguments (line 617)
    kwargs_187305 = {}
    # Getting the type of 'ValueError' (line 617)
    ValueError_187301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 617)
    ValueError_call_result_187306 = invoke(stypy.reporting.localization.Localization(__file__, 617, 14), ValueError_187301, *[result_mod_187304], **kwargs_187305)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 617, 8), ValueError_call_result_187306, 'raise parameter', BaseException)
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rtol' (line 618)
    rtol_187307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 7), 'rtol')
    # Getting the type of '_rtol' (line 618)
    _rtol_187308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 14), '_rtol')
    # Applying the binary operator '<' (line 618)
    result_lt_187309 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 7), '<', rtol_187307, _rtol_187308)
    
    # Testing the type of an if condition (line 618)
    if_condition_187310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 4), result_lt_187309)
    # Assigning a type to the variable 'if_condition_187310' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'if_condition_187310', if_condition_187310)
    # SSA begins for if statement (line 618)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 619)
    # Processing the call arguments (line 619)
    str_187312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 25), 'str', 'rtol too small (%g < %g)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 619)
    tuple_187313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 619)
    # Adding element type (line 619)
    # Getting the type of 'rtol' (line 619)
    rtol_187314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 55), 'rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 55), tuple_187313, rtol_187314)
    # Adding element type (line 619)
    # Getting the type of '_rtol' (line 619)
    _rtol_187315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 61), '_rtol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 55), tuple_187313, _rtol_187315)
    
    # Applying the binary operator '%' (line 619)
    result_mod_187316 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 25), '%', str_187312, tuple_187313)
    
    # Processing the call keyword arguments (line 619)
    kwargs_187317 = {}
    # Getting the type of 'ValueError' (line 619)
    ValueError_187311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 619)
    ValueError_call_result_187318 = invoke(stypy.reporting.localization.Localization(__file__, 619, 14), ValueError_187311, *[result_mod_187316], **kwargs_187317)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 619, 8), ValueError_call_result_187318, 'raise parameter', BaseException)
    # SSA join for if statement (line 618)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 620):
    
    # Assigning a Call to a Name (line 620):
    
    # Call to _brenth(...): (line 620)
    # Processing the call arguments (line 620)
    # Getting the type of 'f' (line 620)
    f_187321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 23), 'f', False)
    # Getting the type of 'a' (line 620)
    a_187322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 25), 'a', False)
    # Getting the type of 'b' (line 620)
    b_187323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 28), 'b', False)
    # Getting the type of 'xtol' (line 620)
    xtol_187324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 31), 'xtol', False)
    # Getting the type of 'rtol' (line 620)
    rtol_187325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 37), 'rtol', False)
    # Getting the type of 'maxiter' (line 620)
    maxiter_187326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 43), 'maxiter', False)
    # Getting the type of 'args' (line 620)
    args_187327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'args', False)
    # Getting the type of 'full_output' (line 620)
    full_output_187328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 58), 'full_output', False)
    # Getting the type of 'disp' (line 620)
    disp_187329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 71), 'disp', False)
    # Processing the call keyword arguments (line 620)
    kwargs_187330 = {}
    # Getting the type of '_zeros' (line 620)
    _zeros_187319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), '_zeros', False)
    # Obtaining the member '_brenth' of a type (line 620)
    _brenth_187320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 8), _zeros_187319, '_brenth')
    # Calling _brenth(args, kwargs) (line 620)
    _brenth_call_result_187331 = invoke(stypy.reporting.localization.Localization(__file__, 620, 8), _brenth_187320, *[f_187321, a_187322, b_187323, xtol_187324, rtol_187325, maxiter_187326, args_187327, full_output_187328, disp_187329], **kwargs_187330)
    
    # Assigning a type to the variable 'r' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'r', _brenth_call_result_187331)
    
    # Call to results_c(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'full_output' (line 621)
    full_output_187333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 21), 'full_output', False)
    # Getting the type of 'r' (line 621)
    r_187334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 34), 'r', False)
    # Processing the call keyword arguments (line 621)
    kwargs_187335 = {}
    # Getting the type of 'results_c' (line 621)
    results_c_187332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'results_c', False)
    # Calling results_c(args, kwargs) (line 621)
    results_c_call_result_187336 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), results_c_187332, *[full_output_187333, r_187334], **kwargs_187335)
    
    # Assigning a type to the variable 'stypy_return_type' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'stypy_return_type', results_c_call_result_187336)
    
    # ################# End of 'brenth(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'brenth' in the type store
    # Getting the type of 'stypy_return_type' (line 523)
    stypy_return_type_187337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187337)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'brenth'
    return stypy_return_type_187337

# Assigning a type to the variable 'brenth' (line 523)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'brenth', brenth)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
